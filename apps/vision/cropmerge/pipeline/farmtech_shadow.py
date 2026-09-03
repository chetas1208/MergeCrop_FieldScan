"""Per-observation FarmTech shadow measurements.

Computed and recorded alongside the existing 1 FPS observation timeline
(FrameQuality.farm_tech in cropmerge/pipeline/schemas.py), but MUST NEVER be
used to alter an InspectionZone's type, priority, score, ranking, or any
other existing production decision -- see cropmerge/pipeline/processor.py's
call site, which is gated by CROP_MERGE_FARMTECH_SHADOW (default off) and
wires this in purely additively.

This is Phase 2/3 of the FarmTech integration campaign: FarmTech metrics run
on every usable observation as recorded, non-authoritative shadow evidence,
so that a later validation pass can compare shadow output against real
outcomes before anything here is allowed to affect a result. See
docs/FARMTECH_RESEARCH_BASIS.md and docs/FARMTECH_METHOD_TRACEABILITY.md for
the formulas' basis.
"""

from __future__ import annotations

import logging

import numpy as np

from cropmerge.features.inter_row_vegetation import analyze_inter_row_vegetation
from cropmerge.features.management_units import segment_management_units, unit_footprint_mask
from cropmerge.features.observability import classify_observability
from cropmerge.features.residue_detection import classify_residue_evidence
from cropmerge.features.rgb_indices import excess_green, vari
from cropmerge.features.row_geometry import RowGeometryConfig, analyze_row_geometry
from cropmerge.features.unit_row_geometry import analyze_management_unit_rows
from cropmerge.features.weed_pressure import understory_vegetation_fraction
from cropmerge.pipeline.schemas import (
    FarmTechObservation,
    FarmTechRowGeometry,
    FarmTechStructure,
    FarmTechVegetation,
    SemanticClass,
)

log = logging.getLogger("cropmerge.pipeline.farmtech_shadow")

_MIN_ROW_GEOMETRY_MASK_PIXELS = 100


def _compute_visible_inter_row_vegetation(
    bgr: np.ndarray,
    field: np.ndarray,
    crop_mask: np.ndarray | None,
    label_map: np.ndarray | None,
) -> float | None:
    """Area-weighted mean of analyze_inter_row_vegetation() across every
    confident ACTIVE_CROP management unit this frame. Returns None (never
    fabricated) when label_map/crop_mask are unavailable, no unit reaches
    confident row geometry, or anything here fails -- isolated in its own
    try/except so a failure here degrades only this one optional field,
    never the rest of the FarmTech shadow observation."""
    if label_map is None or crop_mask is None:
        return None
    try:
        seg = segment_management_units(bgr, field, label_map)
        row_results = {
            r.unit_id: r.row_geometry
            for r in analyze_management_unit_rows(crop_mask, field, seg)
            if r.row_geometry is not None
        }
        if not row_results:
            return None

        weighted_sum = 0.0
        weight_total = 0
        for unit in seg.units:
            geometry = row_results.get(unit.unit_id)
            if geometry is None:
                continue
            unit_mask = unit_footprint_mask(field, seg, unit.unit_id)
            result = analyze_inter_row_vegetation(bgr, unit_mask, geometry)
            if result is None:
                continue
            area = int(np.count_nonzero(unit_mask))
            weighted_sum += result.visible_inter_row_vegetation_fraction_of_unit * area
            weight_total += area

        return weighted_sum / weight_total if weight_total > 0 else None
    except Exception:
        log.exception("Visible inter-row vegetation computation failed (non-fatal, skipping)")
        return None


def compute_farmtech_observation(
    bgr: np.ndarray,
    field_mask: np.ndarray | None,
    crop_mask: np.ndarray | None,
    soil_fraction: float,
    occupancy: np.ndarray | None,
    fragmentation_mask: np.ndarray | None,
    label_map: np.ndarray | None = None,
) -> FarmTechObservation | None:
    """Pure, side-effect-free per-frame FarmTech shadow measurement.

    Returns None (never raises) when there isn't enough field to measure, or
    if anything in the computation itself fails -- callers must treat a None
    result the same as "shadow data unavailable for this observation", never
    as an error that should interrupt analysis. This mirrors the
    never-block-the-real-result pattern already used for LLM enrichment and
    GPU cache release elsewhere in this pipeline.
    """
    try:
        if field_mask is None or not np.any(field_mask):
            return None
        field = field_mask.astype(bool)

        exg = excess_green(bgr)
        var = vari(bgr)
        vegetation = FarmTechVegetation(
            exg_mean=float(np.mean(exg[field])),
            vari_mean=float(np.mean(var[field])),
        )

        crop_occupancy = float(np.mean(occupancy[field])) if occupancy is not None else 0.0
        fragmentation = (
            float(np.mean(fragmentation_mask[field])) if fragmentation_mask is not None else 0.0
        )
        understory_fraction: float | None = None
        vegetated_soil_of_field: float | None = None
        residue_classification: str | None = None
        residue_note: str | None = None
        if label_map is not None:
            weed = understory_vegetation_fraction(bgr, field, label_map)
            understory_fraction = weed["understoryVegetationFraction"]
            vegetated_soil_of_field = weed["vegetatedSoilFractionOfField"]

            soil_mask = (label_map == SemanticClass.BARE_SOIL.value) & field
            if np.any(soil_mask):
                residue = classify_residue_evidence(bgr, soil_mask)
                residue_classification = residue.classification.value
                residue_note = residue.confidence_note

        visible_inter_row_fraction = _compute_visible_inter_row_vegetation(bgr, field, crop_mask, label_map)

        structure = FarmTechStructure(
            crop_occupancy=crop_occupancy,
            soil_fraction=soil_fraction,
            fragmentation=fragmentation,
            understory_vegetation_fraction=understory_fraction,
            vegetated_soil_fraction_of_field=vegetated_soil_of_field,
            residue_classification=residue_classification,
            residue_confidence_note=residue_note,
            visible_inter_row_vegetation_fraction=visible_inter_row_fraction,
        )

        row_geometry: FarmTechRowGeometry | None = None
        mode = "CANOPY_ONLY"
        if crop_mask is not None:
            mask = (crop_mask.astype(bool) & field).astype(np.uint8)
            if int(mask.sum()) >= _MIN_ROW_GEOMETRY_MASK_PIXELS:
                geo = analyze_row_geometry(mask, RowGeometryConfig())
                row_geometry = FarmTechRowGeometry(
                    method=geo.thinning_method,
                    orientation_deg=geo.dominant_angle_deg,
                    coherence=geo.structure_tensor_coherence,
                    row_count=geo.num_consistent_rows,
                    support_px=geo.row_line_support_px,
                )
                mode = classify_observability(
                    row_coherence=geo.structure_tensor_coherence,
                    row_line_support_px=geo.row_line_support_px,
                    num_consistent_rows=geo.num_consistent_rows,
                )

        return FarmTechObservation(
            mode=mode,
            vegetation=vegetation,
            row_geometry=row_geometry,
            structure=structure,
        )
    except Exception:
        log.exception("FarmTech shadow observation failed (non-fatal, shadow-only, skipping)")
        return None
