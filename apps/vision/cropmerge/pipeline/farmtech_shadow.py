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

from cropmerge.features.observability import classify_observability
from cropmerge.features.rgb_indices import excess_green, vari
from cropmerge.features.row_geometry import RowGeometryConfig, analyze_row_geometry
from cropmerge.pipeline.schemas import (
    FarmTechObservation,
    FarmTechRowGeometry,
    FarmTechStructure,
    FarmTechVegetation,
)

log = logging.getLogger("cropmerge.pipeline.farmtech_shadow")

_MIN_ROW_GEOMETRY_MASK_PIXELS = 100


def compute_farmtech_observation(
    bgr: np.ndarray,
    field_mask: np.ndarray | None,
    crop_mask: np.ndarray | None,
    soil_fraction: float,
    occupancy: np.ndarray | None,
    fragmentation_mask: np.ndarray | None,
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
        structure = FarmTechStructure(
            crop_occupancy=crop_occupancy,
            soil_fraction=soil_fraction,
            fragmentation=fragmentation,
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
