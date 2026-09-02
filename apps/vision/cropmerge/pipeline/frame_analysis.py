"""Single-frame anomaly analysis, extracted from pipeline/processor.py so the
live RTSP worker (cropmerge/live/rtsp_worker.py) can call the exact same
per-frame logic recorded-mode batch analysis uses, rather than forking a
parallel implementation.
"""

from __future__ import annotations

import numpy as np

from cropmerge.anomaly.grid import GridCell
from cropmerge.anomaly.spatial import score_frame
from cropmerge.anomaly.structural import StructuralFrameResult, score_structural_cells
from cropmerge.features.management_units import UnitSegmentationResult, segment_management_units
from cropmerge.features.unit_row_geometry import analyze_management_unit_rows


def analyze_single_frame(
    bgr: np.ndarray,
    field_mask: np.ndarray,
    label_map: np.ndarray,
    crop_mask: np.ndarray | None,
    embedder,
    cfg: dict,
    rows: int,
    cols: int,
    quality_weight: float = 1.0,
) -> tuple[list[GridCell], np.ndarray, StructuralFrameResult]:
    """Run appearance + structural anomaly scoring for one frame.

    Returns (cells, heatmap, structural_result). Quality-weight scaling
    (dimming low-quality frames' contribution) is applied here identically
    to the batch pipeline's per-frame loop.

    Management units are computed BEFORE structural scoring (not after) so
    score_structural_cells can use a per-unit local occupancy baseline
    (cropmerge.anomaly.structural._local_unit_baselines) instead of only a
    whole-field one -- the earlier per-cell type stamp alone (which gates
    classify_zone_type()'s output label) doesn't fix the underlying
    baseline arithmetic a residue block next to a crop block was skewing.
    """
    emb = embedder.embed_tiles(bgr, field_mask, rows, cols)
    cells, heat = score_frame(bgr, field_mask, label_map, emb, cfg)

    mucfg = cfg.get("management_units", {})
    management_result: UnitSegmentationResult | None = None
    unit_row_angles: dict[int, float] = {}
    if mucfg.get("enabled", True):
        management_result = _compute_management_units(bgr, field_mask, label_map, rows, cols, mucfg)
        if management_result is not None and crop_mask is not None:
            unit_row_angles = _compute_unit_row_angles(crop_mask, field_mask, management_result)

    cells, sres = score_structural_cells(
        cells, bgr, field_mask, label_map, crop_mask, cfg, management_result, unit_row_angles
    )

    if management_result is not None:
        _stamp_cell_unit_types(cells, management_result)

    for c in cells:
        c.anomaly_score = float(max(c.appearance_anomaly_score, c.structural_anomaly_score))
        c.row_visibility = sres.row_visibility

    if quality_weight < 1.0:
        heat = heat * quality_weight
        for c in cells:
            c.anomaly_score *= quality_weight
            c.appearance_anomaly_score *= quality_weight
            c.structural_anomaly_score *= quality_weight

    return cells, heat, sres


def _compute_management_units(
    bgr: np.ndarray,
    field_mask: np.ndarray,
    label_map: np.ndarray,
    rows: int,
    cols: int,
    mucfg: dict,
) -> UnitSegmentationResult | None:
    """Segment the frame into management units on the SAME rows/cols grid as
    `cells` (so cell.row/cell.col indices line up 1:1). Never raises — a
    segmentation failure just returns None (identical to pre-management-unit
    behavior downstream); this must never take down frame analysis."""
    if label_map is None:
        return None
    try:
        return segment_management_units(
            bgr,
            field_mask,
            label_map,
            rows=rows,
            cols=cols,
            boundary_threshold=float(mucfg.get("boundary_threshold", 0.25)),
        )
    except Exception:
        return None


def _compute_unit_row_angles(
    crop_mask: np.ndarray,
    field_mask: np.ndarray,
    management_result: UnitSegmentationResult,
) -> dict[int, float]:
    """Per-unit dominant row angle, ACTIVE_CROP units only (see
    cropmerge.features.unit_row_geometry's own gating) -- used by
    score_structural_cells to sample gap continuity along the CORRECT row
    direction for each unit instead of one blended whole-frame angle. Never
    raises: a failure here just means no unit gets a per-unit angle, and
    every cell falls back to the whole-frame angle exactly as before."""
    try:
        row_results = analyze_management_unit_rows(crop_mask, field_mask, management_result)
    except Exception:
        return {}
    return {
        r.unit_id: r.row_geometry.dominant_angle_deg
        for r in row_results
        if r.row_geometry is not None and r.row_geometry.dominant_angle_deg is not None
    }


def _stamp_cell_unit_types(cells: list[GridCell], result: UnitSegmentationResult) -> None:
    """Stamp each cell with its unit's type/confidence for
    cropmerge.anomaly.structural's classify_zone_type() gate."""
    unit_by_id = {u.unit_id: u for u in result.units}
    for cell in cells:
        if cell.row >= len(result.cell_unit_ids) or cell.col >= len(result.cell_unit_ids[cell.row]):
            continue
        unit_id = result.cell_unit_ids[cell.row][cell.col]
        unit = unit_by_id.get(unit_id)
        if unit is None:
            continue
        cell.management_unit_type = unit.unit_type.value
        cell.management_unit_confidence = unit.confidence
