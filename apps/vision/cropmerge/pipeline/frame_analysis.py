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
from cropmerge.features.management_units import segment_management_units


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
    """
    emb = embedder.embed_tiles(bgr, field_mask, rows, cols)
    cells, heat = score_frame(bgr, field_mask, label_map, emb, cfg)
    cells, sres = score_structural_cells(cells, bgr, field_mask, label_map, crop_mask, cfg)

    mucfg = cfg.get("management_units", {})
    if mucfg.get("enabled", True):
        _apply_management_unit_types(cells, bgr, field_mask, label_map, rows, cols, mucfg)

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


def _apply_management_unit_types(
    cells: list[GridCell],
    bgr: np.ndarray,
    field_mask: np.ndarray,
    label_map: np.ndarray,
    rows: int,
    cols: int,
    mucfg: dict,
) -> None:
    """Segment the frame into management units on the SAME rows/cols grid as
    `cells` (so cell.row/cell.col indices line up 1:1) and stamp each cell
    with its unit's type/confidence for cropmerge.anomaly.structural's
    classify_zone_type() gate. Never raises — a segmentation failure just
    leaves cells ungated (identical to pre-management-unit behavior); this
    must never take down frame analysis."""
    if label_map is None:
        return
    try:
        result = segment_management_units(
            bgr,
            field_mask,
            label_map,
            rows=rows,
            cols=cols,
            boundary_threshold=float(mucfg.get("boundary_threshold", 0.25)),
        )
    except Exception:
        return

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
