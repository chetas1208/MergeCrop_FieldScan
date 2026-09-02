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
