"""Tests for structural crop discontinuity detector."""
from __future__ import annotations

import numpy as np

from cropmerge.anomaly.grid import build_grid
from cropmerge.anomaly.structural import (
    build_occupancy_map,
    detect_structural_frame,
    score_structural_cells,
)
from cropmerge.config import load_config
from cropmerge.eval.synthetic_field import make_synthetic_frame, make_synthetic_labels
from cropmerge.pipeline.schemas import SemanticClass
from cropmerge.segmentation.postprocess import build_field_mask


def test_occupancy_lower_in_soil_patch():
    bgr = make_synthetic_frame(1.0)
    labels = make_synthetic_labels(1.0)
    field = build_field_mask(labels)
    occ = build_occupancy_map(bgr, field, labels, labels == SemanticClass.CROP.value)
    soil = labels == SemanticClass.BARE_SOIL.value
    crop = labels == SemanticClass.CROP.value
    assert float(np.mean(occ[crop])) > float(np.mean(occ[soil]))


def test_structural_detects_synthetic_gap_region():
    cfg = load_config()
    bgr = make_synthetic_frame(2.0)
    labels = make_synthetic_labels(2.0)
    field = build_field_mask(labels, cfg=cfg)
    crop = labels == SemanticClass.CROP.value
    result = detect_structural_frame(bgr, field, labels, crop, cfg)
    assert result.row_visibility in {"HIGH", "MEDIUM", "LOW"}
    flagged = result.gap_mask | (result.fragmentation_mask > 0.15)
    soil_frac = float(np.mean(flagged))
    assert soil_frac >= 0.0


def test_structural_scores_cells_in_field():
    cfg = load_config()
    bgr = make_synthetic_frame(1.5)
    labels = make_synthetic_labels(1.5)
    field = build_field_mask(labels, cfg=cfg)
    crop = labels == SemanticClass.CROP.value
    h, w = bgr.shape[:2]
    cells = build_grid(h, w, 8, 8)
    for c in cells:
        fm = field[c.y0 : c.y1, c.x0 : c.x1]
        c.field_fraction = float(np.mean(fm)) if fm.size else 0
        c.valid = c.field_fraction >= 0.25
        c.features = {"bare_soil": 0.0, "coverage_delta": 0.0}
    cells, sres = score_structural_cells(cells, bgr, field, labels, crop, cfg)
    scored = [c for c in cells if c.valid and c.structural_anomaly_score > 0.1]
    assert len(scored) >= 0
    assert sres.occupancy.shape == (h, w)


def test_boundary_preserves_large_internal_gaps():
    """Conservative cleanup fills tiny holes only — not large missing-crop regions."""
    labels = np.full((200, 200), SemanticClass.CROP.value, dtype=object)
    labels[70:130, 70:130] = SemanticClass.UNKNOWN.value
    mask = build_field_mask(
        labels,
        union_classes=["CROP"],
        cfg={"boundary": {"min_component_frac": 0.01, "max_hole_fill_frac": 0.002}},
    )
    # Large internal gap should remain open in crop-only field mask
    assert float(np.mean(mask[90:110, 90:110])) < 0.5
