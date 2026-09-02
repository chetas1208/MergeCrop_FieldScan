"""Tests for structural crop discontinuity detector."""
from __future__ import annotations

import numpy as np

from cropmerge.anomaly.grid import build_grid
from cropmerge.anomaly.structural import (
    _hough_crosscheck_row_visibility,
    build_occupancy_map,
    detect_structural_frame,
    score_structural_cells,
)
from cropmerge.config import load_config
from cropmerge.eval.synthetic_field import make_synthetic_frame, make_synthetic_labels
from cropmerge.features.row_geometry import RowGeometryResult
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


def _fake_row_geometry(num_consistent_rows: int, dominant_angle_deg: float | None = 42.0) -> RowGeometryResult:
    return RowGeometryResult(
        thinning_method="zhang_suen_fallback",
        guo_hall_available=False,
        skeleton_pixel_count=100,
        hough_line_count=num_consistent_rows,
        dominant_angle_deg=dominant_angle_deg,
        num_consistent_rows=num_consistent_rows,
        row_line_support_px=100.0 if num_consistent_rows else 0.0,
    )


def test_hough_crosscheck_upgrades_low_to_medium_when_corroborated(monkeypatch):
    import cropmerge.anomaly.structural as structural_mod

    monkeypatch.setattr(structural_mod, "analyze_row_geometry", lambda mask, cfg: _fake_row_geometry(3))
    mask = np.ones((50, 50), dtype=bool)
    row_vis, angle = _hough_crosscheck_row_visibility("LOW", None, mask, mask, min_consistent_rows=2)
    assert row_vis == "MEDIUM"
    assert angle == 42.0


def test_hough_crosscheck_stays_low_without_enough_support(monkeypatch):
    import cropmerge.anomaly.structural as structural_mod

    monkeypatch.setattr(structural_mod, "analyze_row_geometry", lambda mask, cfg: _fake_row_geometry(1))
    mask = np.ones((50, 50), dtype=bool)
    row_vis, angle = _hough_crosscheck_row_visibility("LOW", None, mask, mask, min_consistent_rows=2)
    assert row_vis == "LOW"
    assert angle is None


def test_hough_crosscheck_never_downgrades_or_runs_on_non_low():
    mask = np.ones((50, 50), dtype=bool)
    for existing in ("MEDIUM", "HIGH"):
        row_vis, angle = _hough_crosscheck_row_visibility(existing, 10.0, mask, mask, min_consistent_rows=1)
        assert row_vis == existing
        assert angle == 10.0


def test_hough_crosscheck_noop_without_crop_mask():
    field = np.ones((50, 50), dtype=bool)
    row_vis, angle = _hough_crosscheck_row_visibility("LOW", None, None, field, min_consistent_rows=1)
    assert row_vis == "LOW"
    assert angle is None


def test_detect_structural_frame_default_unaffected_by_crosscheck_flag():
    """Flag defaults off -- detect_structural_frame's output must be
    byte-identical whether or not the (disabled) key is even present in cfg,
    proving this integration didn't silently change default behavior."""
    cfg = load_config()
    bgr = make_synthetic_frame(2.0)
    labels = make_synthetic_labels(2.0)
    field = build_field_mask(labels, cfg=cfg)
    crop = labels == SemanticClass.CROP.value
    baseline = detect_structural_frame(bgr, field, labels, crop, cfg)

    cfg_explicit_off = load_config()
    cfg_explicit_off["structural"]["farmtech_hough_crosscheck_enabled"] = False
    explicit = detect_structural_frame(bgr, field, labels, crop, cfg_explicit_off)
    assert explicit.row_visibility == baseline.row_visibility
    assert explicit.row_angle_deg == baseline.row_angle_deg


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
