from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cropmerge.config import load_config
from cropmerge.pipeline.farmtech_shadow import compute_farmtech_observation
from cropmerge.pipeline.processor import FieldTriageProcessor


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0):
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert wr.isOpened()
    for f in frames:
        wr.write(f)
    wr.release()


def test_compute_farmtech_observation_none_without_field():
    bgr = np.zeros((50, 50, 3), dtype=np.uint8)
    field = np.zeros((50, 50), dtype=bool)
    assert compute_farmtech_observation(bgr, field, None, 0.0, None, None) is None


def test_compute_farmtech_observation_canopy_only_without_crop_mask():
    bgr = np.full((50, 50, 3), 120, dtype=np.uint8)
    field = np.ones((50, 50), dtype=bool)
    result = compute_farmtech_observation(bgr, field, None, 0.1, np.full((50, 50), 0.5), None)
    assert result is not None
    assert result.mode == "CANOPY_ONLY"
    assert result.row_geometry is None
    assert result.structure.soil_fraction == 0.1


def test_compute_farmtech_observation_populates_residue_classification():
    bgr = np.zeros((50, 50, 3), dtype=np.uint8)
    bgr[:] = (90, 140, 180)  # tan/brown, smooth -> likely_bare_soil
    field = np.ones((50, 50), dtype=bool)
    label = np.full((50, 50), "BARE_SOIL", dtype=object)

    result = compute_farmtech_observation(bgr, field, None, 1.0, None, None, label_map=label)

    assert result is not None
    assert result.structure.residue_classification == "likely_bare_soil"
    assert result.structure.residue_confidence_note is not None


def test_compute_farmtech_observation_residue_none_without_soil_area():
    bgr = np.full((50, 50, 3), 120, dtype=np.uint8)
    field = np.ones((50, 50), dtype=bool)
    label = np.full((50, 50), "CROP", dtype=object)  # no BARE_SOIL pixels

    result = compute_farmtech_observation(bgr, field, None, 0.0, None, None, label_map=label)

    assert result is not None
    assert result.structure.residue_classification is None


def test_compute_farmtech_observation_never_raises_on_bad_input():
    # Mismatched shapes should be caught and return None, not propagate.
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    field = np.ones((50, 50), dtype=bool)
    assert compute_farmtech_observation(bgr, field, None, 0.0, None, None) is None


def _crop_row_stripe_field(h: int = 240, w: int = 120, n_stripes: int = 4, stripe_half_width: int = 5):
    """Vertical crop-row stripes over a soil background -- enough real row
    structure for unit_row_geometry to reach confident ACTIVE_CROP geometry
    (see tests/test_inter_row_vegetation.py, same fixture shape)."""
    margin = 10
    usable = w - 2 * margin
    spacing = usable / n_stripes
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:] = (90, 140, 180)
    crop_mask = np.zeros((h, w), dtype=bool)
    for i in range(n_stripes):
        cx = int(margin + spacing * i + spacing / 2)
        x0, x1 = max(0, cx - stripe_half_width), min(w, cx + stripe_half_width)
        bgr[margin : h - margin, x0:x1] = (30, 170, 30)
        crop_mask[margin : h - margin, x0:x1] = True
    field = np.ones((h, w), dtype=bool)
    label = np.full((h, w), "CROP", dtype=object)
    return bgr, field, label, crop_mask


def test_visible_inter_row_vegetation_populated_with_confident_row_geometry():
    bgr, field, label, crop_mask = _crop_row_stripe_field()

    result = compute_farmtech_observation(bgr, field, crop_mask, 0.0, None, None, label_map=label)

    assert result is not None
    assert result.structure.visible_inter_row_vegetation_fraction is not None
    assert 0.0 <= result.structure.visible_inter_row_vegetation_fraction <= 1.0


def test_visible_inter_row_vegetation_none_without_crop_mask():
    bgr, field, label, _crop_mask = _crop_row_stripe_field()

    result = compute_farmtech_observation(bgr, field, None, 0.0, None, None, label_map=label)

    assert result is not None
    assert result.structure.visible_inter_row_vegetation_fraction is None


def test_visible_inter_row_vegetation_none_without_confident_row_geometry():
    # Uniform bare soil -- no crop rows anywhere, no confident geometry.
    bgr = np.full((50, 50, 3), (90, 140, 180), dtype=np.uint8)
    field = np.ones((50, 50), dtype=bool)
    label = np.full((50, 50), "BARE_SOIL", dtype=object)
    crop_mask = np.zeros((50, 50), dtype=bool)

    result = compute_farmtech_observation(bgr, field, crop_mask, 1.0, None, None, label_map=label)

    assert result is not None
    assert result.structure.visible_inter_row_vegetation_fraction is None


def _run_synthetic_field(tmp_path: Path, monkeypatch, shadow_enabled: bool):
    from scripts.generate_demo_video import make_frame

    if shadow_enabled:
        monkeypatch.setenv("CROP_MERGE_FARMTECH_SHADOW", "true")
    else:
        monkeypatch.delenv("CROP_MERGE_FARMTECH_SHADOW", raising=False)

    path = tmp_path / "field.mp4"
    frames = [make_frame(i / 10.0) for i in range(30)]
    _write_video(path, frames, fps=10)
    out = tmp_path / "out"
    cfg = load_config()
    cfg["temporal"]["min_persistent_frames"] = 2
    cfg["anomaly"]["medium_threshold"] = 0.15
    return FieldTriageProcessor(cfg).process(
        path, out, sample_fps=5, max_frames=15, segmentation_backend="heuristic", skip_dino=True,
    )


def test_farmtech_shadow_disabled_by_default(tmp_path: Path, monkeypatch):
    report = _run_synthetic_field(tmp_path, monkeypatch, shadow_enabled=False)
    assert all(q.farm_tech is None for q in report.frame_quality)
    camel = report.to_camel_dict()
    assert all(fq["farmTech"] is None for fq in camel["frameQuality"])


def test_farmtech_shadow_enabled_populates_observations_without_changing_zones(
    tmp_path: Path, monkeypatch
):
    baseline = _run_synthetic_field(tmp_path, monkeypatch, shadow_enabled=False)
    shadow = _run_synthetic_field(tmp_path, monkeypatch, shadow_enabled=True)

    if baseline.field.detected:
        assert any(q.farm_tech is not None for q in shadow.frame_quality)

    # The core requirement: shadow mode must be purely additive. Every
    # existing decision (zone count/type/priority/score) must be identical
    # whether or not shadow computation ran.
    assert len(baseline.inspection_zones) == len(shadow.inspection_zones)
    for a, b in zip(baseline.inspection_zones, shadow.inspection_zones):
        assert a.primary_type == b.primary_type
        assert a.review_priority == b.review_priority
        assert a.anomaly_score == b.anomaly_score
        assert a.review_score == b.review_score
    assert baseline.field.mean_crop_coverage == shadow.field.mean_crop_coverage
    assert baseline.field.mean_bare_soil == shadow.field.mean_bare_soil
