"""Synthetic, deterministic tests for cropmerge.features.row_geometry.

No GPU/model dependency -- all inputs are hand-constructed numpy masks with
known geometry (parallel vertical/horizontal stripes, a known gap, random
noise) so expected outcomes are derivable without a ground-truth dataset.
"""
from __future__ import annotations

import numpy as np

from cropmerge.features.row_geometry import (
    RowGeometryConfig,
    analyze_row_geometry,
    clean_mask,
    cluster_row_lines,
    guo_hall_available,
    hough_row_lines,
    line_angle_deg,
    robust_median_angle_deg,
    structure_tensor_orientation,
    thin_mask,
)


def _vertical_stripe_mask(h: int = 220, w: int = 220, n_stripes: int = 5, stripe_w: int = 6) -> np.ndarray:
    """Synthetic mask with n_stripes evenly spaced vertical crop rows."""
    mask = np.zeros((h, w), dtype=np.uint8)
    margin = 20
    usable = w - 2 * margin
    spacing = usable / n_stripes
    for i in range(n_stripes):
        cx = int(margin + spacing * i + spacing / 2)
        x0 = max(0, cx - stripe_w // 2)
        x1 = min(w, cx + stripe_w // 2)
        mask[margin:h - margin, x0:x1] = 1
    return mask


def _noise_mask(h: int = 220, w: int = 220, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) > 0.7).astype(np.uint8)


# ---------------------------------------------------------------------------
# Guo-Hall availability + thinning fallback behavior
# ---------------------------------------------------------------------------


def test_guo_hall_availability_matches_environment():
    """This project depends on opencv-python-headless, not opencv-contrib,
    so Guo-Hall thinning is expected to be unavailable in this venv. This
    test documents/enforces that the module reports its actual environment
    rather than silently assuming either way."""
    try:
        import cv2.ximgproc  # noqa: F401

        expected = True
    except ImportError:
        expected = False
    assert guo_hall_available() == expected


def test_thin_mask_reports_method_and_shrinks_pixel_count():
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[15:25, 10:70] = 1  # thick horizontal bar, 10px tall
    skeleton, method = thin_mask(mask)
    assert method in ("guo_hall", "zhang_suen_fallback")
    assert method == ("guo_hall" if guo_hall_available() else "zhang_suen_fallback")
    assert skeleton.dtype == bool
    assert skeleton.shape == mask.shape
    # Thinning must substantially reduce pixel count vs. the thick original,
    # and must not erase the shape entirely.
    assert 0 < int(np.count_nonzero(skeleton)) < int(np.count_nonzero(mask)) / 2


def test_thin_mask_on_empty_mask_stays_empty():
    mask = np.zeros((30, 30), dtype=np.uint8)
    skeleton, _ = thin_mask(mask)
    assert not np.any(skeleton)


# ---------------------------------------------------------------------------
# Robust circular-median angle
# ---------------------------------------------------------------------------


def test_robust_median_angle_handles_wraparound():
    # Angles clustered near the 0/180 boundary: a plain numeric median would
    # be pulled toward ~90 (wrong); the circular median must stay near 0/180.
    angles = np.array([1.0, 2.0, 179.0, 178.0, 3.0])
    med = robust_median_angle_deg(angles)
    # Circular distance from med to 0 (mod 180) should be small.
    dist_to_zero = min(med, 180.0 - med)
    assert dist_to_zero < 10.0


def test_robust_median_angle_ignores_outlier():
    angles = np.array([45.0, 46.0, 44.0, 45.5, 120.0])  # 120 is an outlier
    med = robust_median_angle_deg(angles)
    assert abs(med - 45.0) < 3.0


def test_robust_median_angle_single_value():
    assert robust_median_angle_deg(np.array([73.0])) == 73.0


# ---------------------------------------------------------------------------
# Hough line detection + clustering on synthetic parallel rows
# ---------------------------------------------------------------------------


def test_hough_detects_lines_on_clean_vertical_stripes():
    mask = _vertical_stripe_mask()
    cleaned = clean_mask(mask)
    skeleton, _ = thin_mask(cleaned)
    lines = hough_row_lines(skeleton)
    assert lines.shape[0] > 0
    assert lines.shape[1] == 4


def test_hough_detects_no_lines_on_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    skeleton, _ = thin_mask(mask)
    lines = hough_row_lines(skeleton)
    assert lines.shape == (0, 4)


def test_cluster_row_lines_recovers_multiple_rows():
    mask = _vertical_stripe_mask(n_stripes=5, stripe_w=6)
    cleaned = clean_mask(mask)
    skeleton, _ = thin_mask(cleaned)
    lines = hough_row_lines(skeleton)
    dominant_angle, clusters = cluster_row_lines(lines)

    assert dominant_angle is not None
    # Vertical stripes -> line angle near 90 degrees (mod 180).
    assert min(abs(dominant_angle - 90.0), abs(dominant_angle - 90.0) % 180) < 15.0
    # Expect roughly one cluster per stripe (allow some merging/splitting
    # slack from Hough segmentation, but must find more than one row).
    assert len(clusters) >= 3
    for c in clusters:
        assert c.support_px > 0
        assert c.line_count >= 1


def test_cluster_row_lines_empty_input():
    dominant, clusters = cluster_row_lines(np.zeros((0, 4)))
    assert dominant is None
    assert clusters == []


def test_line_angle_deg_vertical_and_horizontal():
    assert line_angle_deg(0, 0, 0, 10) == 90.0
    assert line_angle_deg(0, 0, 10, 0) == 0.0


# ---------------------------------------------------------------------------
# Structure-tensor orientation/coherence
# ---------------------------------------------------------------------------


def test_structure_tensor_high_coherence_on_stripes():
    mask = _vertical_stripe_mask()
    angle, coherence = structure_tensor_orientation(mask.astype(np.float32))
    assert angle is not None
    assert 0.0 <= coherence <= 1.0
    assert coherence > 0.3  # strong, consistent orientation signal


def test_structure_tensor_low_coherence_on_noise():
    mask = _noise_mask()
    angle, coherence = structure_tensor_orientation(mask.astype(np.float32))
    assert coherence < 0.3


def test_structure_tensor_none_on_uniform_field():
    flat = np.ones((50, 50), dtype=np.float32)
    angle, coherence = structure_tensor_orientation(flat)
    assert angle is None
    assert coherence == 0.0


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------


def test_analyze_row_geometry_on_clear_parallel_rows():
    mask = _vertical_stripe_mask(n_stripes=5, stripe_w=6)
    result = analyze_row_geometry(mask)

    assert result.thinning_method in ("guo_hall", "zhang_suen_fallback")
    assert result.guo_hall_available == guo_hall_available()
    assert result.skeleton_pixel_count > 0
    assert result.hough_line_count > 0
    assert result.dominant_angle_deg is not None
    assert result.num_consistent_rows >= 3
    assert result.row_line_support_px > 0
    assert result.structure_tensor_coherence > 0.3
    # Two independent orientation estimators should broadly agree on
    # clean, unambiguous synthetic geometry.
    assert result.angle_agreement_deg is not None
    assert result.angle_agreement_deg < 20.0


def test_analyze_row_geometry_on_noise_finds_few_or_no_consistent_rows():
    mask = _noise_mask()
    result = analyze_row_geometry(mask)
    assert result.num_consistent_rows <= 1
    assert result.structure_tensor_coherence < 0.3


def test_analyze_row_geometry_on_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = analyze_row_geometry(mask)
    assert result.dominant_angle_deg is None
    assert result.num_consistent_rows == 0
    assert result.row_line_support_px == 0.0
    assert result.structure_tensor_angle_deg is None
    assert result.structure_tensor_coherence == 0.0


def test_analyze_row_geometry_gap_reduces_hough_support_on_that_row():
    """A single row with a large gap cut out of it should still be
    detected, but with less total segment support than the unbroken
    version -- a coarse proxy for 'a gap affects row-continuity evidence,'
    without asserting on gap *location* (that's the future
    row-continuity-claim logic downstream of this module, not this
    module's job)."""
    full = np.zeros((220, 60), dtype=np.uint8)
    full[20:200, 25:35] = 1  # single vertical row, full length

    gapped = full.copy()
    gapped[90:130, :] = 0  # cut a 40px gap out of the middle

    cfg = RowGeometryConfig(min_row_support_px=1.0)
    r_full = analyze_row_geometry(full, cfg)
    r_gapped = analyze_row_geometry(gapped, cfg)

    assert r_full.row_line_support_px > 0
    assert r_gapped.row_line_support_px > 0
    assert r_gapped.row_line_support_px <= r_full.row_line_support_px
