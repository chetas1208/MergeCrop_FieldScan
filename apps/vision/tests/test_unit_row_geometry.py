"""Synthetic tests for cropmerge.features.unit_row_geometry -- Phase 4 of
the "Make FarmTech Load-Bearing" campaign: per-management-unit row
detection, computed ONLY for confident ACTIVE_CROP units."""
from __future__ import annotations

import numpy as np

from cropmerge.features.management_units import segment_management_units
from cropmerge.features.unit_row_geometry import analyze_management_unit_rows


def _left_crop_right_residue_field(h: int = 240, w: int = 240, n_stripes: int = 5, stripe_w: int = 6):
    """Left half: a green field with real vertical crop-row stripes in
    `crop_mask` (the fine-grained plant-presence signal) -- ACTIVE_CROP.
    Right half: textured tan residue, no crop rows -- RESIDUE_STUBBLE.
    Returns (bgr, field_mask, label_map, crop_mask)."""
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    half = w // 2
    bgr[:, :half] = (30, 170, 30)  # BGR green
    for y in range(h):
        shade = 1.0 if (y // 3) % 2 == 0 else 0.55
        bgr[y, half:] = tuple(int(c * shade) for c in (90, 140, 180))  # striped tan residue

    field_mask = np.ones((h, w), dtype=bool)
    label_map = np.full((h, w), "CROP", dtype=object)
    label_map[:, half:] = "BARE_SOIL"

    crop_mask = np.zeros((h, w), dtype=bool)
    margin = 15
    usable = half - 2 * margin
    spacing = usable / n_stripes
    for i in range(n_stripes):
        cx = int(margin + spacing * i + spacing / 2)
        x0 = max(0, cx - stripe_w // 2)
        x1 = min(half, cx + stripe_w // 2)
        crop_mask[margin : h - margin, x0:x1] = True

    return bgr, field_mask, label_map, crop_mask


def test_residue_unit_is_skipped_never_given_a_row_angle():
    bgr, field, label, crop_mask = _left_crop_right_residue_field()
    seg = segment_management_units(bgr, field, label, rows=6, cols=6)

    results = analyze_management_unit_rows(crop_mask, field, seg)

    by_id = {r.unit_id: r for r in results}
    for unit in seg.units:
        if unit.dominant_residue_class == "likely_residue":
            r = by_id[unit.unit_id]
            assert r.row_geometry is None
            assert r.skipped_reason is not None
            assert "ACTIVE_CROP" in r.skipped_reason


def test_active_crop_unit_recovers_row_geometry():
    bgr, field, label, crop_mask = _left_crop_right_residue_field()
    seg = segment_management_units(bgr, field, label, rows=6, cols=6)

    results = analyze_management_unit_rows(crop_mask, field, seg)

    crop_results = [r for r in results if r.row_geometry is not None]
    assert crop_results, "expected at least one ACTIVE_CROP unit to yield row geometry"
    for r in crop_results:
        assert r.skipped_reason is None
        assert r.row_geometry.dominant_angle_deg is not None
        assert r.row_geometry.num_consistent_rows >= 1


def test_results_are_returned_in_same_order_and_count_as_units():
    bgr, field, label, crop_mask = _left_crop_right_residue_field()
    seg = segment_management_units(bgr, field, label, rows=6, cols=6)

    results = analyze_management_unit_rows(crop_mask, field, seg)

    assert [r.unit_id for r in results] == [u.unit_id for u in seg.units]


def test_low_confidence_threshold_can_be_tightened_to_skip_more():
    bgr, field, label, crop_mask = _left_crop_right_residue_field()
    seg = segment_management_units(bgr, field, label, rows=6, cols=6)

    lenient = analyze_management_unit_rows(crop_mask, field, seg, min_confidence=0.0)
    strict = analyze_management_unit_rows(crop_mask, field, seg, min_confidence=1.01)

    lenient_computed = sum(1 for r in lenient if r.row_geometry is not None)
    strict_computed = sum(1 for r in strict if r.row_geometry is not None)
    assert strict_computed == 0
    assert lenient_computed >= strict_computed


def test_empty_field_produces_no_row_geometry():
    h, w = 100, 100
    bgr = np.full((h, w, 3), (90, 140, 180), dtype=np.uint8)
    field = np.ones((h, w), dtype=bool)
    label = np.full((h, w), "BARE_SOIL", dtype=object)
    crop_mask = np.zeros((h, w), dtype=bool)

    seg = segment_management_units(bgr, field, label, rows=4, cols=4)
    results = analyze_management_unit_rows(crop_mask, field, seg)

    assert all(r.row_geometry is None for r in results)
