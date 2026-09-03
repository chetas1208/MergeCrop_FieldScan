"""Tests for cropmerge.features.inter_row_vegetation -- row-aware
inter-row vegetation classification, the close-canopy/visible-inter-row-
vegetation slice of the "Make FarmTech Load-Bearing" campaign."""
from __future__ import annotations

import numpy as np

from cropmerge.features.inter_row_vegetation import analyze_inter_row_vegetation, classify_row_band
from cropmerge.features.management_units import UnitType, segment_management_units, unit_footprint_mask
from cropmerge.features.row_geometry import RowGeometryResult, RowLineCluster
from cropmerge.features.unit_row_geometry import analyze_management_unit_rows

_H, _W = 240, 240
_CROP_W = 120
_MARGIN = 10
_N_STRIPES = 4
_STRIPE_HALF_WIDTH = 5
_ROW_BAND_HALF_WIDTH = 6.0  # must be >= _STRIPE_HALF_WIDTH or real stripe edges leak into "inter-row"


def _stripe_x_ranges() -> list[tuple[int, int]]:
    usable = _CROP_W - 2 * _MARGIN
    spacing = usable / _N_STRIPES
    ranges = []
    for i in range(_N_STRIPES):
        cx = int(_MARGIN + spacing * i + spacing / 2)
        ranges.append((max(0, cx - _STRIPE_HALF_WIDTH), min(_CROP_W, cx + _STRIPE_HALF_WIDTH)))
    return ranges


def _crop_rows_next_to_residue(inject_weed_patch: bool):
    """Left block: vertical crop-row stripes over a soil background.
    Right block: residue. Optionally injects a real green "weed" patch
    strictly between stripes 1 and 2 (inter-row space, never touching
    crop_mask) to prove vegetation-in-the-gap is actually detected."""
    stripes = _stripe_x_ranges()
    bgr = np.zeros((_H, _W, 3), dtype=np.uint8)
    bgr[:, :_CROP_W] = (90, 140, 180)  # soil background inside the crop block
    crop_mask = np.zeros((_H, _W), dtype=bool)
    for x0, x1 in stripes:
        bgr[_MARGIN : _H - _MARGIN, x0:x1] = (30, 170, 30)
        crop_mask[_MARGIN : _H - _MARGIN, x0:x1] = True

    if inject_weed_patch:
        w_x0, w_x1 = stripes[1][1] + 3, stripes[2][0] - 3
        bgr[100:140, w_x0:w_x1] = (30, 170, 30)

    label = np.full((_H, _W), "CROP", dtype=object)
    label[:, _CROP_W:] = "BARE_SOIL"
    for y in range(_H):
        shade = 1.0 if (y // 3) % 2 == 0 else 0.55
        bgr[y, _CROP_W:] = tuple(int(c * shade) for c in (90, 140, 180))
    field = np.ones((_H, _W), dtype=bool)

    return bgr, field, label, crop_mask


def _active_crop_unit_and_geometry(bgr, field, label, crop_mask):
    seg = segment_management_units(bgr, field, label, rows=8, cols=8)
    row_results = analyze_management_unit_rows(crop_mask, field, seg)
    crop_unit = next(u for u in seg.units if u.unit_type == UnitType.ACTIVE_CROP)
    geometry = next(r.row_geometry for r in row_results if r.unit_id == crop_unit.unit_id)
    unit_mask = unit_footprint_mask(field, seg, crop_unit.unit_id)
    return unit_mask, geometry


def test_weed_patch_between_rows_is_detected_as_inter_row_vegetation():
    bgr_with, field, label, crop_mask = _crop_rows_next_to_residue(inject_weed_patch=True)
    unit_mask_with, geometry_with = _active_crop_unit_and_geometry(bgr_with, field, label, crop_mask)
    with_patch = analyze_inter_row_vegetation(
        bgr_with, unit_mask_with, geometry_with, row_band_half_width_px=_ROW_BAND_HALF_WIDTH
    )

    bgr_without, field, label, crop_mask = _crop_rows_next_to_residue(inject_weed_patch=False)
    unit_mask_without, geometry_without = _active_crop_unit_and_geometry(bgr_without, field, label, crop_mask)
    without_patch = analyze_inter_row_vegetation(
        bgr_without, unit_mask_without, geometry_without, row_band_half_width_px=_ROW_BAND_HALF_WIDTH
    )

    assert with_patch is not None and without_patch is not None
    assert with_patch.inter_row_vegetation_fraction > without_patch.inter_row_vegetation_fraction
    assert with_patch.inter_row_vegetation_fraction > 0.0
    assert without_patch.inter_row_vegetation_fraction == 0.0


def test_row_band_and_inter_row_fractions_sum_to_the_unit_footprint():
    bgr, field, label, crop_mask = _crop_rows_next_to_residue(inject_weed_patch=False)
    unit_mask, geometry = _active_crop_unit_and_geometry(bgr, field, label, crop_mask)

    result = analyze_inter_row_vegetation(bgr, unit_mask, geometry, row_band_half_width_px=_ROW_BAND_HALF_WIDTH)

    assert result is not None
    assert abs((result.row_band_fraction + result.inter_row_fraction) - 1.0) < 1e-6


def test_smaller_row_band_width_classifies_more_area_as_inter_row():
    bgr, field, label, crop_mask = _crop_rows_next_to_residue(inject_weed_patch=False)
    unit_mask, geometry = _active_crop_unit_and_geometry(bgr, field, label, crop_mask)

    narrow = analyze_inter_row_vegetation(bgr, unit_mask, geometry, row_band_half_width_px=_ROW_BAND_HALF_WIDTH)
    wide = analyze_inter_row_vegetation(bgr, unit_mask, geometry, row_band_half_width_px=10.0)

    assert narrow.inter_row_fraction > wide.inter_row_fraction


def test_returns_none_when_row_geometry_has_no_dominant_angle():
    unit_mask = np.ones((50, 50), dtype=bool)
    empty_geometry = RowGeometryResult(
        thinning_method="guo_hall",
        guo_hall_available=True,
        skeleton_pixel_count=0,
        hough_line_count=0,
        dominant_angle_deg=None,
    )

    assert analyze_inter_row_vegetation(np.zeros((50, 50, 3), dtype=np.uint8), unit_mask, empty_geometry) is None


def test_returns_none_when_no_row_clusters_detected():
    unit_mask = np.ones((50, 50), dtype=bool)
    geometry = RowGeometryResult(
        thinning_method="guo_hall",
        guo_hall_available=True,
        skeleton_pixel_count=10,
        hough_line_count=1,
        dominant_angle_deg=90.0,
        row_clusters=[],
    )

    assert analyze_inter_row_vegetation(np.zeros((50, 50, 3), dtype=np.uint8), unit_mask, geometry) is None


def test_returns_none_for_empty_unit_mask():
    geometry = RowGeometryResult(
        thinning_method="guo_hall",
        guo_hall_available=True,
        skeleton_pixel_count=10,
        hough_line_count=1,
        dominant_angle_deg=90.0,
        row_clusters=[RowLineCluster(angle_deg=90.0, offset_px=25.0, support_px=100.0, line_count=1)],
    )
    empty_mask = np.zeros((50, 50), dtype=bool)

    assert analyze_inter_row_vegetation(np.zeros((50, 50, 3), dtype=np.uint8), empty_mask, geometry) is None


def test_classify_row_band_empty_when_no_row_clusters():
    unit_mask = np.ones((30, 30), dtype=bool)
    geometry = RowGeometryResult(
        thinning_method="guo_hall",
        guo_hall_available=True,
        skeleton_pixel_count=0,
        hough_line_count=0,
        dominant_angle_deg=None,
    )

    band = classify_row_band(unit_mask, geometry)

    assert not np.any(band)
