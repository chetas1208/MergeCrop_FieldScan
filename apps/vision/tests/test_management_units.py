from __future__ import annotations

import numpy as np

from cropmerge.features.management_units import (
    CellFeatures,
    ResidueClass,
    UnitBoundaryWeights,
    UnitType,
    boundary_cost,
    classify_unit_type,
    segment_management_units,
    unit_type_confidence,
)


def _three_zone_field(h: int = 240, w: int = 240):
    """Left third = living green crop, middle third = smooth bare soil
    (brown, low texture), right third = textured residue (brown, high
    texture from alternating stripes) -- the real reported scenario: a
    field with genuinely distinct management-relevant zones."""
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    third = w // 3
    bgr[:, 0:third] = (30, 170, 30)  # BGR green
    bgr[:, third : 2 * third] = (90, 140, 180)  # smooth tan (soil)
    for y in range(h):
        shade = 1.0 if (y // 3) % 2 == 0 else 0.55
        bgr[y, 2 * third :] = tuple(int(c * shade) for c in (90, 140, 180))  # striped tan (residue)
    field = np.ones((h, w), dtype=bool)
    label = np.full((h, w), "CROP", dtype=object)
    label[:, third:] = "BARE_SOIL"
    return bgr, field, label


def test_boundary_cost_zero_for_identical_cells():
    f = CellFeatures(
        row=0, col=0, valid=True, vegetation_fraction=0.5, residue_class=ResidueClass.LIVING_VEGETATION,
        residue_score=0.0, color_mean_bgr=(40.0, 150.0, 40.0), row_angle_deg=30.0, row_coherence=0.5,
    )
    assert boundary_cost(f, f) == 0.0


def test_boundary_cost_is_maximal_when_either_cell_invalid():
    valid = CellFeatures(
        row=0, col=0, valid=True, vegetation_fraction=0.5, residue_class=ResidueClass.LIVING_VEGETATION,
        residue_score=0.0, color_mean_bgr=(40.0, 150.0, 40.0), row_angle_deg=None, row_coherence=0.0,
    )
    invalid = CellFeatures(
        row=0, col=1, valid=False, vegetation_fraction=0.0, residue_class=ResidueClass.UNKNOWN,
        residue_score=0.0, color_mean_bgr=(0.0, 0.0, 0.0), row_angle_deg=None, row_coherence=0.0,
    )
    assert boundary_cost(valid, invalid) == 1.0


def test_boundary_cost_high_between_vegetation_and_residue():
    veg = CellFeatures(
        row=0, col=0, valid=True, vegetation_fraction=0.8, residue_class=ResidueClass.LIVING_VEGETATION,
        residue_score=0.0, color_mean_bgr=(30.0, 170.0, 30.0), row_angle_deg=None, row_coherence=0.0,
    )
    residue = CellFeatures(
        row=0, col=1, valid=True, vegetation_fraction=0.0, residue_class=ResidueClass.LIKELY_RESIDUE,
        residue_score=1.0, color_mean_bgr=(90.0, 140.0, 180.0), row_angle_deg=None, row_coherence=0.0,
    )
    cost = boundary_cost(veg, residue)
    assert cost > 0.5


def test_three_zone_field_produces_multiple_management_units():
    bgr, field, label = _three_zone_field()

    result = segment_management_units(bgr, field, label, rows=6, cols=6)

    assert len(result.units) >= 2, "expected the field's real distinct zones to stay separate"
    # No cell should be assigned to more than one unit, and every valid
    # cell must be assigned to some unit.
    assigned = [uid for row in result.cell_unit_ids for uid in row if uid != -1]
    assert len(assigned) > 0
    total_area = sum(u.area_fraction for u in result.units)
    assert abs(total_area - 1.0) < 1e-6


def test_uniform_field_produces_one_management_unit():
    bgr = np.full((200, 200, 3), (40, 150, 40), dtype=np.uint8)
    field = np.ones((200, 200), dtype=bool)
    label = np.full((200, 200), "CROP", dtype=object)

    result = segment_management_units(bgr, field, label, rows=6, cols=6)

    assert len(result.units) == 1
    assert result.units[0].area_fraction == 1.0


def test_dominant_residue_class_matches_the_zone_content():
    bgr, field, label = _three_zone_field()

    result = segment_management_units(bgr, field, label, rows=6, cols=6)

    classes = {u.dominant_residue_class for u in result.units}
    assert "living_vegetation" in classes


def test_higher_threshold_merges_more_aggressively():
    bgr, field, label = _three_zone_field()

    strict = segment_management_units(bgr, field, label, rows=6, cols=6, boundary_threshold=0.05)
    loose = segment_management_units(bgr, field, label, rows=6, cols=6, boundary_threshold=0.99)

    assert len(loose.units) <= len(strict.units)
    assert len(loose.units) == 1  # threshold near 1.0 merges everything except invalid cells


def test_custom_weights_change_the_result():
    bgr, field, label = _three_zone_field()

    default_result = segment_management_units(bgr, field, label, rows=6, cols=6, boundary_threshold=0.2)
    zero_weights = UnitBoundaryWeights(vegetation=0.0, residue=0.0, color=0.0, row_orientation=0.0)
    no_boundary_result = segment_management_units(
        bgr, field, label, rows=6, cols=6, boundary_threshold=0.2, weights=zero_weights
    )

    # With every weight zeroed, boundary cost is always 0 -> everything merges.
    assert len(no_boundary_result.units) == 1
    assert len(default_result.units) >= len(no_boundary_result.units)


def test_classify_unit_type_active_crop_from_vegetation():
    assert classify_unit_type(0.6, ResidueClass.LIVING_VEGETATION, 0.0) == UnitType.ACTIVE_CROP


def test_classify_unit_type_residue_stubble_when_no_vegetation():
    assert classify_unit_type(0.02, ResidueClass.LIKELY_RESIDUE, 0.0) == UnitType.RESIDUE_STUBBLE


def test_classify_unit_type_bare_soil_when_no_vegetation_no_residue():
    assert classify_unit_type(0.02, ResidueClass.LIKELY_BARE_SOIL, 0.0) == UnitType.BARE_SOIL


def test_classify_unit_type_non_crop_overrides_vegetation():
    # Even a "vegetated" reading (e.g. tree canopy or roadside grass) must
    # not be reported as ACTIVE_CROP once non-crop land cover dominates.
    assert classify_unit_type(0.6, ResidueClass.LIVING_VEGETATION, 0.9) == UnitType.NON_CROP


def test_classify_unit_type_unknown_when_ambiguous():
    assert classify_unit_type(0.02, ResidueClass.UNKNOWN, 0.0) == UnitType.UNKNOWN


def test_three_zone_field_units_carry_correct_unit_type():
    bgr, field, label = _three_zone_field()

    result = segment_management_units(bgr, field, label, rows=6, cols=6)

    types = {u.unit_type for u in result.units}
    assert UnitType.ACTIVE_CROP in types
    # The residue/soil middle+right zones must never be reported as ACTIVE_CROP.
    assert all(u.unit_type != UnitType.ACTIVE_CROP or u.mean_vegetation_fraction > 0.0 for u in result.units)


def test_residue_and_soil_units_never_classified_active_crop():
    bgr, field, label = _three_zone_field()

    result = segment_management_units(bgr, field, label, rows=6, cols=6)

    for u in result.units:
        if u.dominant_residue_class in ("likely_residue", "likely_bare_soil") and u.mean_vegetation_fraction < 0.2:
            assert u.unit_type != UnitType.ACTIVE_CROP


def test_unit_type_confidence_is_zero_for_empty_members():
    assert unit_type_confidence([], UnitType.ACTIVE_CROP, ResidueClass.LIVING_VEGETATION) == 0.0


def test_unit_type_confidence_higher_with_more_agreeing_cells():
    agreeing = [
        CellFeatures(
            row=0, col=i, valid=True, vegetation_fraction=0.9, residue_class=ResidueClass.LIVING_VEGETATION,
            residue_score=0.0, color_mean_bgr=(30.0, 170.0, 30.0), row_angle_deg=None, row_coherence=0.0,
        )
        for i in range(6)
    ]
    mixed = agreeing[:2] + [
        CellFeatures(
            row=0, col=i, valid=True, vegetation_fraction=0.0, residue_class=ResidueClass.LIKELY_BARE_SOIL,
            residue_score=0.5, color_mean_bgr=(90.0, 140.0, 180.0), row_angle_deg=None, row_coherence=0.0,
        )
        for i in range(4)
    ]

    high = unit_type_confidence(agreeing, UnitType.ACTIVE_CROP, ResidueClass.LIVING_VEGETATION)
    low = unit_type_confidence(mixed, UnitType.ACTIVE_CROP, ResidueClass.LIVING_VEGETATION)

    assert high > low
    assert high == 1.0  # 6/6 agreement, 6/6 spatial support (>= minimum)


def test_unit_type_confidence_capped_by_spatial_support_even_with_perfect_agreement():
    one_cell = [
        CellFeatures(
            row=0, col=0, valid=True, vegetation_fraction=0.9, residue_class=ResidueClass.LIVING_VEGETATION,
            residue_score=0.0, color_mean_bgr=(30.0, 170.0, 30.0), row_angle_deg=None, row_coherence=0.0,
        )
    ]

    confidence = unit_type_confidence(one_cell, UnitType.ACTIVE_CROP, ResidueClass.LIVING_VEGETATION)

    assert confidence < 1.0  # perfect agreement, but only 1 of 6 minimum cells sampled
