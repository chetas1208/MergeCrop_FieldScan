from __future__ import annotations

import numpy as np

from cropmerge.features.weed_pressure import understory_vegetation_fraction


def test_no_field_returns_zeros():
    bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    field = np.zeros((20, 20), dtype=bool)
    label = np.full((20, 20), "CROP", dtype=object)
    result = understory_vegetation_fraction(bgr, field, label)
    assert result == {"soilFraction": 0.0, "understoryVegetationFraction": 0.0, "vegetatedSoilFractionOfField": 0.0}


def test_no_soil_in_field_gives_zero_understory():
    bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    field = np.ones((20, 20), dtype=bool)
    label = np.full((20, 20), "CROP", dtype=object)
    result = understory_vegetation_fraction(bgr, field, label)
    assert result["soilFraction"] == 0.0
    assert result["understoryVegetationFraction"] == 0.0


def test_bare_soil_with_no_vegetation_signal():
    bgr = np.full((20, 20, 3), 120, dtype=np.uint8)  # flat gray, ExG ~ 0
    field = np.ones((20, 20), dtype=bool)
    label = np.full((20, 20), "BARE_SOIL", dtype=object)
    result = understory_vegetation_fraction(bgr, field, label)
    assert result["soilFraction"] == 1.0
    assert result["understoryVegetationFraction"] == 0.0
    assert result["vegetatedSoilFractionOfField"] == 0.0


def test_bare_soil_with_strong_green_signal_detected_as_understory():
    bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    bgr[:] = (20, 200, 20)  # strong green (BGR) -> high ExG
    field = np.ones((20, 20), dtype=bool)
    label = np.full((20, 20), "BARE_SOIL", dtype=object)
    result = understory_vegetation_fraction(bgr, field, label)
    assert result["soilFraction"] == 1.0
    assert result["understoryVegetationFraction"] == 1.0
    assert result["vegetatedSoilFractionOfField"] == 1.0


def test_partial_vegetated_soil_within_larger_field():
    h, w = 20, 40
    bgr = np.full((h, w, 3), 120, dtype=np.uint8)  # gray crop half
    bgr[:, w // 2 :] = (20, 200, 20)  # green weedy soil half
    field = np.ones((h, w), dtype=bool)
    label = np.full((h, w), "CROP", dtype=object)
    label[:, w // 2 :] = "BARE_SOIL"  # right half classified as soil, but green

    result = understory_vegetation_fraction(bgr, field, label)

    assert result["soilFraction"] == 0.5
    assert result["understoryVegetationFraction"] == 1.0  # all soil pixels are vegetated
    assert result["vegetatedSoilFractionOfField"] == 0.5  # but only half the FIELD is
