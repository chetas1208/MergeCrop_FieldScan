"""Synthetic, deterministic tests for cropmerge.features.spacing_metrics.

All inputs are hand-constructed arrays with known Miss/Multiple/QFI answers
worked out by hand -- no plant detector, no GPU/model dependency.
"""
from __future__ import annotations

import numpy as np
import pytest

from cropmerge.features.spacing_metrics import (
    interplant_distances,
    miss_index,
    multiple_index,
    planting_uniformity,
    quality_of_feed_index,
    relative_spacing_irregularity,
)


def test_interplant_distances_basic():
    positions = np.array([0, 10, 20, 20.5, 35])
    d = interplant_distances(positions)
    assert d == pytest.approx([10, 10, 0.5, 14.5])


def test_interplant_distances_unsorted_input():
    positions = np.array([20, 0, 35, 10])
    d = interplant_distances(positions)
    assert d == pytest.approx([10, 10, 15])


def test_interplant_distances_too_few_points():
    assert interplant_distances(np.array([5.0])).size == 0
    assert interplant_distances(np.array([])).size == 0


def test_miss_multiple_qfi_hand_calculated():
    # nominal = 10. spacings = [10,10,10,25,10,3,10] -> N=7
    # miss: spacing > 15  -> only 25            -> 1/7 * 100 = 14.285714...
    # multiple: spacing <= 5 -> only 3          -> 1/7 * 100 = 14.285714...
    # qfi = 100 - miss - multiple = 71.428571...
    spacings = np.array([10, 10, 10, 25, 10, 3, 10])
    nominal = 10.0

    miss = miss_index(spacings, nominal)
    mult = multiple_index(spacings, nominal)
    qfi = quality_of_feed_index(spacings, nominal)

    assert miss == pytest.approx(100.0 / 7.0, rel=1e-6)
    assert mult == pytest.approx(100.0 / 7.0, rel=1e-6)
    assert qfi == pytest.approx(100.0 - 200.0 / 7.0, rel=1e-6)


def test_planting_uniformity_bundle_matches_individual_functions():
    spacings = np.array([10, 10, 10, 25, 10, 3, 10])
    nominal = 10.0
    result = planting_uniformity(spacings, nominal)
    assert result.miss_index == pytest.approx(miss_index(spacings, nominal))
    assert result.multiple_index == pytest.approx(multiple_index(spacings, nominal))
    assert result.quality_of_feed_index == pytest.approx(quality_of_feed_index(spacings, nominal))
    assert result.nominal_spacing == nominal
    assert result.n == 7


def test_perfect_uniform_stand_scores_100():
    spacings = np.full(20, 10.0)
    result = planting_uniformity(spacings, nominal_spacing=10.0)
    assert result.miss_index == 0.0
    assert result.multiple_index == 0.0
    assert result.quality_of_feed_index == 100.0


def test_miss_multiple_boundary_conditions_are_exclusive_at_threshold():
    # Exactly 1.5x nominal should NOT count as a miss (strict >).
    # Exactly 0.5x nominal SHOULD count as a multiple (<=).
    nominal = 10.0
    spacings = np.array([15.0, 5.0])  # exactly at both boundaries
    assert miss_index(spacings, nominal) == 0.0  # 15 is not > 15
    assert multiple_index(spacings, nominal) == pytest.approx(50.0)  # 5 <= 5


def test_empty_and_invalid_nominal_are_safe():
    assert miss_index(np.array([]), 10.0) == 0.0
    assert multiple_index(np.array([]), 10.0) == 0.0
    assert miss_index(np.array([10.0, 20.0]), 0.0) == 0.0
    assert multiple_index(np.array([10.0, 20.0]), -5.0) == 0.0


def test_relative_spacing_irregularity_uniform_stand():
    spacings = np.full(10, 12.0)
    result = relative_spacing_irregularity(spacings)
    assert result.median_spacing == pytest.approx(12.0)
    assert result.large_gap_fraction_pct == 0.0
    assert result.tight_cluster_fraction_pct == 0.0
    assert result.coefficient_of_variation == pytest.approx(0.0)
    assert result.n == 10


def test_relative_spacing_irregularity_detects_gap_and_cluster():
    # median = 10; one spacing (25) is a "large gap" (> 15), one (4) is a
    # "tight cluster" (<= 5).
    spacings = np.array([10, 10, 10, 25, 10, 4, 10])
    result = relative_spacing_irregularity(spacings)
    assert result.median_spacing == pytest.approx(10.0)
    assert result.large_gap_fraction_pct == pytest.approx(100.0 / 7.0, rel=1e-6)
    assert result.tight_cluster_fraction_pct == pytest.approx(100.0 / 7.0, rel=1e-6)
    assert result.coefficient_of_variation > 0.0


def test_relative_spacing_irregularity_empty_input():
    result = relative_spacing_irregularity(np.array([]))
    assert result.n == 0
    assert result.median_spacing == 0.0
    assert result.large_gap_fraction_pct == 0.0
    assert result.tight_cluster_fraction_pct == 0.0
