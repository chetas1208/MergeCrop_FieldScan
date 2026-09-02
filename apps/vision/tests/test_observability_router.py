"""Deterministic tests for cropmerge.features.observability.

Pure threshold logic -- no imagery, no model dependency.
"""
from __future__ import annotations

from cropmerge.features.observability import ObservabilityThresholds, classify_observability


def test_canopy_only_when_no_row_evidence():
    assert classify_observability(0.0, 0.0, 0) == "CANOPY_ONLY"


def test_canopy_only_when_only_one_criterion_weak():
    # Strong coherence + support, but only a single row line -> not
    # resolvable as "rows" (could be a road/fence edge).
    assert classify_observability(0.9, 500.0, 1) == "CANOPY_ONLY"
    # Strong coherence + enough rows, but weak pixel support.
    assert classify_observability(0.9, 5.0, 3) == "CANOPY_ONLY"
    # Enough rows + support, but weak coherence.
    assert classify_observability(0.1, 500.0, 3) == "CANOPY_ONLY"


def test_row_resolvable_when_all_row_criteria_met_without_plant_evidence():
    tier = classify_observability(0.5, 100.0, 3)
    assert tier == "ROW_RESOLVABLE"


def test_row_resolvable_default_when_plant_evidence_absent():
    # Defaults for plant_detection_count/confidence are 0 -- by design this
    # can never reach PLANT_RESOLVABLE without the caller explicitly
    # supplying qualifying plant evidence (no plant detector exists yet).
    tier = classify_observability(0.9, 1000.0, 10)
    assert tier == "ROW_RESOLVABLE"


def test_plant_resolvable_requires_row_resolvable_plus_plant_evidence():
    tier = classify_observability(
        0.5,
        100.0,
        3,
        plant_detection_count=5,
        plant_detection_confidence=0.8,
    )
    assert tier == "PLANT_RESOLVABLE"


def test_plant_resolvable_downgrades_to_row_resolvable_if_rows_not_met():
    # Strong plant evidence cannot compensate for failing row-resolvability
    # criteria -- PLANT_RESOLVABLE is a strict superset of ROW_RESOLVABLE.
    tier = classify_observability(
        0.1,  # weak coherence -> rows not resolvable
        100.0,
        3,
        plant_detection_count=10,
        plant_detection_confidence=0.99,
    )
    assert tier == "CANOPY_ONLY"


def test_plant_resolvable_requires_confidence_not_just_count():
    tier = classify_observability(
        0.5,
        100.0,
        3,
        plant_detection_count=10,
        plant_detection_confidence=0.1,  # below default 0.6 threshold
    )
    assert tier == "ROW_RESOLVABLE"


def test_boundary_values_are_inclusive():
    t = ObservabilityThresholds()
    tier = classify_observability(
        t.min_row_coherence,
        t.min_row_line_support_px,
        t.min_consistent_rows,
    )
    assert tier == "ROW_RESOLVABLE"

    tier_plant = classify_observability(
        t.min_row_coherence,
        t.min_row_line_support_px,
        t.min_consistent_rows,
        plant_detection_count=t.min_plant_detections_per_row,
        plant_detection_confidence=t.min_plant_detection_confidence,
    )
    assert tier_plant == "PLANT_RESOLVABLE"


def test_custom_thresholds_override_defaults():
    strict = ObservabilityThresholds(
        min_row_coherence=0.9,
        min_row_line_support_px=1000.0,
        min_consistent_rows=10,
    )
    # Would be ROW_RESOLVABLE under defaults, but not under a stricter
    # custom threshold set.
    assert classify_observability(0.5, 100.0, 3, thresholds=strict) == "CANOPY_ONLY"

    lenient = ObservabilityThresholds(
        min_row_coherence=0.01,
        min_row_line_support_px=1.0,
        min_consistent_rows=1,
    )
    assert classify_observability(0.02, 2.0, 1, thresholds=lenient) == "ROW_RESOLVABLE"
