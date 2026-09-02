from __future__ import annotations

from cropmerge.anomaly.consensus import ConsensusThresholds, evaluate_consensus
from cropmerge.anomaly.grid import build_grid
from cropmerge.anomaly.temporal import aggregate_zones
from cropmerge.config import load_config
from cropmerge.pipeline.schemas import ZoneEvidence


def _evidence(**overrides) -> ZoneEvidence:
    defaults = dict(
        structural_anomaly_score=0.0,
        appearance_anomaly_score=0.0,
        persistence=0.0,
    )
    defaults.update(overrides)
    return ZoneEvidence(**defaults)


def test_appearance_only_evidence_does_not_reach_high_confidence():
    """The exact real-world complaint: color/texture/DINO differs, but no
    structural signal at all -- must not be treated as high-confidence."""
    evidence = _evidence(appearance_anomaly_score=0.45, structural_anomaly_score=0.0, persistence=1.0)

    result = evaluate_consensus(evidence)

    assert result.appearance_has_signal is True
    assert result.structure_has_signal is False
    assert result.families_agreeing == 1
    assert result.strong_structure_alone is False
    assert result.high_confidence_supported is False


def test_both_families_agreeing_reaches_high_confidence():
    evidence = _evidence(appearance_anomaly_score=0.30, structural_anomaly_score=0.25, persistence=0.5)

    result = evaluate_consensus(evidence)

    assert result.families_agreeing == 2
    assert result.high_confidence_supported is True


def test_strong_structural_signal_alone_with_high_persistence_is_sufficient():
    evidence = _evidence(appearance_anomaly_score=0.0, structural_anomaly_score=0.40, persistence=0.8)

    result = evaluate_consensus(evidence)

    assert result.families_agreeing == 1  # appearance alone doesn't count it
    assert result.strong_structure_alone is True
    assert result.high_confidence_supported is True


def test_moderate_structural_signal_alone_with_low_persistence_is_insufficient():
    evidence = _evidence(appearance_anomaly_score=0.0, structural_anomaly_score=0.20, persistence=0.3)

    result = evaluate_consensus(evidence)

    assert result.families_agreeing == 1
    assert result.strong_structure_alone is False
    assert result.high_confidence_supported is False


def test_no_evidence_is_never_high_confidence():
    result = evaluate_consensus(_evidence())
    assert result.high_confidence_supported is False


def test_none_fields_treated_as_zero_not_an_error():
    evidence = ZoneEvidence()  # all fields default to None
    result = evaluate_consensus(evidence)
    assert result.high_confidence_supported is False


def test_custom_thresholds_are_respected():
    evidence = _evidence(appearance_anomaly_score=0.10, structural_anomaly_score=0.10, persistence=0.0)
    strict = ConsensusThresholds(structure_min=0.5, appearance_min=0.5)

    result = evaluate_consensus(evidence, strict)

    assert result.families_agreeing == 0
    assert result.high_confidence_supported is False


def _cells_with_scores(rows: int, cols: int, appearance: float, structural: float):
    """Real GridCells (real geometry via build_grid), all set to the same
    controllable appearance/structural score at one fixed cell -- everything
    else invalid so aggregate_zones only ever tracks that one cell."""
    cells = build_grid(160, 160, rows, cols)
    target = cells[0]
    target.valid = True
    target.field_fraction = 1.0
    target.appearance_anomaly_score = appearance
    target.structural_anomaly_score = structural
    target.anomaly_score = max(appearance, structural)
    target.row_visibility = "LOW"
    target.features = {}
    return cells


def test_aggregate_zones_caps_appearance_only_zone_at_medium():
    """Integration proof, not just the pure evaluate_consensus() unit test:
    a real track through aggregate_zones() with strong appearance evidence,
    zero structural evidence, and full persistence must land at 'medium',
    not 'high', with a reason explaining why."""
    cfg = load_config()
    cfg["anomaly"]["medium_threshold"] = 0.30
    cfg["anomaly"]["high_threshold"] = 0.55
    cfg["temporal"]["min_persistent_frames"] = 2
    cfg["temporal"]["centroid_link_distance"] = 0.5

    frame_cells = [_cells_with_scores(8, 8, appearance=0.70, structural=0.0) for _ in range(3)]

    zones = aggregate_zones(frame_cells, [0.0, 1.0, 2.0], [1.0, 1.0, 1.0], (160, 160), cfg)

    assert len(zones) >= 1
    zone = zones[0]
    assert zone.review_priority.value == "medium"
    assert any("Capped at Medium" in r for r in zone.reasons)


def test_aggregate_zones_allows_high_when_structural_corroborates():
    cfg = load_config()
    cfg["anomaly"]["medium_threshold"] = 0.30
    cfg["anomaly"]["high_threshold"] = 0.55
    cfg["temporal"]["min_persistent_frames"] = 2
    cfg["temporal"]["centroid_link_distance"] = 0.5

    frame_cells = [_cells_with_scores(8, 8, appearance=0.70, structural=0.40) for _ in range(3)]

    zones = aggregate_zones(frame_cells, [0.0, 1.0, 2.0], [1.0, 1.0, 1.0], (160, 160), cfg)

    assert len(zones) >= 1
    zone = zones[0]
    assert zone.review_priority.value == "high"
    assert not any("Capped at Medium" in r for r in zone.reasons)
