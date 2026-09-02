from __future__ import annotations

import pytest

from cropmerge.pipeline.schemas import (
    AnalysisSummary,
    ArtifactPaths,
    FieldSummary,
    FieldTriageReport,
    InspectionZone,
    InspectionZoneType,
    RelativeLocation,
    ReviewPriority,
    VideoSourceMeta,
    ZoneEvidence,
)
from cropmerge.storage.drift_metrics import (
    DriftGate,
    _bbox_iou,
    compare_reports,
    match_zones,
    passes_gate,
)


def _zone(
    zone_id: str,
    *,
    x=0.4, y=0.4, w=0.2, h=0.2,
    priority=ReviewPriority.HIGH,
    zone_type=InspectionZoneType.STAND_GAP,
) -> InspectionZone:
    return InspectionZone(
        id=zone_id,
        review_priority=priority,
        anomaly_score=0.8,
        review_score=0.8,
        persistence_score=0.8,
        primary_type=zone_type,
        persistent_observations=5,
        total_observations=6,
        first_seen_ms=0, last_seen_ms=1000, first_seen_sec=0.0, last_seen_sec=1.0,
        frames_seen=6,
        relative_location=RelativeLocation.C,
        centroid_norm={"x": x + w / 2, "y": y + h / 2},
        bbox_norm={"x": x, "y": y, "w": w, "h": h},
        evidence=ZoneEvidence(),
        reasons=["test"],
        recommendation="Review",
    )


def _report(zones: list[InspectionZone], crop_cov=0.5, bare_soil=0.1, field_frac=0.7) -> FieldTriageReport:
    return FieldTriageReport(
        run_id="abc123456789", created_at="t", disclaimer="d",
        source=VideoSourceMeta(filename="f.mp4", duration_sec=1.0, width=10, height=10, fps=1.0, frame_count=1),
        analysis=AnalysisSummary(
            frames_sampled=1, frames_usable=1, sample_fps=1.0, segmentation_backend="heuristic",
            dino_backend="heuristic", used_fallback=True, device="cpu", stage_latency_sec={}, total_runtime_sec=0.1,
        ),
        field=FieldSummary(
            detected=True, mean_crop_coverage=crop_cov, mean_bare_soil=bare_soil, road_path_detected=False,
            tree_vegetation_detected=False, water_detected=False, infrastructure_detected=False,
            mean_field_fraction=field_frac,
        ),
        inspection_zones=zones, frame_quality=[], class_coverage={}, limitations=[],
        artifacts=ArtifactPaths(results_json="r.json"),
    )


def test_bbox_iou_identical_rects_is_one():
    r = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    assert _bbox_iou(r, r) == pytest.approx(1.0)


def test_bbox_iou_disjoint_rects_is_zero():
    a = {"x": 0.0, "y": 0.0, "w": 0.1, "h": 0.1}
    b = {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}
    assert _bbox_iou(a, b) == 0.0


def test_match_zones_finds_overlapping_candidate():
    orig = [_zone("z1", x=0.4, y=0.4, w=0.2, h=0.2)]
    cand = [_zone("z1c", x=0.42, y=0.42, w=0.2, h=0.2)]  # slightly shifted, still overlapping

    matches = match_zones(orig, cand, min_iou=0.3)

    assert len(matches) == 1
    assert matches[0].candidate is not None
    assert matches[0].iou > 0.3
    assert matches[0].type_agrees is True
    assert matches[0].priority_agrees is True


def test_match_zones_reports_unmatched_when_candidate_missing():
    orig = [_zone("z1", x=0.4, y=0.4, w=0.2, h=0.2)]
    cand: list[InspectionZone] = []  # compression dropped the finding entirely

    matches = match_zones(orig, cand)

    assert len(matches) == 1
    assert matches[0].candidate is None


def test_compare_reports_zero_drift_for_identical_reports():
    zones = [_zone("z1")]
    report = _report(zones)

    drift = compare_reports(report, report)

    assert drift.crop_coverage_mae == 0.0
    assert drift.soil_fraction_mae == 0.0
    assert drift.type_agreement == 1.0
    assert drift.priority_agreement == 1.0
    assert drift.high_priority_recall == 1.0
    assert drift.mean_matched_iou == pytest.approx(1.0)


def test_compare_reports_detects_dropped_high_priority_finding():
    original = _report([_zone("z1", priority=ReviewPriority.HIGH)])
    candidate = _report([])  # candidate lost the high-priority finding entirely

    drift = compare_reports(original, candidate)

    assert drift.high_priority_recall == 0.0
    ok, reasons = passes_gate(drift)
    assert ok is False
    assert any("high-priority recall" in r for r in reasons)


def test_compare_reports_detects_coverage_drift():
    original = _report([], crop_cov=0.60, bare_soil=0.10)
    candidate = _report([], crop_cov=0.50, bare_soil=0.10)  # 10pt crop coverage shift

    drift = compare_reports(original, candidate)

    assert drift.crop_coverage_mae == pytest.approx(0.10)
    ok, reasons = passes_gate(drift, DriftGate(max_crop_coverage_mae=0.01))
    assert ok is False
    assert any("crop coverage MAE" in r for r in reasons)


def test_passes_gate_true_for_clean_match():
    original = _report([_zone("z1", priority=ReviewPriority.HIGH)])
    candidate = _report([_zone("z1c", priority=ReviewPriority.HIGH)])

    drift = compare_reports(original, candidate)
    ok, reasons = passes_gate(drift)

    assert ok is True
    assert reasons == []
