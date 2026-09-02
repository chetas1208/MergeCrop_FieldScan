from cropmerge.pipeline.schemas import (
    AnalysisSummary,
    ArtifactPaths,
    FieldSummary,
    FieldTriageReport,
    FrameQuality,
    InspectionZone,
    RelativeLocation,
    ReviewPriority,
    VideoSourceMeta,
    ZoneEvidence,
)
from cropmerge.pipeline.summary import build_field_analysis_summary


def _frame_quality(index: int, usable: bool) -> FrameQuality:
    return FrameQuality(
        frame_index=index,
        timestamp_sec=float(index),
        sharpness=0.8,
        exposure_score=0.8,
        mean_luminance=120.0,
        near_black_fraction=0.0,
        saturated_fraction=0.0,
        usable=usable,
        quality_weight=1.0 if usable else 0.15,
        warnings=[] if usable else ["low_sharpness"],
    )


def _zone(zone_id: str, priority: ReviewPriority, anomaly_score: float, persistent=5, total=6) -> InspectionZone:
    return InspectionZone(
        id=zone_id,
        review_priority=priority,
        anomaly_score=anomaly_score,
        review_score=anomaly_score,
        persistence_score=0.8,
        primary_signal_label="Possible stand gap / crop discontinuity",
        persistent_observations=persistent,
        total_observations=total,
        first_seen_ms=0,
        last_seen_ms=1000,
        first_seen_sec=0.0,
        last_seen_sec=1.0,
        frames_seen=total,
        relative_location=RelativeLocation.NE,
        centroid_norm={"x": 0.5, "y": 0.5},
        bbox_norm={"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2},
        evidence=ZoneEvidence(crop_coverage_delta=0.35, soil_exposure_delta=0.2),
        reasons=["Crop-row continuity breaks in this region"],
        recommendation="Review this area",
    )


def _base_report(frame_quality, zones) -> FieldTriageReport:
    return FieldTriageReport(
        run_id="abc123456789",
        created_at="2026-01-01T00:00:00Z",
        disclaimer="test",
        source=VideoSourceMeta(filename="f.mp4", duration_sec=10.0, width=640, height=360, fps=30.0, frame_count=300),
        analysis=AnalysisSummary(
            frames_sampled=len(frame_quality),
            frames_usable=sum(1 for f in frame_quality if f.usable),
            sample_fps=1.0,
            segmentation_backend="sam2",
            dino_backend="dinov2",
            used_fallback=False,
            device="cuda",
            stage_latency_sec={},
            total_runtime_sec=1.0,
        ),
        field=FieldSummary(
            detected=True,
            mean_crop_coverage=0.78,
            mean_bare_soil=0.09,
            road_path_detected=False,
            tree_vegetation_detected=False,
            water_detected=False,
            infrastructure_detected=False,
            mean_field_fraction=0.86,
        ),
        inspection_zones=zones,
        frame_quality=frame_quality,
        class_coverage={},
        limitations=["RGB-only visual analysis"],
        artifacts=ArtifactPaths(results_json="results.json"),
    )


def test_summary_zero_zones():
    frames = [_frame_quality(i, usable=True) for i in range(10)]
    report = _base_report(frames, [])
    summary = build_field_analysis_summary(report)
    assert summary.scheduled_observations == 10
    assert summary.usable_observations == 10
    assert summary.limited_observations == 0
    assert summary.high_priority_count == 0
    assert summary.highest_priority_zone_id is None
    assert "No significant" in summary.headline
    assert any("No persistent" in f for f in summary.key_findings)


def test_summary_counts_limited_observations():
    frames = [_frame_quality(i, usable=(i % 3 != 0)) for i in range(9)]
    report = _base_report(frames, [])
    summary = build_field_analysis_summary(report)
    assert summary.scheduled_observations == 9
    assert summary.usable_observations == 6
    assert summary.limited_observations == 3


def test_summary_picks_highest_priority_zone():
    zones = [
        _zone("z-low", ReviewPriority.LOW, 0.2),
        _zone("z-high", ReviewPriority.HIGH, 0.9),
        _zone("z-med", ReviewPriority.MEDIUM, 0.5),
    ]
    frames = [_frame_quality(i, usable=True) for i in range(10)]
    report = _base_report(frames, zones)
    summary = build_field_analysis_summary(report)
    assert summary.highest_priority_zone_id == "z-high"
    assert summary.high_priority_count == 1
    assert summary.medium_priority_count == 1
    assert summary.low_priority_count == 1
    assert "strongest review target" in summary.headline.lower() or "stand gap" in summary.headline.lower()
    assert any("Crop coverage is lower" in f for f in summary.key_findings)
    assert any("soil is higher" in f for f in summary.key_findings)


def test_summary_never_invents_multispectral_or_gps_claims():
    frames = [_frame_quality(i, usable=True) for i in range(5)]
    report = _base_report(frames, [_zone("z1", ReviewPriority.HIGH, 0.9)])
    summary = build_field_analysis_summary(report)
    full_text = summary.headline + " ".join(summary.key_findings)
    assert "NDVI" not in full_text
    assert "NDRE" not in full_text
    assert "northwest corner" not in full_text.lower()
