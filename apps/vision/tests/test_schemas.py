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


def test_report_roundtrip_camel():
    report = FieldTriageReport(
        run_id="abc",
        created_at="2026-01-01T00:00:00Z",
        disclaimer="test",
        source=VideoSourceMeta(
            filename="f.mp4",
            duration_sec=10.0,
            width=640,
            height=360,
            fps=30.0,
            frame_count=300,
        ),
        analysis=AnalysisSummary(
            frames_sampled=10,
            frames_usable=9,
            sample_fps=2.0,
            segmentation_backend="heuristic",
            dino_backend="heuristic",
            used_fallback=True,
            device="cpu",
            stage_latency_sec={"total": 1.0},
            total_runtime_sec=1.0,
        ),
        field=FieldSummary(
            detected=True,
            mean_crop_coverage=0.8,
            mean_bare_soil=0.05,
            road_path_detected=True,
            tree_vegetation_detected=False,
            water_detected=False,
            infrastructure_detected=False,
            mean_field_fraction=0.85,
        ),
        inspection_zones=[
            InspectionZone(
                id="ZONE_001",
                review_priority=ReviewPriority.HIGH,
                anomaly_score=0.84,
                persistence_score=0.86,
                first_seen_ms=14200,
                last_seen_ms=18800,
                first_seen_sec=14.2,
                last_seen_sec=18.8,
                frames_seen=10,
                relative_location=RelativeLocation.NE,
                centroid_norm={"x": 0.7, "y": 0.2},
                bbox_norm={"x": 0.6, "y": 0.1, "w": 0.2, "h": 0.2},
                evidence=ZoneEvidence(crop_coverage_delta=-0.19, persistence=0.86),
                reasons=["Lower visible crop coverage than the field baseline"],
                recommendation="Inspect this area at closer range.",
            )
        ],
        frame_quality=[
            FrameQuality(
                frame_index=0,
                timestamp_sec=0.0,
                sharpness=0.8,
                exposure_score=0.9,
                mean_luminance=120,
                near_black_fraction=0.0,
                saturated_fraction=0.0,
                usable=True,
                quality_weight=1.0,
            )
        ],
        class_coverage={"CROP": 0.8},
        limitations=["RGB-only visual analysis"],
        artifacts=ArtifactPaths(results_json="/tmp/r.json", annotated_video=None),
    )
    d = report.to_camel_dict()

    assert d["schemaVersion"] == "1.0"
    assert d["inspectionZones"][0]["reviewPriority"] == "high"
    assert d["georeferenced"] is False
    # validate pydantic
    FieldTriageReport.model_validate(report.model_dump())
