from __future__ import annotations

from cropmerge.pipeline.schemas import (
    ANALYSIS_VERSION,
    FARMTECH_VERSION,
    STORAGE_POLICY_VERSION,
    AnalysisSummary,
    ArtifactPaths,
    FieldSummary,
    FieldTriageReport,
    VideoSourceMeta,
)


def _minimal_report(**overrides) -> FieldTriageReport:
    defaults = dict(
        run_id="abc123456789",
        created_at="2026-01-01T00:00:00Z",
        disclaimer="test",
        source=VideoSourceMeta(filename="f.mp4", duration_sec=1.0, width=10, height=10, fps=1.0, frame_count=1),
        analysis=AnalysisSummary(
            frames_sampled=1, frames_usable=1, sample_fps=1.0, segmentation_backend="heuristic",
            dino_backend="heuristic", used_fallback=True, device="cpu", stage_latency_sec={}, total_runtime_sec=0.1,
        ),
        field=FieldSummary(
            detected=False, mean_crop_coverage=0.0, mean_bare_soil=0.0, road_path_detected=False,
            tree_vegetation_detected=False, water_detected=False, infrastructure_detected=False,
            mean_field_fraction=0.0,
        ),
        inspection_zones=[],
        frame_quality=[],
        class_coverage={},
        limitations=[],
        artifacts=ArtifactPaths(results_json="results.json"),
    )
    defaults.update(overrides)
    return FieldTriageReport(**defaults)


def test_report_defaults_to_current_policy_versions():
    report = _minimal_report()
    assert report.storage_policy_version == STORAGE_POLICY_VERSION == "fieldscan-storage-v1"
    assert report.farm_tech_version == FARMTECH_VERSION == "farmtech-v1"
    assert report.source_sha256 is None

    camel = report.to_camel_dict()
    assert camel["storagePolicyVersion"] == "fieldscan-storage-v1"
    assert camel["farmTechVersion"] == "farmtech-v1"
    assert camel["sourceSha256"] is None


def test_report_source_sha256_round_trips_through_camel_dict():
    report = _minimal_report()
    report.source_sha256 = "a" * 64  # mutated post-construction, mirroring api/main.py's JobManager
    assert report.to_camel_dict()["sourceSha256"] == "a" * 64


def test_report_analysis_version_and_config_version():
    report = _minimal_report()
    assert report.analysis_version == ANALYSIS_VERSION == "analysis-v1"
    assert report.config_version is None

    report.config_version = "deadbeef12345678"
    camel = report.to_camel_dict()
    assert camel["analysisVersion"] == "analysis-v1"
    assert camel["configVersion"] == "deadbeef12345678"
