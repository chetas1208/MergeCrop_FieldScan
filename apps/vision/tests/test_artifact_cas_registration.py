from __future__ import annotations

from pathlib import Path

from api.main import _register_run_artifacts_in_cas, _safe_artifact_path, _storage
from cropmerge.pipeline.schemas import (
    AnalysisSummary,
    ArtifactPaths,
    FieldSummary,
    FieldTriageReport,
    VideoSourceMeta,
)

JOB_ID = "abc123456789"  # matches JOB_ID_RE ([a-f0-9]{12})


def _configure_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_SHARED_SECRET", "0123456789abcdef0123456789abcdef")
    monkeypatch.setenv("VISION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VISION_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VISION_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VISION_JOB_DATABASE_PATH", str(tmp_path / "data" / "jobs.sqlite"))


def _report_with_artifacts(run_dir: Path) -> FieldTriageReport:
    (run_dir / "results.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "annotated.mp4").write_bytes(b"fake-mp4-bytes")
    (run_dir / "heatmap.png").write_bytes(b"fake-png-bytes")

    return FieldTriageReport(
        run_id=JOB_ID,
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
        artifacts=ArtifactPaths(
            results_json=str(run_dir / "results.json"),
            metrics_json=str(run_dir / "metrics.json"),
            annotated_video=str(run_dir / "annotated.mp4"),
            heatmap_png=str(run_dir / "heatmap.png"),
        ),
    )


def test_stable_artifacts_are_registered_in_cas(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    run_dir = tmp_path / "outputs" / JOB_ID
    run_dir.mkdir(parents=True)
    # results.json must exist at the exact path _safe_artifact_path resolves to.
    _safe_artifact_path(JOB_ID, "results.json").parent.mkdir(parents=True, exist_ok=True)
    report = _report_with_artifacts(run_dir)
    _safe_artifact_path(JOB_ID, "results.json").write_text("{\"ok\": true}", encoding="utf-8")

    _register_run_artifacts_in_cas(JOB_ID, report)

    for name in ("results.json", "metrics.json", "annotated_video", "heatmap.png"):
        entry = _storage().get(f"run:{JOB_ID}:{name}")
        assert entry is not None, f"expected {name} to be registered"

    results_entry = _storage().get(f"run:{JOB_ID}:results.json")
    assert results_entry.storage_class == "derived_analysis"
    video_entry = _storage().get(f"run:{JOB_ID}:annotated_video")
    assert video_entry.storage_class == "derived_preview"


def test_missing_optional_artifacts_are_skipped_not_errored(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    run_dir = tmp_path / "outputs" / JOB_ID
    run_dir.mkdir(parents=True)
    _safe_artifact_path(JOB_ID, "results.json").write_text("{}", encoding="utf-8")

    report = FieldTriageReport(
        run_id=JOB_ID, created_at="x", disclaimer="x",
        source=VideoSourceMeta(filename="f.mp4", duration_sec=1.0, width=1, height=1, fps=1.0, frame_count=1),
        analysis=AnalysisSummary(
            frames_sampled=1, frames_usable=1, sample_fps=1.0, segmentation_backend="heuristic",
            dino_backend="heuristic", used_fallback=True, device="cpu", stage_latency_sec={}, total_runtime_sec=0.1,
        ),
        field=FieldSummary(
            detected=False, mean_crop_coverage=0.0, mean_bare_soil=0.0, road_path_detected=False,
            tree_vegetation_detected=False, water_detected=False, infrastructure_detected=False,
            mean_field_fraction=0.0,
        ),
        inspection_zones=[], frame_quality=[], class_coverage={}, limitations=[],
        artifacts=ArtifactPaths(results_json=str(run_dir / "results.json")),  # no video/heatmap/metrics
    )

    _register_run_artifacts_in_cas(JOB_ID, report)  # must not raise

    assert _storage().get(f"run:{JOB_ID}:results.json") is not None
    assert _storage().get(f"run:{JOB_ID}:metrics.json") is None
    assert _storage().get(f"run:{JOB_ID}:annotated_video") is None


def test_artifact_registration_failure_is_non_fatal(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    run_dir = tmp_path / "outputs" / JOB_ID
    run_dir.mkdir(parents=True)
    _safe_artifact_path(JOB_ID, "results.json").write_text("{}", encoding="utf-8")
    report = _report_with_artifacts(run_dir)

    monkeypatch.setattr("api.main._storage", lambda: (_ for _ in ()).throw(RuntimeError("disk full")))

    _register_run_artifacts_in_cas(JOB_ID, report)  # must not raise
