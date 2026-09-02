"""Phase 16 (storage integration campaign) audit finding: 'continuity'
overlay images were computed and written to disk for every observation but
never consumed anywhere in the product (grepped apps/web -- only
overlay_NNNN.jpg and frame_NNNN.jpg are ever linked/served). Default flipped
to off in configs/default.yaml; these tests prove the gate actually works
and that toggling it has zero effect on the analysis result itself (pure
rendering side-effect, not fed back into scoring)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cropmerge.config import load_config
from cropmerge.pipeline.processor import FieldTriageProcessor


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0):
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert wr.isOpened()
    for f in frames:
        wr.write(f)
    wr.release()


def _run(tmp_path: Path, out_name: str, write_continuity: bool):
    from scripts.generate_demo_video import make_frame

    path = tmp_path / "field.mp4"
    if not path.exists():
        frames = [make_frame(i / 10.0) for i in range(20)]
        _write_video(path, frames, fps=10)

    cfg = load_config()
    cfg["temporal"]["min_persistent_frames"] = 2
    cfg["anomaly"]["medium_threshold"] = 0.15
    cfg["output"]["write_continuity_overlays"] = write_continuity
    out = tmp_path / out_name
    return FieldTriageProcessor(cfg).process(
        path, out, sample_fps=5, max_frames=10, segmentation_backend="heuristic", skip_dino=True,
    )


def test_continuity_overlays_default_off_in_shipped_config():
    cfg = load_config()
    assert cfg["output"]["write_continuity_overlays"] is False


def test_continuity_overlay_files_absent_when_disabled(tmp_path: Path):
    report = _run(tmp_path, "out-off", write_continuity=False)
    overlays_dir = Path(report.artifacts.overlays_dir)
    continuity_files = list(overlays_dir.glob("continuity_*.jpg"))
    overlay_files = list(overlays_dir.glob("overlay_*.jpg"))

    assert continuity_files == []
    assert len(overlay_files) > 0  # the actually-consumed overlay still writes


def test_continuity_overlay_files_present_when_enabled(tmp_path: Path):
    report = _run(tmp_path, "out-on", write_continuity=True)
    overlays_dir = Path(report.artifacts.overlays_dir)
    continuity_files = list(overlays_dir.glob("continuity_*.jpg"))

    assert len(continuity_files) > 0


def test_toggling_continuity_overlays_does_not_change_analysis_result(tmp_path: Path):
    off_report = _run(tmp_path, "out-off2", write_continuity=False)
    on_report = _run(tmp_path, "out-on2", write_continuity=True)

    assert len(off_report.inspection_zones) == len(on_report.inspection_zones)
    for a, b in zip(off_report.inspection_zones, on_report.inspection_zones):
        assert a.primary_type == b.primary_type
        assert a.anomaly_score == b.anomaly_score
    assert off_report.field.mean_crop_coverage == on_report.field.mean_crop_coverage
