from pathlib import Path

import cv2
import numpy as np
import pytest

from cropmerge.config import load_config
from cropmerge.pipeline.processor import FieldTriageProcessor
from cropmerge.video.metadata import extract_metadata


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0):
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert wr.isOpened()
    for f in frames:
        wr.write(f)
    wr.release()


def test_metadata_schema(tmp_path: Path):
    path = tmp_path / "t.mp4"
    frames = [np.full((120, 160, 3), 100, dtype=np.uint8) for _ in range(15)]
    _write_video(path, frames)
    meta = extract_metadata(path)
    assert meta.width == 160
    assert meta.height == 120
    assert meta.fps > 0
    assert meta.filename == "t.mp4"


def test_unsupported_extension(tmp_path: Path):
    p = tmp_path / "x.avi"
    p.write_bytes(b"nope")
    with pytest.raises(ValueError):
        extract_metadata(p)


def test_no_field_urban(tmp_path: Path):
    """Urban/gray video should not invent farm zones."""
    path = tmp_path / "urban.mp4"
    frames = []
    for i in range(20):
        img = np.full((180, 240, 3), 90, dtype=np.uint8)
        # buildings-like blocks
        img[20:80, 30:90] = 160
        img[100:160, 120:200] = 140
        frames.append(img)
    _write_video(path, frames)
    out = tmp_path / "out"
    cfg = load_config()
    cfg["video"]["sample_fps"] = 5
    cfg["temporal"]["min_persistent_frames"] = 3
    report = FieldTriageProcessor(cfg).process(
        path, out, segmentation_backend="heuristic", max_frames=10, sample_fps=5
    )
    # Either no field or no zones
    if not report.field.detected:
        assert report.inspection_zones == []
    assert report.artifacts.results_json
    assert Path(report.artifacts.results_json).is_file()


def test_pipeline_synthetic_field(tmp_path: Path):
    from scripts.generate_demo_video import make_frame

    path = tmp_path / "field.mp4"
    frames = [make_frame(i / 10.0) for i in range(30)]
    _write_video(path, frames, fps=10)
    out = tmp_path / "out"
    cfg = load_config()
    cfg["temporal"]["min_persistent_frames"] = 2
    cfg["anomaly"]["medium_threshold"] = 0.15
    report = FieldTriageProcessor(cfg).process(
        path,
        out,
        sample_fps=5,
        max_frames=15,
        segmentation_backend="heuristic",
        skip_dino=True,
    )
    assert report.schema_version == "1.0"
    assert report.analysis.frames_sampled > 0
    assert Path(report.artifacts.results_json).exists()
    assert report.disclaimer
    # Product language
    blob = str(report.to_camel_dict()).lower()
    assert "disease" not in blob or "not" in report.disclaimer.lower()
