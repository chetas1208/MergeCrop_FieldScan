from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cropmerge.video.metadata import extract_metadata


@dataclass
class SampledFrame:
    index: int
    source_frame_index: int
    timestamp_sec: float
    bgr: np.ndarray


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sample_frames(
    path: str | Path,
    sample_fps: float = 2.0,
    max_frames: int | None = 120,
) -> tuple[list[SampledFrame], object]:
    """Decode video (or still image) and sample at approximately sample_fps."""
    path = Path(path)
    meta = extract_metadata(path)
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    # Still images: repeat the frame so temporal persistence can form zones
    if path.suffix.lower() in IMAGE_EXTS:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"OpenCV cannot read image {path}")
        # Default 3 frames matches temporal.min_persistent_frames
        if max_frames is None:
            n = 3
        else:
            n = max(1, min(int(max_frames), 8))
            if n < 3:
                n = 3
        frames = [
            SampledFrame(
                index=i,
                source_frame_index=0,
                timestamp_sec=float(i) * 0.5,
                bgr=bgr.copy(),
            )
            for i in range(n)
        ]
        return frames, meta

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")

    src_fps = meta.fps if meta.fps > 1e-3 else 30.0
    step = max(1, int(round(src_fps / sample_fps)))
    frames: list[SampledFrame] = []
    i = 0
    kept = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            ts = i / src_fps
            frames.append(
                SampledFrame(
                    index=kept,
                    source_frame_index=i,
                    timestamp_sec=float(ts),
                    bgr=bgr.copy(),
                )
            )
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames, meta
