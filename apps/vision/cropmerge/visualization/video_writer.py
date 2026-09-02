from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_video(frames: list[np.ndarray], path: str | Path, fps: float = 2.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    # Try mp4v then avc1
    for fourcc_name in ("mp4v", "avc1", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), (w, h))
        if writer.isOpened():
            for fr in frames:
                if fr.shape[0] != h or fr.shape[1] != w:
                    fr = cv2.resize(fr, (w, h))
                writer.write(fr)
            writer.release()
            return path
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {path}")


def write_image(path: str | Path, bgr: np.ndarray, quality: int = 90) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(str(path), bgr)
    return path
