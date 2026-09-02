"""Procedural synthetic field imagery + pixel-level semantic ground truth."""
from __future__ import annotations

import numpy as np

from cropmerge.pipeline.schemas import SemanticClass


def _regions(t: float, w: int, h: int) -> dict[str, np.ndarray]:
    y, x = np.mgrid[0:h, 0:w]
    shift = int(20 * np.sin(t * 0.7))
    xx = x + shift

    cy, cx = int(0.28 * h), int(0.72 * w) + shift // 2
    soil = ((y - cy) ** 2 / (50**2) + (x - cx) ** 2 / (60**2)) < 1.0

    cy2, cx2 = int(0.72 * h), int(0.25 * w)
    thin = ((y - cy2) ** 2 / (40**2) + (x - cx2) ** 2 / (48**2)) < 1.0

    road = (y > h * 0.55) & (y < h * 0.62) & (x > w * 0.05) & (x < w * 0.95)
    trees = (y < h * 0.12) & (np.sin(xx / 10.0) > -0.15)

    return {"soil": soil, "thin": thin, "road": road, "trees": trees}


def make_synthetic_labels(t: float, w: int = 640, h: int = 360) -> np.ndarray:
    """Exclusive semantic label map aligned with generate_demo_video.py regions."""
    regions = _regions(t, w, h)
    label_map = np.full((h, w), SemanticClass.CROP.value, dtype=object)
    label_map[regions["soil"] | regions["thin"]] = SemanticClass.BARE_SOIL.value
    label_map[regions["road"]] = SemanticClass.ROAD_PATH.value
    label_map[regions["trees"]] = SemanticClass.TREE_VEGETATION.value
    return label_map


def make_synthetic_frame(t: float, w: int = 640, h: int = 360) -> np.ndarray:
    """BGR frame — same procedural recipe as scripts/generate_demo_video.py."""
    import cv2

    y, x = np.mgrid[0:h, 0:w]
    shift = int(20 * np.sin(t * 0.7))
    xx = x + shift

    row = 25 * (((xx // 4 + y // 3) % 2) * 2 - 1)
    g = 110 + 35 * np.sin((xx + y * 0.2) / 6.0) + 12 * np.sin(y / 15.0) + row * 0.35
    r = 45 + 12 * np.sin(xx / 22.0)
    b = 40 + 10 * np.cos(y / 18.0)
    img = np.stack([b, g, r], axis=-1).astype(np.float32)

    regions = _regions(t, w, h)
    img[regions["soil"]] = np.array([55, 100, 170], dtype=np.float32)
    img[regions["thin"]] = img[regions["thin"]] * 0.4 + np.array([45, 65, 120], dtype=np.float32) * 0.6
    img[regions["road"]] = np.array([75, 90, 105], dtype=np.float32)
    img[regions["road"]] += (((x[regions["road"]] // 3) % 2) * 8 - 4)[:, None]
    img[regions["trees"]] = np.array([20, 50, 15], dtype=np.float32)

    rng = np.random.default_rng(int(t * 1000) % 10000)
    img += rng.normal(0, 4.0, img.shape).astype(np.float32)
    out = np.clip(img, 0, 255).astype(np.uint8)
    if int(t * 10) % 23 == 0:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def make_synthetic_pair(t: float, w: int = 640, h: int = 360) -> tuple[np.ndarray, np.ndarray]:
    return make_synthetic_frame(t, w, h), make_synthetic_labels(t, w, h)
