from __future__ import annotations

import cv2
import numpy as np


def render_heatmap(
    heat: np.ndarray,
    ref_bgr: np.ndarray | None = None,
    title: str = "Visual Field Variation",
) -> np.ndarray:
    """Image-relative field inspection map — NOT a crop health map."""
    hmap = heat.astype(np.float32)
    if hmap.size == 0:
        hmap = np.zeros((8, 8), dtype=np.float32)
    norm = hmap.copy()
    if norm.max() > norm.min():
        norm = (norm - norm.min()) / (norm.max() - norm.min())
    if ref_bgr is not None:
        th, tw = ref_bgr.shape[:2]
    else:
        th, tw = 480, 640
    up = cv2.resize(norm, (tw, th), interpolation=cv2.INTER_CUBIC)
    up_u8 = np.clip(up * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(up_u8, cv2.COLORMAP_TURBO)
    if ref_bgr is not None:
        base = ref_bgr.copy()
        color = cv2.addWeighted(color, 0.55, base, 0.45, 0)
    # Banner
    cv2.rectangle(color, (0, 0), (tw, 36), (20, 20, 20), -1)
    cv2.putText(color, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        color,
        "Image-relative · not georeferenced · not crop health",
        (12, th - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return color


def accumulate_heatmaps(heats: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if not heats:
        return np.zeros((8, 8), dtype=np.float32)
    acc = np.zeros_like(heats[0], dtype=np.float64)
    wsum = 0.0
    for h, w in zip(heats, weights):
        acc += h.astype(np.float64) * float(w)
        wsum += float(w)
    if wsum <= 0:
        return heats[0].astype(np.float32)
    return (acc / wsum).astype(np.float32)
