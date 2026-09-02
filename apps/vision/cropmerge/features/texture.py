from __future__ import annotations

import cv2
import numpy as np


def texture_features(bgr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if mask is None:
        mask = np.ones(gray.shape, dtype=bool)
    if not np.any(mask):
        return {"variance": 0.0, "grad_mag": 0.0, "edge_density": 0.0, "entropy": 0.0}

    vals = gray[mask]
    variance = float(np.var(vals) / (255.0 ** 2))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    grad_mag = float(np.mean(mag[mask]) / 255.0)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edge_density = float(np.mean(edges[mask] > 0))

    # Simple histogram entropy
    hist, _ = np.histogram(vals, bins=32, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist + 1e-12)) / 5.0)  # ~normalize
    return {
        "variance": variance,
        "grad_mag": grad_mag,
        "edge_density": edge_density,
        "entropy": float(np.clip(entropy, 0.0, 1.0)),
    }
