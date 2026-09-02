from __future__ import annotations

import cv2
import numpy as np


def excess_green(bgr: np.ndarray) -> np.ndarray:
    """Excess Green (ExG) visual proxy — NOT NDVI."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return 2.0 * g - r - b


def vegetation_mask(exg: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    return exg > threshold


def color_stats(bgr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is None:
        mask = np.ones(bgr.shape[:2], dtype=bool)
    if not np.any(mask):
        return {
            "r_mean": 0.0, "g_mean": 0.0, "b_mean": 0.0,
            "h_mean": 0.0, "s_mean": 0.0, "v_mean": 0.0,
            "l_mean": 0.0, "a_mean": 0.0, "bb_mean": 0.0,
            "g_std": 0.0, "exg_mean": 0.0, "exg_std": 0.0,
        }
    pix = bgr[mask]
    rgb = pix[:, ::-1].astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[mask].astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[mask].astype(np.float32)
    exg = excess_green(bgr)[mask]
    return {
        "r_mean": float(np.mean(rgb[:, 0])),
        "g_mean": float(np.mean(rgb[:, 1])),
        "b_mean": float(np.mean(rgb[:, 2])),
        "g_std": float(np.std(rgb[:, 1])),
        "h_mean": float(np.mean(hsv[:, 0])),
        "s_mean": float(np.mean(hsv[:, 1])),
        "v_mean": float(np.mean(hsv[:, 2])),
        "l_mean": float(np.mean(lab[:, 0])),
        "a_mean": float(np.mean(lab[:, 1])),
        "bb_mean": float(np.mean(lab[:, 2])),
        "exg_mean": float(np.mean(exg)),
        "exg_std": float(np.std(exg)),
    }


def lab_distance(stats_a: dict[str, float], stats_b: dict[str, float]) -> float:
    da = stats_a.get("a_mean", 0) - stats_b.get("a_mean", 0)
    db = stats_a.get("bb_mean", 0) - stats_b.get("bb_mean", 0)
    dl = stats_a.get("l_mean", 0) - stats_b.get("l_mean", 0)
    # Lab L~[0,255] in OpenCV
    return float(np.sqrt((dl / 255.0) ** 2 + (da / 255.0) ** 2 + (db / 255.0) ** 2))
