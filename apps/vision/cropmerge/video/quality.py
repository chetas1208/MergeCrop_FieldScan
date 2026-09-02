from __future__ import annotations

import cv2
import numpy as np

from cropmerge.pipeline.schemas import FrameQuality


def _laplacian_sharpness(gray: np.ndarray) -> float:
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Log-scale map: soft aerial + synthetic still score useful range
    return float(np.clip(np.log1p(var) / np.log1p(800.0), 0.0, 1.0))


def _exposure_stats(gray: np.ndarray) -> tuple[float, float, float, float]:
    mean_l = float(np.mean(gray))
    near_black = float(np.mean(gray < 15))
    saturated = float(np.mean(gray > 245))
    # Prefer mid-range luminance; penalize extremes + clipping
    lum_score = 1.0 - min(abs(mean_l - 128.0) / 128.0, 1.0)
    clip_pen = min(1.0, near_black + saturated)
    exposure = float(np.clip(lum_score * (1.0 - 0.7 * clip_pen), 0.0, 1.0))
    return mean_l, near_black, saturated, exposure


def analyze_frame_quality(
    bgr: np.ndarray,
    frame_index: int,
    timestamp_sec: float,
    cfg: dict,
    prev_gray: np.ndarray | None = None,
) -> tuple[FrameQuality, np.ndarray]:
    qcfg = cfg.get("quality", {})
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp = _laplacian_sharpness(gray)
    mean_l, near_black, saturated, exposure = _exposure_stats(gray)
    warnings: list[str] = []

    if sharp < float(qcfg.get("min_sharpness", 0.25)):
        warnings.append("low_sharpness")
    if near_black > float(qcfg.get("max_near_black", 0.35)):
        warnings.append("near_black_clipping")
    if saturated > float(qcfg.get("max_saturated", 0.25)):
        warnings.append("highlight_clipping")
    if mean_l < float(qcfg.get("min_luminance", 20.0)):
        warnings.append("underexposed")
    if mean_l > float(qcfg.get("max_luminance", 235.0)):
        warnings.append("overexposed")

    if prev_gray is not None and prev_gray.shape == gray.shape:
        # Extreme motion proxy via mean abs diff
        mad = float(np.mean(cv2.absdiff(prev_gray, gray))) / 255.0
        if mad > 0.35:
            warnings.append("extreme_interframe_motion")

    # Prefer down-weight over discard — temporal scoring still sees weak frames
    hard_fail = sharp < 0.05 and mean_l < 12
    if "extreme_interframe_motion" in warnings and sharp < 0.08:
        hard_fail = True
    usable = not hard_fail

    weight = 1.0
    if hard_fail:
        weight = float(qcfg.get("unusable_weight", 0.15))
    else:
        if "low_sharpness" in warnings:
            weight *= 0.65
        if "highlight_clipping" in warnings or "near_black_clipping" in warnings:
            weight *= 0.75
        if "extreme_interframe_motion" in warnings:
            weight *= 0.7
        weight = max(weight, 0.35)

    fq = FrameQuality(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        sharpness=round(sharp, 4),
        exposure_score=round(exposure, 4),
        mean_luminance=round(mean_l, 2),
        near_black_fraction=round(near_black, 4),
        saturated_fraction=round(saturated, 4),
        usable=usable,
        quality_weight=round(float(np.clip(weight, 0.0, 1.0)), 4),
        warnings=warnings,
    )
    return fq, gray


def analyze_sequence(frames: list, cfg: dict) -> list[FrameQuality]:
    out: list[FrameQuality] = []
    prev = None
    for fr in frames:
        fq, gray = analyze_frame_quality(fr.bgr, fr.index, fr.timestamp_sec, cfg, prev)
        out.append(fq)
        prev = gray
    return out
