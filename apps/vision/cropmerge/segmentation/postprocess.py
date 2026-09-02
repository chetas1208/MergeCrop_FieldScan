from __future__ import annotations

import cv2
import numpy as np

from cropmerge.pipeline.schemas import SemanticClass
from cropmerge.segmentation.base import FrameSegmentation, SegmentMask
from cropmerge.segmentation.labels import CLASS_PRIORITY


def resolve_overlaps(masks: list[SegmentMask], shape: tuple[int, int]) -> np.ndarray:
    """Build exclusive label map; higher priority classes win."""
    h, w = shape
    label_map = np.full((h, w), SemanticClass.UNKNOWN.value, dtype=object)
    priority = np.zeros((h, w), dtype=np.int16)
    for m in sorted(masks, key=lambda x: CLASS_PRIORITY.get(x.label, 0)):
        if m.mask is None or m.mask.shape[:2] != (h, w):
            continue
        sel = m.mask.astype(bool)
        p = CLASS_PRIORITY.get(m.label, 0)
        win = sel & (p >= priority)
        label_map[win] = m.label.value
        priority[win] = p
    return label_map


def build_field_mask(label_map: np.ndarray, union_classes: list[str] | None = None, cfg: dict | None = None) -> np.ndarray:
    """
    Conservative analysis field boundary — analyzable region, not property boundary.

    Removes tiny speckle, fills only small internal holes, preserves large gaps
    (missing crop / roads) and avoids aggressive closing.
    """
    union = set(union_classes or ["CROP", "BARE_SOIL", "FIELD"])
    field = np.isin(label_map, list(union))
    bcfg = (cfg or {}).get("boundary", {})
    min_comp_frac = float(bcfg.get("min_component_frac", 0.02))
    max_hole_frac = float(bcfg.get("max_hole_fill_frac", 0.008))

    h, w = field.shape
    total = max(h * w, 1)
    min_area = int(min_comp_frac * total)
    max_hole = int(max_hole_frac * total)

    field_u8 = field.astype(np.uint8) * 255
    # Light open only — remove 1–2 px speckle without closing large gaps
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    field_u8 = cv2.morphologyEx(field_u8, cv2.MORPH_OPEN, k3, iterations=1)

    # Fill only small internal holes
    inv = cv2.bitwise_not(field_u8)
    n_h, lab_h, stats_h, _ = cv2.connectedComponentsWithStats(inv, 8)
    for i in range(1, n_h):
        area = stats_h[i, cv2.CC_STAT_AREA]
        if area <= max_hole:
            # hole not touching image border → internal
            x, y, bw, bh, _ = stats_h[i]
            touches_border = x <= 0 or y <= 0 or x + bw >= w or y + bh >= h
            if not touches_border:
                field_u8[lab_h == i] = 255

    # Drop tiny disconnected field components
    n, labels, stats, _ = cv2.connectedComponentsWithStats((field_u8 > 0).astype(np.uint8), 8)
    if n <= 1:
        return field_u8 > 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max()) if len(areas) else 0
    keep_thr = max(min_area, int(0.15 * largest))
    out = np.zeros_like(field_u8, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= keep_thr:
            out[labels == i] = True
    return out


def boundary_confidence(field_mask: np.ndarray, label_map: np.ndarray | None = None) -> str:
    """High / Medium / Low — no fake numeric precision."""
    if not np.any(field_mask):
        return "Low"
    frac = float(np.mean(field_mask))
    if frac < 0.05 or frac > 0.98:
        return "Low"
    if label_map is not None:
        edge = cv2.Canny(field_mask.astype(np.uint8) * 255, 50, 150)
        edge_frac = float(np.sum(edge > 0)) / max(np.sum(field_mask), 1)
        if edge_frac > 0.35:
            return "Medium"
    if frac >= 0.12:
        return "High"
    return "Medium"


def finalize_segmentation(seg: FrameSegmentation, cfg: dict) -> FrameSegmentation:
    if not seg.masks:
        h = seg.label_map.shape[0] if seg.label_map is not None else 0
        w = seg.label_map.shape[1] if seg.label_map is not None else 0
        if h == 0:
            return seg
    shape = seg.masks[0].mask.shape[:2] if seg.masks else seg.label_map.shape[:2]
    seg.label_map = resolve_overlaps(seg.masks, shape)
    union = cfg.get("segmentation", {}).get("field_union_classes", ["CROP", "BARE_SOIL", "FIELD"])
    seg.field_mask = build_field_mask(seg.label_map, union, cfg)
    seg.crop_mask = seg.label_map == SemanticClass.CROP.value
    return seg


def class_fractions(label_map: np.ndarray) -> dict[str, float]:
    flat = label_map.reshape(-1)
    total = max(flat.size, 1)
    out: dict[str, float] = {}
    for c in SemanticClass:
        out[c.value] = float(np.mean(flat == c.value))
    out["_total_pixels"] = float(total)
    return out
