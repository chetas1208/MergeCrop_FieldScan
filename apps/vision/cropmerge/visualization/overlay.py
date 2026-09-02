from __future__ import annotations

import cv2
import numpy as np

from cropmerge.pipeline.schemas import InspectionZone, SemanticClass
from cropmerge.segmentation.labels import CLASS_COLORS_BGR


def draw_segmentation(bgr: np.ndarray, label_map: np.ndarray, opacity: float = 0.35) -> np.ndarray:
    out = bgr.copy()
    overlay = bgr.copy()
    for cls, color in CLASS_COLORS_BGR.items():
        if cls == SemanticClass.UNKNOWN:
            continue
        sel = label_map == cls.value
        overlay[sel] = color
    return cv2.addWeighted(overlay, opacity, out, 1.0 - opacity, 0)


def draw_field_boundary(
    bgr: np.ndarray,
    field_mask: np.ndarray,
    *,
    label: str = "Analysis Field Boundary",
    confidence: str = "Medium",
) -> np.ndarray:
    out = bgr.copy()
    if field_mask is None or not np.any(field_mask):
        return out
    cnts, _ = cv2.findContours(field_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    # Boundary legend strip (top-left, below class legend if present)
    text = f"{label} ({confidence})"
    cv2.rectangle(out, (5, out.shape[0] - 42), (min(out.shape[1] - 5, 420), out.shape[0] - 24), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (10, out.shape[0] - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def draw_continuity_overlay(bgr: np.ndarray, structural, opacity: float = 0.45) -> np.ndarray:
    """Crop continuity mode — gaps and fragmented regions only when row visibility sufficient."""
    out = bgr.copy()
    if structural is None:
        return out
    if structural.row_visibility == "LOW":
        cv2.putText(
            out,
            "Row visibility LOW — fragmentation map only",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 0),
            1,
            cv2.LINE_AA,
        )
    overlay = out.copy()
    gap = structural.gap_mask.astype(np.uint8) * 255
    frag = (structural.fragmentation_mask > 0.15).astype(np.uint8) * 255
    overlay[gap > 0] = (0, 0, 255)
    overlay[frag > 0] = (0, 165, 255)
    out = cv2.addWeighted(overlay, opacity * 0.5, out, 1 - opacity * 0.5, 0)
    return out


def draw_zones(bgr: np.ndarray, zones: list[InspectionZone], opacity: float = 0.45) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    colors = {"high": (0, 0, 255), "medium": (0, 165, 255), "low": (0, 255, 255)}
    for z in zones:
        bb = z.bbox_norm
        x0 = int(bb["x"] * w)
        y0 = int(bb["y"] * h)
        x1 = int((bb["x"] + bb["w"]) * w)
        y1 = int((bb["y"] + bb["h"]) * h)
        color = colors.get(z.review_priority.value, (0, 255, 255))
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
        out = cv2.addWeighted(overlay, opacity * 0.35, out, 1 - opacity * 0.35, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        label = f"{z.id} {z.primary_signal_label[:24]}"
        cv2.putText(out, label, (x0 + 4, max(16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def draw_legend(bgr: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    items = [
        ("CROP", CLASS_COLORS_BGR[SemanticClass.CROP]),
        ("SOIL", CLASS_COLORS_BGR[SemanticClass.BARE_SOIL]),
        ("ROAD", CLASS_COLORS_BGR[SemanticClass.ROAD_PATH]),
        ("TREE", CLASS_COLORS_BGR[SemanticClass.TREE_VEGETATION]),
        ("WATER", CLASS_COLORS_BGR[SemanticClass.WATER]),
    ]
    x, y = 10, 20
    cv2.rectangle(out, (5, 5), (150, 20 + 18 * len(items)), (0, 0, 0), -1)
    for name, color in items:
        cv2.rectangle(out, (x, y - 10), (x + 14, y + 4), color, -1)
        cv2.putText(out, name, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return out


def draw_timestamp(bgr: np.ndarray, t: float) -> np.ndarray:
    out = bgr.copy()
    m = int(t // 60)
    s = t - m * 60
    text = f"{m:02d}:{s:04.1f}"
    cv2.putText(out, text, (out.shape[1] - 100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def annotate_frame(
    bgr: np.ndarray,
    label_map: np.ndarray | None,
    field_mask: np.ndarray | None,
    zones: list[InspectionZone],
    timestamp_sec: float,
    cfg: dict,
    structural=None,
    overlay_mode: str = "segmentation",
) -> np.ndarray:
    vcfg = cfg.get("visualization", {})
    out = bgr.copy()
    mode = overlay_mode or vcfg.get("default_overlay_mode", "segmentation")
    if mode == "continuity" and structural is not None:
        out = draw_continuity_overlay(out, structural, float(vcfg.get("zone_opacity", 0.45)))
    elif label_map is not None and mode != "variation":
        out = draw_segmentation(out, label_map, float(vcfg.get("mask_opacity", 0.35)))
    if field_mask is not None:
        conf = "Medium"
        if zones and zones[0].evidence.registration_confidence is not None:
            rc = zones[0].evidence.registration_confidence
            conf = "High" if rc >= 0.7 else ("Medium" if rc >= 0.4 else "Low")
        out = draw_field_boundary(out, field_mask, confidence=conf)
    out = draw_zones(out, zones, float(vcfg.get("zone_opacity", 0.45)))
    if vcfg.get("draw_legend", True):
        out = draw_legend(out)
    out = draw_timestamp(out, timestamp_sec)
    # Disclaimer strip
    cv2.putText(
        out,
        "Exploratory RGB visual analysis — not agronomic diagnosis",
        (10, out.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return out
