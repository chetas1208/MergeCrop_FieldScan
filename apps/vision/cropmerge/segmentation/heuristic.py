from __future__ import annotations

import cv2
import numpy as np

from cropmerge.pipeline.schemas import SemanticClass
from cropmerge.segmentation.base import FrameSegmentation, SegmentMask, Segmenter
from cropmerge.segmentation.postprocess import finalize_segmentation


class HeuristicSegmenter(Segmenter):
    """
    Explicit RGB heuristic backend for offline demos without SAM weights.
    Marked is_fallback=True — never claimed as SAM 3.1.
    """

    name = "heuristic"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def segment_image(self, bgr: np.ndarray, frame_index: int = 0, timestamp_sec: float = 0.0) -> FrameSegmentation:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        exg = 2 * g - r - b
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hch, sch, vch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        crop = (exg > 0.05) & (g > r) & (g > b * 0.9) & (vch > 30)
        bare = (exg < 0.02) & (sch < 80) & (vch > 40) & (vch < 210) & (~crop)
        # Dirt roads: elongated mid-luma low-green strips
        edges = cv2.Canny(gray, 60, 140)
        edge_dense = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
        road = bare & edge_dense & (vch > 50) & (vch < 180)
        # Trees: darker green, high local variance
        var = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (0, 0), 3) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3) ** 2
        trees = (exg > 0.02) & (vch < 90) & (var > 200) & (g > r)
        water = (b > r) & (b > g) & (sch > 40) & (vch < 160) & (exg < 0.05)
        # Simple building proxy: low sat high edge
        infra = (sch < 40) & edge_dense & (vch > 80) & (~crop) & (~water)

        # Morph clean
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        def clean(m: np.ndarray) -> np.ndarray:
            u = m.astype(np.uint8) * 255
            u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
            u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=2)
            return u > 0

        crop, bare, road, trees, water, infra = map(clean, [crop, bare, road, trees, water, infra])
        field = clean(crop | bare)

        masks = [
            SegmentMask(SemanticClass.FIELD, field, 0.7, "heuristic:field"),
            SegmentMask(SemanticClass.CROP, crop, 0.75, "heuristic:crop"),
            SegmentMask(SemanticClass.BARE_SOIL, bare & ~road, 0.65, "heuristic:bare_soil"),
            SegmentMask(SemanticClass.ROAD_PATH, road, 0.55, "heuristic:road"),
            SegmentMask(SemanticClass.TREE_VEGETATION, trees & ~crop, 0.55, "heuristic:trees"),
            SegmentMask(SemanticClass.WATER, water, 0.5, "heuristic:water"),
            SegmentMask(SemanticClass.INFRASTRUCTURE, infra, 0.45, "heuristic:infra"),
        ]
        seg = FrameSegmentation(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            masks=masks,
            backend=self.name,
            is_fallback=True,
        )
        return finalize_segmentation(seg, self.cfg)

    def start_video(self, frames_bgr: list[np.ndarray]) -> None:
        self._frames = frames_bgr

    def track_video(self, frames_meta: list[tuple[int, float]]) -> list[FrameSegmentation]:
        frames = getattr(self, "_frames", [])
        out = []
        for (idx, ts), bgr in zip(frames_meta, frames):
            out.append(self.segment_image(bgr, idx, ts))
        return out
