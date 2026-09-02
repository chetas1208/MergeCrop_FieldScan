from __future__ import annotations

import logging
import os

import numpy as np

from cropmerge.segmentation.base import FrameSegmentation, Segmenter
from cropmerge.segmentation.heuristic import HeuristicSegmenter

log = logging.getLogger("cropmerge.segmentation.sam3")


def sam3_available() -> bool:
    """True only when torch + optional sam3 package + checkpoint present."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    ckpt = os.environ.get("SAM3_CHECKPOINT", "")
    if ckpt and os.path.isfile(ckpt):
        return True
    try:
        import sam3  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


class SAM3Segmenter(Segmenter):
    """
    SAM 3.1 adapter.

    Real weights require Meta SAM3 install + checkpoint.
    If unavailable, refuses silent fake — caller must choose heuristic explicitly
    or use create_segmenter() which logs fallback.
    """

    name = "sam3"

    def __init__(self, cfg: dict | None = None, allow_missing: bool = False):
        self.cfg = cfg or {}
        self._impl = None
        self._fallback: HeuristicSegmenter | None = None
        if sam3_available():
            self._try_load()
        elif allow_missing:
            log.warning(
                "SAM 3.1 unavailable (install sam3 + set SAM3_CHECKPOINT). "
                "Using explicit heuristic fallback — outputs marked is_fallback=True."
            )
            self._fallback = HeuristicSegmenter(cfg)
        else:
            raise RuntimeError(
                "SAM 3.1 not available. Install Meta sam3, set SAM3_CHECKPOINT, "
                "or pass --segmentation-backend heuristic."
            )

    def _try_load(self) -> None:
        # Placeholder for real SAM3 API wiring when package present.
        # Keep isolated so missing deps don't break import of the rest.
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("SAM3 environment detected on %s — full API wiring pending checkpoint schema", self.device)
            # Until official lightweight API is pinned, fall back explicitly
            self._fallback = HeuristicSegmenter(self.cfg)
            log.warning("SAM3 package detected but runtime adapter uses documented fallback until checkpoint bound")
        except Exception as e:
            log.error("SAM3 init failed: %s", e)
            raise

    def segment_image(self, bgr: np.ndarray, frame_index: int = 0, timestamp_sec: float = 0.0) -> FrameSegmentation:
        if self._fallback is not None:
            seg = self._fallback.segment_image(bgr, frame_index, timestamp_sec)
            seg.backend = "sam3_fallback_heuristic"
            seg.is_fallback = True
            return seg
        raise RuntimeError("SAM3 backend not initialized")

    def start_video(self, frames_bgr: list[np.ndarray]) -> None:
        if self._fallback:
            self._fallback.start_video(frames_bgr)

    def track_video(self, frames_meta: list[tuple[int, float]]) -> list[FrameSegmentation]:
        if self._fallback:
            segs = self._fallback.track_video(frames_meta)
            for s in segs:
                s.backend = "sam3_fallback_heuristic"
                s.is_fallback = True
            return segs
        raise RuntimeError("SAM3 backend not initialized")


def create_segmenter(backend: str, cfg: dict, allow_fallback: bool = True) -> Segmenter:
    backend = (backend or "field_cv").lower()
    if backend in {"field_cv", "cv", "auto"}:
        from cropmerge.segmentation.field_cv import FieldCVSegmenter

        return FieldCVSegmenter(cfg)
    if backend == "heuristic":
        return HeuristicSegmenter(cfg)
    if backend == "mock":
        return HeuristicSegmenter(cfg)
    if backend in {"sam2", "sam2.1"}:
        from cropmerge.segmentation.sam2 import SAM2Segmenter

        return SAM2Segmenter(cfg, allow_missing=allow_fallback)
    if backend in {"sam3", "sam3.1", "sam"}:
        if sam3_available() or allow_fallback:
            return SAM3Segmenter(cfg, allow_missing=allow_fallback)
        raise RuntimeError("SAM3 requested but unavailable")
    raise ValueError(f"Unknown segmentation backend: {backend}")
