"""SAM 2.1 backend — real Meta weights (public). Never labeled as SAM 3.1."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from cropmerge.pipeline.schemas import SemanticClass
from cropmerge.segmentation.base import FrameSegmentation, SegmentMask, Segmenter
from cropmerge.segmentation.heuristic import HeuristicSegmenter
from cropmerge.segmentation.postprocess import finalize_segmentation

log = logging.getLogger("cropmerge.segmentation.sam2")

DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "models" / "sam2.1_hiera_small.pt"


def sam2_checkpoint() -> Path | None:
    env = os.environ.get("SAM2_CHECKPOINT", "")
    for p in [Path(env) if env else None, DEFAULT_CKPT]:
        if p and p.is_file():
            return p
    return None


def sam2_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return sam2_checkpoint() is not None


class SAM2Segmenter(Segmenter):
    """
    SAM 2.1 automatic-mask + RGB class assignment.

    Uses official `sam2` package when installed; otherwise a torch-feature
    hybrid still powered by the downloaded SAM2 checkpoint is not assumed —
    we use a DINO-free grid proposal + SAM2 if package present.
    """

    name = "sam2"

    def __init__(self, cfg: dict | None = None, allow_missing: bool = True):
        self.cfg = cfg or {}
        self._predictor = None
        self._fallback: HeuristicSegmenter | None = None
        self._frames: list[np.ndarray] = []
        self.device = "cpu"
        self.is_real = False

        ckpt = sam2_checkpoint()
        if ckpt is None:
            if not allow_missing:
                raise RuntimeError("SAM2 checkpoint missing. Run scripts/download_weights.py")
            log.warning("SAM2 checkpoint missing — heuristic fallback")
            self._fallback = HeuristicSegmenter(cfg)
            return

        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._init_sam2(ckpt)
        except Exception as e:
            log.warning("SAM2 init failed (%s) — heuristic fallback", e)
            if not allow_missing:
                raise
            self._fallback = HeuristicSegmenter(cfg)

    def _init_sam2(self, ckpt: Path) -> None:
        """Try official sam2; else torch-only automatic mask via hydra-free path."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            # Prefer small config bundled with package
            model_cfg = os.environ.get("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_s.yaml")
            sam = build_sam2(model_cfg, str(ckpt), device=self.device)
            self._predictor = SAM2AutomaticMaskGenerator(
                model=sam,
                points_per_side=16,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.85,
                min_mask_region_area=200,
            )
            self.is_real = True
            log.info("SAM2.1 AMG ready on %s (%s)", self.device, ckpt.name)
            return
        except Exception as e:
            log.info("Official sam2 package path unavailable (%s); using torch mask hybrid", e)

        # Torch hybrid: load state dict if possible is package-specific;
        # use high-quality multi-scale heuristic boosted by edge/energy — still fallback.
        # Keep honest labeling.
        self._fallback = HeuristicSegmenter(self.cfg)
        self._sam2_ckpt_present = True
        log.warning(
            "SAM2 weights present at %s but sam2 package not importable. "
            "Install: pip install 'git+https://github.com/facebookresearch/sam2.git'. "
            "Using heuristic until package installed.",
            ckpt,
        )

    def segment_image(self, bgr: np.ndarray, frame_index: int = 0, timestamp_sec: float = 0.0) -> FrameSegmentation:
        if self._fallback is not None and not self.is_real:
            seg = self._fallback.segment_image(bgr, frame_index, timestamp_sec)
            seg.backend = "sam2_weights_present_heuristic_runtime"
            seg.is_fallback = True
            return seg

        assert self._predictor is not None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks = self._predictor.generate(rgb)
        seg_masks = self._masks_to_semantic(bgr, masks)
        seg = FrameSegmentation(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            masks=seg_masks,
            backend="sam2",
            is_fallback=False,
        )
        return finalize_segmentation(seg, self.cfg)

    def _masks_to_semantic(self, bgr: np.ndarray, amg_masks: list[dict]) -> list[SegmentMask]:
        """Map SAM instances → V1 semantic classes via RGB cues inside each mask."""
        h, w = bgr.shape[:2]
        if not amg_masks:
            return HeuristicSegmenter(self.cfg).segment_image(bgr).masks

        # Sort large → small
        amg_masks = sorted(amg_masks, key=lambda m: int(m.get("area", 0)), reverse=True)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        exg = 2 * g - r - b
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        buckets: dict[SemanticClass, np.ndarray] = {
            c: np.zeros((h, w), dtype=bool)
            for c in [
                SemanticClass.CROP,
                SemanticClass.BARE_SOIL,
                SemanticClass.ROAD_PATH,
                SemanticClass.TREE_VEGETATION,
                SemanticClass.WATER,
                SemanticClass.INFRASTRUCTURE,
                SemanticClass.FIELD,
            ]
        }

        for m in amg_masks[:40]:
            mask = m.get("segmentation")
            if mask is None:
                continue
            mask = mask.astype(bool)
            if mask.shape[:2] != (h, w) or not np.any(mask):
                continue
            conf = float(m.get("predicted_iou", m.get("stability_score", 0.7)))
            ex = float(np.mean(exg[mask]))
            sat = float(np.mean(hsv[:, :, 1][mask]))
            val = float(np.mean(hsv[:, :, 2][mask]))
            blue = float(np.mean(b[mask]) - np.mean(g[mask]))
            var = float(np.var(gray[mask]))
            area_frac = float(np.mean(mask))

            if blue > 0.05 and val < 160:
                label = SemanticClass.WATER
            elif ex > 0.06 and val < 95 and var > 180:
                label = SemanticClass.TREE_VEGETATION
            elif ex > 0.04:
                label = SemanticClass.CROP
            elif sat < 55 and var > 250 and area_frac < 0.15:
                label = SemanticClass.INFRASTRUCTURE
            elif sat < 90 and 40 < val < 190 and (area_frac < 0.12 or self._elongated(mask)):
                label = SemanticClass.ROAD_PATH
            elif ex < 0.03:
                label = SemanticClass.BARE_SOIL
            else:
                label = SemanticClass.CROP

            buckets[label] |= mask
            if label in (SemanticClass.CROP, SemanticClass.BARE_SOIL):
                buckets[SemanticClass.FIELD] |= mask

        out: list[SegmentMask] = []
        for label, mask in buckets.items():
            if np.any(mask):
                out.append(SegmentMask(label=label, mask=mask, confidence=0.8, prompt=f"sam2:{label.value}"))
        if not out:
            return HeuristicSegmenter(self.cfg).segment_image(bgr).masks
        return out

    @staticmethod
    def _elongated(mask: np.ndarray) -> bool:
        ys, xs = np.where(mask)
        if len(xs) < 20:
            return False
        return (xs.max() - xs.min() + 1) / max(ys.max() - ys.min() + 1, 1) > 3.0 or (
            ys.max() - ys.min() + 1
        ) / max(xs.max() - xs.min() + 1, 1) > 3.0

    def start_video(self, frames_bgr: list[np.ndarray]) -> None:
        self._frames = frames_bgr
        if self._fallback and not self.is_real:
            self._fallback.start_video(frames_bgr)

    def track_video(self, frames_meta: list[tuple[int, float]]) -> list[FrameSegmentation]:
        if self._fallback and not self.is_real:
            segs = self._fallback.track_video(frames_meta)
            for s in segs:
                s.backend = "sam2_weights_present_heuristic_runtime"
                s.is_fallback = True
            return segs
        out = []
        for (idx, ts), bgr in zip(frames_meta, self._frames):
            out.append(self.segment_image(bgr, idx, ts))
        return out
