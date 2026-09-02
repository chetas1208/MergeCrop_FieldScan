from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import cv2
import numpy as np

log = logging.getLogger("cropmerge.features.dinov3")


def dinov3_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    ckpt = os.environ.get("DINOV3_CHECKPOINT", "")
    return bool(ckpt and os.path.isfile(ckpt))


class EmbeddingExtractor(ABC):
    name: str = "base"

    @abstractmethod
    def embed_tiles(
        self,
        bgr: np.ndarray,
        mask: np.ndarray | None,
        grid_rows: int,
        grid_cols: int,
    ) -> np.ndarray:
        """Return (rows, cols, dim) embeddings; zeros where mask insufficient."""


class HeuristicEmbeddingExtractor(EmbeddingExtractor):
    """
    Dense-ish local descriptor from color/texture histograms.
    Explicit fallback — never labeled as DINOv3.
    """

    name = "heuristic"

    def embed_tiles(
        self,
        bgr: np.ndarray,
        mask: np.ndarray | None,
        grid_rows: int,
        grid_cols: int,
    ) -> np.ndarray:
        h, w = bgr.shape[:2]
        if mask is None:
            mask = np.ones((h, w), dtype=bool)
        dim = 32
        out = np.zeros((grid_rows, grid_cols, dim), dtype=np.float32)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cell_h, cell_w = h / grid_rows, w / grid_cols
        for r in range(grid_rows):
            for c in range(grid_cols):
                y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
                x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
                m = mask[y0:y1, x0:x1]
                if m.size == 0 or float(np.mean(m)) < 0.25:
                    continue
                patch = bgr[y0:y1, x0:x1]
                ph = hsv[y0:y1, x0:x1]
                pg = gray[y0:y1, x0:x1]
                vec = []
                for ch in range(3):
                    hist, _ = np.histogram(patch[:, :, ch][m], bins=8, range=(0, 256), density=True)
                    vec.extend(hist.tolist())
                hhist, _ = np.histogram(ph[:, :, 0][m], bins=4, range=(0, 180), density=True)
                vec.extend(hhist.tolist())
                ghist, _ = np.histogram(pg[m], bins=4, range=(0, 256), density=True)
                vec.extend(ghist.tolist())
                arr = np.array(vec, dtype=np.float32)
                if arr.size < dim:
                    arr = np.pad(arr, (0, dim - arr.size))
                else:
                    arr = arr[:dim]
                n = np.linalg.norm(arr) + 1e-8
                out[r, c] = arr / n
        return out


class DINOv3Extractor(EmbeddingExtractor):
    name = "dinov3"

    def __init__(self, allow_missing: bool = True):
        self._fallback = None
        if dinov3_available():
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                log.info("DINOv3 env detected on %s", self.device)
                # Checkpoint schema varies; use explicit fallback until bound
                self._fallback = HeuristicEmbeddingExtractor()
                log.warning("DINOv3 checkpoint present but adapter uses documented heuristic until model bound")
            except Exception as e:
                if not allow_missing:
                    raise
                log.warning("DINOv3 init failed (%s); heuristic fallback", e)
                self._fallback = HeuristicEmbeddingExtractor()
        elif allow_missing:
            log.warning("DINOv3 unavailable — heuristic embeddings (is_fallback)")
            self._fallback = HeuristicEmbeddingExtractor()
        else:
            raise RuntimeError("DINOv3 not available. Set DINOV3_CHECKPOINT or use heuristic.")

    def embed_tiles(
        self,
        bgr: np.ndarray,
        mask: np.ndarray | None,
        grid_rows: int,
        grid_cols: int,
    ) -> np.ndarray:
        assert self._fallback is not None
        return self._fallback.embed_tiles(bgr, mask, grid_rows, grid_cols)


def create_embedder(backend: str, allow_fallback: bool = True) -> EmbeddingExtractor:
    backend = (backend or "auto").lower()
    if backend in {"auto"}:
        try:
            from cropmerge.features.dinov2 import DINOv2Extractor, dinov2_available

            if dinov2_available():
                return DINOv2Extractor(allow_missing=True)
        except Exception:
            pass
        from cropmerge.features.dense_rgb import DenseRGBExtractor

        return DenseRGBExtractor()
    if backend in {"dense_rgb", "rgb"}:
        from cropmerge.features.dense_rgb import DenseRGBExtractor

        return DenseRGBExtractor()
    if backend in {"heuristic", "mock"}:
        return HeuristicEmbeddingExtractor()
    if backend in {"dinov2"}:
        from cropmerge.features.dinov2 import DINOv2Extractor

        return DINOv2Extractor(allow_missing=allow_fallback)
    if backend in {"dinov3", "dino"}:
        return DINOv3Extractor(allow_missing=allow_fallback)
    raise ValueError(f"Unknown dino backend: {backend}")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(1.0 - np.dot(a, b) / (na * nb))
