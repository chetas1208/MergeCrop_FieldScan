"""DINOv2 tile embeddings from local Meta checkpoint (public weights)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from cropmerge.features.dinov3 import EmbeddingExtractor, HeuristicEmbeddingExtractor

log = logging.getLogger("cropmerge.features.dinov2")

DEFAULT_CKPT = (
    Path(__file__).resolve().parents[2] / "models" / "dinov2_vitb14_pretrain.pth"
)


def dinov2_checkpoint() -> Path | None:
    env = os.environ.get("DINOV2_CHECKPOINT", "")
    for p in [Path(env) if env else None, DEFAULT_CKPT]:
        if p and p.is_file() and p.stat().st_size > 1_000_000:
            return p
    return None


def dinov2_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return dinov2_checkpoint() is not None


class DINOv2Extractor(EmbeddingExtractor):
    name = "dinov2"

    def __init__(self, allow_missing: bool = True):
        self._model = None
        self._fallback: HeuristicEmbeddingExtractor | None = None
        self.device = "cpu"
        self.is_real = False
        self._hf = False
        ckpt = dinov2_checkpoint()
        if ckpt is None:
            if not allow_missing:
                raise RuntimeError("DINOv2 checkpoint missing")
            log.warning("DINOv2 checkpoint missing — heuristic embeddings")
            self._fallback = HeuristicEmbeddingExtractor()
            return
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._load(ckpt)
            self.is_real = True
            log.info("DINOv2 ready on %s (%s)", self.device, ckpt.name)
        except Exception as e:
            log.warning("DINOv2 load failed (%s) — heuristic", e)
            if not allow_missing:
                raise
            self._fallback = HeuristicEmbeddingExtractor()

    def _load(self, ckpt: Path) -> None:
        import torch

        # Prefer torch.hub architecture + local weights
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14",
                pretrained=False,
                trust_repo=True,
                verbose=False,
            )
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                if "model" in state:
                    state = state["model"]
                elif "teacher" in state:
                    state = state["teacher"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                log.debug("DINOv2 missing keys: %d", len(missing))
            self._hf = False
            self._model = model.to(self.device).eval()
            return
        except Exception as e:
            log.info("torch.hub dinov2 failed (%s); trying transformers", e)

        try:
            from transformers import Dinov2Model

            model = Dinov2Model.from_pretrained("facebook/dinov2-base")
            self._hf = True
            self._model = model.to(self.device).eval()
            log.info("DINOv2 via transformers facebook/dinov2-base")
            return
        except Exception as e2:
            raise RuntimeError(f"Could not load DINOv2: hub={e}; hf={e2}") from e2

    def embed_tiles(
        self,
        bgr: np.ndarray,
        mask: np.ndarray | None,
        grid_rows: int,
        grid_cols: int,
    ) -> np.ndarray:
        if self._fallback is not None or self._model is None:
            assert self._fallback is not None
            return self._fallback.embed_tiles(bgr, mask, grid_rows, grid_cols)

        import torch

        h, w = bgr.shape[:2]
        if mask is None:
            mask = np.ones((h, w), dtype=bool)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        side = 224
        dim = 768
        out = np.zeros((grid_rows, grid_cols, dim), dtype=np.float32)
        cell_h, cell_w = h / grid_rows, w / grid_cols

        tensors: list = []
        coords: list[tuple[int, int]] = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
                x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
                m = mask[y0:y1, x0:x1]
                if m.size == 0 or float(np.mean(m)) < 0.25:
                    continue
                patch = rgb[y0:y1, x0:x1]
                # mask-aware: zero non-field for cleaner crop embedding
                if m.shape[:2] == patch.shape[:2]:
                    patch = patch.copy()
                    patch[~m] = mean  # neutral fill
                patch = cv2.resize(patch, (side, side), interpolation=cv2.INTER_AREA)
                patch = (patch - mean) / std
                tensors.append(torch.from_numpy(patch.transpose(2, 0, 1)))
                coords.append((r, c))

        if not tensors:
            return out

        # batched inference
        bs = 16
        all_feats = []
        with torch.inference_mode():
            for i in range(0, len(tensors), bs):
                batch = torch.stack(tensors[i : i + bs], dim=0).to(self.device)
                if self._hf:
                    feats = self._model(pixel_values=batch).last_hidden_state[:, 0]
                else:
                    if hasattr(self._model, "forward_features"):
                        ff = self._model.forward_features(batch)
                        if isinstance(ff, dict):
                            feats = ff.get("x_norm_clstoken")
                            if feats is None:
                                pt = ff.get("x_norm_patchtokens")
                                feats = pt.mean(dim=1) if pt is not None else None
                        else:
                            feats = ff
                    else:
                        feats = self._model(batch)
                    if isinstance(feats, (tuple, list)):
                        feats = feats[0]
                    if feats is not None and getattr(feats, "ndim", 0) == 3:
                        feats = feats[:, 0]
                if feats is None:
                    continue
                feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                all_feats.append(feats.detach().cpu().numpy().astype(np.float32))

        if not all_feats:
            return out
        arr = np.concatenate(all_feats, axis=0)
        for (r, c), vec in zip(coords, arr):
            if vec.shape[0] != dim:
                v = np.zeros(dim, dtype=np.float32)
                n = min(dim, vec.shape[0])
                v[:n] = vec[:n]
                vec = v / (np.linalg.norm(v) + 1e-8)
            out[r, c] = vec
        return out
