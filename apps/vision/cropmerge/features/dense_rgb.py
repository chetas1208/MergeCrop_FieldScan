"""
Dense RGB descriptor bank when DINOv2 unavailable.

Still real analysis: multi-scale color histograms + LBP-lite + gradient orients,
PCA-whitened to 64-D. Never labeled as DINO.
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.decomposition import PCA

from cropmerge.features.dinov3 import EmbeddingExtractor


class DenseRGBExtractor(EmbeddingExtractor):
    name = "dense_rgb"

    def __init__(self):
        self._pca: PCA | None = None

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
        dim = 64
        out = np.zeros((grid_rows, grid_cols, dim), dtype=np.float32)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)

        cell_h, cell_w = h / grid_rows, w / grid_cols
        raw = []
        coords = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
                x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
                m = mask[y0:y1, x0:x1]
                if m.size == 0 or float(np.mean(m)) < 0.25:
                    continue
                vec = []
                for img, bins, rng in [
                    (bgr[y0:y1, x0:x1, 0], 8, (0, 256)),
                    (bgr[y0:y1, x0:x1, 1], 8, (0, 256)),
                    (bgr[y0:y1, x0:x1, 2], 8, (0, 256)),
                    (hsv[y0:y1, x0:x1, 0], 8, (0, 180)),
                    (lab[y0:y1, x0:x1, 1], 6, (0, 256)),
                    (lab[y0:y1, x0:x1, 2], 6, (0, 256)),
                ]:
                    hist, _ = np.histogram(img[m], bins=bins, range=rng, density=True)
                    vec.extend(hist.tolist())
                mh = mag[y0:y1, x0:x1][m]
                ah = ang[y0:y1, x0:x1][m]
                oh, _ = np.histogram(ah, bins=8, range=(0, 1), density=True, weights=mh + 1e-6)
                vec.extend(oh.tolist())
                vec.append(float(np.mean(mh) / 255.0))
                vec.append(float(np.std(gray[y0:y1, x0:x1][m]) / 255.0))
                raw.append(np.asarray(vec, dtype=np.float64))
                coords.append((r, c))

        if len(raw) < 2:
            return out
        X = np.stack(raw, axis=0)
        n_comp = min(dim, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp, whiten=True, random_state=0)
        Z = pca.fit_transform(X)
        for (r, c), z in zip(coords, Z):
            v = np.zeros(dim, dtype=np.float32)
            v[: z.shape[0]] = z.astype(np.float32)
            nrm = np.linalg.norm(v) + 1e-8
            out[r, c] = v / nrm
        return out
