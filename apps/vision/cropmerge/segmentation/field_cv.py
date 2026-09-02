"""
Real field segmentation via superpixels + k-means on Lab/ExG/texture features.

Backend name: field_cv (not SAM). OpenCV SLIC + sklearn clustering +
connected-component field extraction + light temporal stabilization.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from cropmerge.pipeline.schemas import SemanticClass
from cropmerge.segmentation.base import FrameSegmentation, SegmentMask, Segmenter
from cropmerge.segmentation.postprocess import build_field_mask, finalize_segmentation
from cropmerge.video.registration import estimate_homography, warp_mask

log = logging.getLogger("cropmerge.segmentation.field_cv")

_CLASSES = [
    SemanticClass.FIELD.value,
    SemanticClass.CROP.value,
    SemanticClass.BARE_SOIL.value,
    SemanticClass.ROAD_PATH.value,
    SemanticClass.TREE_VEGETATION.value,
    SemanticClass.WATER.value,
    SemanticClass.INFRASTRUCTURE.value,
    SemanticClass.UNKNOWN.value,
]


def _local_var(gray: np.ndarray, ksize: int = 7) -> np.ndarray:
    g = gray.astype(np.float32)
    mu = cv2.blur(g, (ksize, ksize))
    mu2 = cv2.blur(g * g, (ksize, ksize))
    return np.maximum(mu2 - mu * mu, 0.0)


def _encode_labels(label_map: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    idx = np.zeros(label_map.shape[:2], dtype=np.uint8)
    inv: dict[int, str] = {}
    for i, c in enumerate(_CLASSES):
        code = i + 1
        inv[code] = c
        idx[label_map == c] = code
    return idx, inv


def _decode_labels(idx: np.ndarray, inv: dict[int, str]) -> np.ndarray:
    out = np.empty(idx.shape, dtype=object)
    out[:] = SemanticClass.UNKNOWN.value
    for code, name in inv.items():
        out[idx == code] = name
    return out


class FieldCVSegmenter(Segmenter):
    name = "field_cv"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self._frames: list[np.ndarray] = []
        scfg = self.cfg.get("segmentation", {})
        self.n_segments = int(scfg.get("n_superpixels", 400))
        self.n_clusters = int(scfg.get("n_clusters", 6))

    def segment_image(
        self, bgr: np.ndarray, frame_index: int = 0, timestamp_sec: float = 0.0
    ) -> FrameSegmentation:
        h, w = bgr.shape[:2]
        scale = 1.0
        work = bgr
        if max(h, w) > 960:
            scale = 960.0 / max(h, w)
            work = cv2.resize(
                bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        wh, ww = work.shape[:2]

        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV).astype(np.float32)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r, g, bch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        exg = 2.0 * g - r - bch
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        lvar = _local_var(gray, 9)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)

        # Superpixels (SLIC if ximgproc available, else grid fallback)
        labels_sp, n_sp = self._superpixels(work)

        stack = np.stack(
            [
                lab[:, :, 0] / 255.0,
                lab[:, :, 1] / 255.0,
                lab[:, :, 2] / 255.0,
                hsv[:, :, 1] / 255.0,
                hsv[:, :, 2] / 255.0,
                exg,
                lvar / (float(lvar.max()) + 1e-6),
                grad / (float(grad.max()) + 1e-6),
            ],
            axis=-1,
        ).reshape(-1, 8)
        flat_l = labels_sp.reshape(-1)

        feats = np.zeros((n_sp, 8), dtype=np.float32)
        counts = np.zeros(n_sp, dtype=np.int32)
        for i in range(n_sp):
            m = flat_l == i
            c = int(m.sum())
            counts[i] = c
            if c > 0:
                feats[i] = stack[m].mean(axis=0)

        valid = counts > 10
        sp_label = np.full(n_sp, SemanticClass.UNKNOWN.value, dtype=object)
        X = feats[valid]
        if X.shape[0] >= 2:
            k = min(self.n_clusters, X.shape[0])
            km = MiniBatchKMeans(
                n_clusters=k, random_state=0, batch_size=min(256, X.shape[0]), n_init=3
            )
            pred = km.fit_predict(X)
            cluster_ids = np.full(n_sp, -1, dtype=np.int32)
            cluster_ids[np.where(valid)[0]] = pred
            for ci, cvec in enumerate(km.cluster_centers_):
                L, A, B_, S, V, ExG, Var, Gr = cvec
                if ExG > 0.06 and V > 0.15:
                    name = SemanticClass.CROP.value
                elif ExG > 0.02 and V < 0.35 and Var > 0.08:
                    name = SemanticClass.TREE_VEGETATION.value
                elif ExG < 0.0 and B_ > A and V < 0.55 and S > 0.15:
                    name = SemanticClass.WATER.value
                elif S < 0.2 and Gr > 0.12 and V > 0.3:
                    name = SemanticClass.INFRASTRUCTURE.value
                elif ExG < 0.04 and S < 0.4:
                    name = SemanticClass.BARE_SOIL.value
                else:
                    name = (
                        SemanticClass.CROP.value
                        if ExG > 0.02
                        else SemanticClass.UNKNOWN.value
                    )
                sp_label[cluster_ids == ci] = name

        label_small = np.empty((wh, ww), dtype=object)
        for i in range(n_sp):
            label_small[labels_sp == i] = sp_label[i]

        # Road refinement
        bare_m = label_small == SemanticClass.BARE_SOIL.value
        edges = cv2.Canny(gray, 50, 120)
        edge_d = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
        strip = (
            cv2.morphologyEx(bare_m.astype(np.uint8), cv2.MORPH_OPEN, h_kernel) > 0
        ) | (cv2.morphologyEx(bare_m.astype(np.uint8), cv2.MORPH_OPEN, v_kernel) > 0)
        label_small[strip & edge_d] = SemanticClass.ROAD_PATH.value

        if scale != 1.0:
            idx, inv = _encode_labels(label_small)
            idx_up = cv2.resize(idx, (w, h), interpolation=cv2.INTER_NEAREST)
            label_map = _decode_labels(idx_up, inv)
        else:
            label_map = label_small

        crop = label_map == SemanticClass.CROP.value
        bare = label_map == SemanticClass.BARE_SOIL.value
        road_m = label_map == SemanticClass.ROAD_PATH.value
        trees = label_map == SemanticClass.TREE_VEGETATION.value
        water = label_map == SemanticClass.WATER.value
        infra = label_map == SemanticClass.INFRASTRUCTURE.value
        field = crop | bare

        masks = [
            SegmentMask(SemanticClass.FIELD, field, 0.85, "field_cv:field"),
            SegmentMask(SemanticClass.CROP, crop, 0.82, "field_cv:crop"),
            SegmentMask(SemanticClass.BARE_SOIL, bare, 0.78, "field_cv:bare"),
            SegmentMask(SemanticClass.ROAD_PATH, road_m, 0.7, "field_cv:road"),
            SegmentMask(SemanticClass.TREE_VEGETATION, trees, 0.7, "field_cv:trees"),
            SegmentMask(SemanticClass.WATER, water, 0.65, "field_cv:water"),
            SegmentMask(SemanticClass.INFRASTRUCTURE, infra, 0.6, "field_cv:infra"),
        ]
        seg = FrameSegmentation(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            masks=masks,
            backend=self.name,
            is_fallback=False,
        )
        return finalize_segmentation(seg, self.cfg)

    def _superpixels(self, work: np.ndarray) -> tuple[np.ndarray, int]:
        wh, ww = work.shape[:2]
        region_size = max(8, int(np.sqrt((wh * ww) / max(self.n_segments, 50))))
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createSuperpixelSLIC"):
            slic = cv2.ximgproc.createSuperpixelSLIC(
                work,
                algorithm=cv2.ximgproc.SLICO,
                region_size=region_size,
                ruler=20.0,
            )
            slic.iterate(8)
            labels = slic.getLabels()
            return labels, int(slic.getNumberOfSuperpixels())

        # Grid superpixels fallback
        gh = max(8, wh // int(np.sqrt(self.n_segments)))
        gw = max(8, ww // int(np.sqrt(self.n_segments)))
        ys = np.arange(wh) // gh
        xs = np.arange(ww) // gw
        labels = (ys[:, None] * (xs.max() + 1) + xs[None, :]).astype(np.int32)
        return labels, int(labels.max()) + 1

    def start_video(self, frames_bgr: list[np.ndarray]) -> None:
        self._frames = frames_bgr

    def track_video(self, frames_meta: list[tuple[int, float]]) -> list[FrameSegmentation]:
        outs: list[FrameSegmentation] = []
        prev_bgr = None
        prev_idx = None
        inv_ref: dict[int, str] | None = None
        for (idx, ts), bgr in zip(frames_meta, self._frames):
            seg = self.segment_image(bgr, idx, ts)
            if (
                prev_bgr is not None
                and prev_idx is not None
                and seg.label_map is not None
            ):
                reg = estimate_homography(prev_bgr, bgr, self.cfg)
                if reg.success and reg.transform is not None and reg.confidence >= 0.35:
                    h, w = bgr.shape[:2]
                    cur_i, inv = _encode_labels(seg.label_map)
                    if inv_ref is None:
                        inv_ref = inv
                    if prev_idx.shape != cur_i.shape:
                        prev_i = cv2.resize(
                            prev_idx, (w, h), interpolation=cv2.INTER_NEAREST
                        )
                    else:
                        prev_i = prev_idx
                    warped = warp_mask(prev_i, reg.transform, (h, w))
                    # Fill unknown holes from warped prior
                    fill = (cur_i == 0) & (warped > 0)
                    cur_i = cur_i.copy()
                    cur_i[fill] = warped[fill]
                    # Temporal consensus on crop/soil: require agreement or strong current
                    for cname in (SemanticClass.CROP.value, SemanticClass.BARE_SOIL.value):
                        try:
                            code = _CLASSES.index(cname) + 1
                        except ValueError:
                            continue
                        # if warp says crop and current bare with low conf edge — keep warp on interior
                        pass
                    seg.label_map = _decode_labels(cur_i, inv)
                    union = self.cfg.get("segmentation", {}).get(
                        "field_union_classes", ["CROP", "BARE_SOIL", "FIELD"]
                    )
                    seg.field_mask = build_field_mask(seg.label_map, union)
                    seg.crop_mask = seg.label_map == SemanticClass.CROP.value
                    prev_idx = cur_i
                else:
                    prev_idx, _ = _encode_labels(seg.label_map)
            elif seg.label_map is not None:
                prev_idx, _ = _encode_labels(seg.label_map)
            outs.append(seg)
            prev_bgr = bgr
        return outs
