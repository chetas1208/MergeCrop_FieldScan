#!/usr/bin/env python3
"""
Evaluate segmentation backends against pixel-level ground truth.

Built-in synthetic benchmark (procedural Midwest-like field):
  python scripts/eval_segmentation.py

Custom labeled pairs (image + indexed mask PNG):
  python scripts/eval_segmentation.py --pairs-dir data/eval/

Mask PNG index map (default):
  0=UNKNOWN, 1=CROP, 2=BARE_SOIL, 3=ROAD_PATH, 4=TREE_VEGETATION,
  5=WATER, 6=INFRASTRUCTURE
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cropmerge.config import load_config
from cropmerge.eval.metrics import compare_label_maps
from cropmerge.eval.synthetic_field import make_synthetic_pair
from cropmerge.pipeline.schemas import SemanticClass
from cropmerge.segmentation import create_segmenter

INDEX_TO_CLASS = {
    0: SemanticClass.UNKNOWN.value,
    1: SemanticClass.CROP.value,
    2: SemanticClass.BARE_SOIL.value,
    3: SemanticClass.ROAD_PATH.value,
    4: SemanticClass.TREE_VEGETATION.value,
    5: SemanticClass.WATER.value,
    6: SemanticClass.INFRASTRUCTURE.value,
}


def _load_weights_env() -> None:
    env_path = ROOT / "models" / "weights.env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


def _mask_png_to_labels(path: Path) -> np.ndarray:
    idx = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if idx is None:
        raise RuntimeError(f"Could not read mask: {path}")
    out = np.full(idx.shape, SemanticClass.UNKNOWN.value, dtype=object)
    for code, name in INDEX_TO_CLASS.items():
        out[idx == code] = name
    return out


def _iter_pairs(pairs_dir: Path) -> list[tuple[Path, Path]]:
    images = sorted(pairs_dir.glob("images/*"))
    pairs: list[tuple[Path, Path]] = []
    for img in images:
        mask = pairs_dir / "masks" / f"{img.stem}.png"
        if not mask.is_file():
            raise FileNotFoundError(f"Missing mask for {img.name}: {mask}")
        pairs.append((img, mask))
    if not pairs:
        raise FileNotFoundError(f"No images found under {pairs_dir / 'images'}")
    return pairs


def _run_backend(
    backend: str,
    cfg: dict,
    samples: list[tuple[np.ndarray, np.ndarray, str]],
) -> dict:
    segmenter = create_segmenter(backend, cfg, allow_fallback=True)
    cms = []
    frame_reports = []
    actual_backend = backend
    used_fallback = False

    for bgr, gt, sample_id in samples:
        seg = segmenter.segment_image(bgr, 0, 0.0)
        actual_backend = seg.backend
        used_fallback = bool(seg.is_fallback)
        if seg.label_map is None:
            raise RuntimeError(f"No label_map for sample {sample_id}")
        report = compare_label_maps(seg.label_map, gt)
        cm = np.array(report["confusion_matrix"]["matrix"], dtype=np.int64)
        cms.append(cm)
        frame_reports.append({"sample_id": sample_id, **report})

    total_cm = np.sum(cms, axis=0)
    summary = compare_label_maps(
        np.full((1, 1), SemanticClass.UNKNOWN.value, dtype=object),
        np.full((1, 1), SemanticClass.UNKNOWN.value, dtype=object),
    )
    # Recompute aggregate from summed confusion matrix
    from cropmerge.eval.metrics import summarize

    summary = summarize(total_cm)

    return {
        "requested_backend": backend,
        "actual_backend": actual_backend,
        "used_fallback": used_fallback,
        "frames_evaluated": len(samples),
        **summary,
        "frames": frame_reports,
    }


def main() -> int:
    _load_weights_env()

    p = argparse.ArgumentParser(description="Segmentation mIoU / precision / recall evaluation")
    p.add_argument(
        "--segmentation-backend",
        default=os.environ.get("CROP_MERGE_SEGMENTATION_BACKEND", "field_cv"),
        help="field_cv | sam2 | heuristic | sam3",
    )
    p.add_argument("--pairs-dir", type=Path, default=None, help="Dir with images/ + masks/")
    p.add_argument("--synthetic-frames", type=int, default=12, help="Synthetic benchmark frame count")
    p.add_argument("--output", type=Path, default=None, help="Write JSON report here")
    p.add_argument("--compare-all", action="store_true", help="Run field_cv, heuristic, and sam2")
    args = p.parse_args()

    cfg = load_config()
    samples: list[tuple[np.ndarray, np.ndarray, str]] = []

    if args.pairs_dir:
        for img_path, mask_path in _iter_pairs(args.pairs_dir):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise RuntimeError(f"Could not read image: {img_path}")
            gt = _mask_png_to_labels(mask_path)
            samples.append((bgr, gt, img_path.name))
    else:
        for i in range(args.synthetic_frames):
            t = i / max(args.synthetic_frames - 1, 1) * 5.0
            bgr, gt = make_synthetic_pair(t)
            samples.append((bgr, gt, f"synthetic_t{t:.2f}"))

    backends = ["field_cv", "heuristic", "sam2"] if args.compare_all else [args.segmentation_backend]
    results = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.pairs_dir) if args.pairs_dir else "synthetic_field",
        "note": (
            "Metrics are segmentation-class mIoU vs provided ground truth. "
            "Synthetic GT is procedural — not Agriculture-Vision. "
            "SAM2 requires `pip install git+https://github.com/facebookresearch/sam2.git` "
            "or it falls back to heuristic."
        ),
        "weights": _weights_info(),
        "backends": {},
    }

    for backend in backends:
        print(f"\n=== Evaluating backend: {backend} ===")
        report = _run_backend(backend, cfg, samples)
        results["backends"][backend] = report
        print(f"  actual backend : {report['actual_backend']}")
        print(f"  used fallback  : {report['used_fallback']}")
        print(f"  pixel accuracy : {report['pixel_accuracy']:.4f}")
        print(f"  mean IoU       : {report['mean_iou']:.4f}")
        print(f"  macro F1       : {report['macro_f1']:.4f}")
        for cls, scores in report["per_class"].items():
            iou = scores.get("iou")
            if iou is None:
                continue
            p = scores.get("precision")
            r = scores.get("recall")
            p_s = f"{p:.3f}" if p is not None else "n/a"
            r_s = f"{r:.3f}" if r is not None else "n/a"
            print(f"    {cls:18s} IoU={iou:.3f}  P={p_s}  R={r_s}")

    out = args.output or (ROOT / "outputs" / "eval_segmentation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote report → {out}")
    return 0


def _weights_info() -> dict:
    models = ROOT / "models"
    entries = {
        "sam2": {
            "file": "sam2.1_hiera_small.pt",
            "env": "SAM2_CHECKPOINT",
            "architecture": "SAM 2.1 Hiera-S",
            "pretraining": "Meta SA-1B / SA-V (segment-anything pretrain)",
            "role": "Instance masks → RGB-assigned semantic classes",
        },
        "dinov2": {
            "file": "dinov2_vitb14_pretrain.pth",
            "env": "DINOV2_CHECKPOINT",
            "architecture": "DINOv2 ViT-B/14",
            "pretraining": "Meta LVD-142M self-supervised",
            "role": "Tile embeddings for visual anomaly (not segmentation eval)",
        },
        "sam3": {
            "file": "sam3.1_multiplex.pt",
            "env": "SAM3_CHECKPOINT",
            "architecture": "SAM 3.1 multiplex",
            "pretraining": "Meta (gated HF weights)",
            "role": "Optional future segmentation backend",
        },
        "dinov3": {
            "file": "dinov3_vitb16_pretrain.safetensors",
            "env": "DINOV3_CHECKPOINT",
            "architecture": "DINOv3 ViT-B/16",
            "pretraining": "Meta LVD-1689M self-supervised (gated)",
            "role": "Optional future embedding backend",
        },
        "field_cv": {
            "file": None,
            "architecture": "OpenCV SLIC + sklearn MiniBatchKMeans + RGB rules",
            "pretraining": "None — classical CV, not ML-trained",
            "role": "Default offline segmentation",
        },
        "heuristic": {
            "file": None,
            "architecture": "ExG / HSV threshold rules",
            "pretraining": "None",
            "role": "Explicit fallback when SAM weights/package missing",
        },
    }
    info = {}
    for name, meta in entries.items():
        path = None
        env_key = meta.get("env")
        if env_key and os.environ.get(env_key):
            path = os.environ[env_key]
        elif meta.get("file"):
            candidate = models / meta["file"]
            if candidate.is_file():
                path = str(candidate)
        info[name] = {**meta, "resolved_path": path, "present": bool(path) or meta.get("file") is None}
    info["configured_segmentation_backend"] = os.environ.get("CROP_MERGE_SEGMENTATION_BACKEND", "field_cv")
    info["configured_dino_backend"] = os.environ.get("CROP_MERGE_DINO_BACKEND", "auto")
    return info


if __name__ == "__main__":
    raise SystemExit(main())
