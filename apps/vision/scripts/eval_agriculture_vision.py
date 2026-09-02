#!/usr/bin/env python3
"""
Agriculture-Vision reference eval (planter_skip labels).

Does NOT claim transfer to DJI Mini 2 video — RGB-only sanity benchmark hook.
Requires dataset under data/agriculture-vision/ (not committed).

Layout:
  data/agriculture-vision/images/*.jpg
  data/agriculture-vision/labels/planter_skip/*.png  (binary masks)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cropmerge.eval.metrics import compare_label_maps
from cropmerge.pipeline.schemas import SemanticClass

NOTE = (
    "Agriculture-Vision planter_skip is an aerial still-image benchmark. "
    "Results do not automatically transfer to Midwest drone video geometry or crop stage."
)


def main() -> int:
    p = argparse.ArgumentParser(description="Agriculture-Vision planter_skip RGB eval hook")
    p.add_argument("--data-dir", type=Path, default=ROOT.parents[1] / "data" / "agriculture-vision")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    img_dir = args.data_dir / "images"
    skip_dir = args.data_dir / "labels" / "planter_skip"
    if not img_dir.is_dir():
        print(f"Dataset not found at {args.data_dir}")
        print("Download Agriculture-Vision and place RGB images + planter_skip masks locally.")
        print(json.dumps({"note": NOTE, "status": "missing_dataset"}, indent=2))
        return 0

    import cv2

    from cropmerge.config import load_config
    from cropmerge.segmentation import create_segmenter

    cfg = load_config()
    seg = create_segmenter("field_cv", cfg, allow_fallback=True)
    results = {"note": NOTE, "samples": [], "mean_iou_planter_skip_proxy": None}

    images = sorted(img_dir.glob("*.jpg"))[: args.limit]
    ious = []
    for img_path in images:
        mask_path = skip_dir / f"{img_path.stem}.png"
        if not mask_path.is_file():
            continue
        bgr = cv2.imread(str(img_path))
        gt_bin = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        gt = np_full_labels(bgr.shape[:2], gt_bin)
        pred_seg = seg.segment_image(bgr)
        pred = np.zeros(bgr.shape[:2], dtype=object)
        if pred_seg.label_map is not None:
            lm = pred_seg.label_map
            pred[(lm == SemanticClass.BARE_SOIL.value) | (lm == SemanticClass.UNKNOWN.value)] = (
                SemanticClass.BARE_SOIL.value
            )
            pred[lm == SemanticClass.CROP.value] = SemanticClass.CROP.value
        rep = compare_label_maps(pred, gt)
        iou = rep["per_class"].get("BARE_SOIL", {}).get("iou")
        if iou is not None:
            ious.append(iou)
        results["samples"].append({"image": img_path.name, "bare_soil_iou": iou})

    if ious:
        results["mean_iou_planter_skip_proxy"] = round(float(sum(ious) / len(ious)), 4)

    out = ROOT / "outputs" / "agriculture_vision_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")
    return 0


def np_full_labels(shape, skip_mask):
    import numpy as np

    gt = np.full(shape, SemanticClass.CROP.value, dtype=object)
    gt[skip_mask] = SemanticClass.BARE_SOIL.value
    return gt


if __name__ == "__main__":
    raise SystemExit(main())
