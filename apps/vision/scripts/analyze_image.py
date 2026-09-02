#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cropmerge.config import load_config
from cropmerge.logging_utils import setup_logging
from cropmerge.segmentation import create_segmenter
from cropmerge.segmentation.postprocess import class_fractions
from cropmerge.visualization.overlay import draw_field_boundary, draw_segmentation
from cropmerge.visualization.video_writer import write_image


def main() -> int:
    p = argparse.ArgumentParser(description="Segment a single farm image")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default="outputs/image_seg")
    p.add_argument("--segmentation-backend", default="heuristic")
    args = p.parse_args()
    setup_logging()
    bgr = cv2.imread(args.input)
    if bgr is None:
        raise SystemExit(f"Cannot read {args.input}")
    cfg = load_config()
    seg = create_segmenter(args.segmentation_backend, cfg)
    result = seg.segment_image(bgr)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    vis = draw_segmentation(bgr, result.label_map)
    vis = draw_field_boundary(vis, result.field_mask)
    write_image(out / "segmentation.jpg", vis)
    fracs = class_fractions(result.label_map)
    summary = {
        "backend": result.backend,
        "isFallback": result.is_fallback,
        "classCoverage": {k: v for k, v in fracs.items() if not k.startswith("_")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
