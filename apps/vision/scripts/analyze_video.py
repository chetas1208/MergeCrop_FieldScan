#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cropmerge.config import load_config
from cropmerge.logging_utils import setup_logging
from cropmerge.pipeline.processor import FieldTriageProcessor


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CropMerge Field Triage — analyze RGB drone video")
    p.add_argument("--input", "-i", required=True, help="Input .mp4/.mov/.m4v")
    p.add_argument("--output", "-o", default="outputs", help="Output directory")
    p.add_argument("--config", "-c", default=None, help="YAML config path")
    p.add_argument("--sample-fps", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default=None, help="auto|cpu|cuda (logged only unless torch)")
    p.add_argument("--skip-dino", action="store_true")
    p.add_argument(
        "--segmentation-backend",
        default=None,
        choices=["field_cv", "sam2", "sam3", "heuristic", "mock", "auto"],
    )
    p.add_argument(
        "--dino-backend",
        default=None,
        choices=["auto", "dinov2", "dinov3", "dense_rgb", "heuristic", "mock"],
    )
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    setup_logging("DEBUG" if args.debug else "INFO")
    cfg = load_config(args.config)
    if args.device:
        cfg.setdefault("runtime", {})["device"] = args.device

    proc = FieldTriageProcessor(cfg)
    report = proc.process(
        args.input,
        args.output,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        skip_dino=args.skip_dino or None,
        segmentation_backend=args.segmentation_backend,
        dino_backend=args.dino_backend,
    )
    print(json.dumps(report.to_camel_dict(), indent=2)[:2000])
    print(f"\nresults → {report.artifacts.results_json}")
    if report.artifacts.annotated_video:
        print(f"video   → {report.artifacts.annotated_video}")
    if report.artifacts.heatmap_png:
        print(f"heatmap → {report.artifacts.heatmap_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
