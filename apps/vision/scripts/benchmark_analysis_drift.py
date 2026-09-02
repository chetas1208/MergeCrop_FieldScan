#!/usr/bin/env python3
"""
Run the SAME FieldScan analysis (same 1 FPS schedule, same config, same
models) on an original source and one or more compressed candidate
derivatives, and report analysis drift -- the acceptance gate the storage
campaign requires to exist BEFORE any lossy source-replacement compression
is ever turned on.

This does not itself enable or disable anything. It only measures.
CROPMERGE_LOSSY_SOURCE_REPLACEMENT stays off regardless of the result until
a human reviews real benchmark output.

Usage:
  python scripts/benchmark_analysis_drift.py \
      --original samples/real_field_drone.mp4 \
      --candidate hevc_crf20=candidates/hevc_crf20.mp4 \
      --candidate hevc_crf24=candidates/hevc_crf24.mp4 \
      --sample-fps 1 \
      --out docs/COMPRESSION_BENCHMARK.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cropmerge.config import load_config
from cropmerge.pipeline.processor import FieldTriageProcessor
from cropmerge.storage.drift_metrics import DriftGate, compare_reports, passes_gate

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(path: Path, out_dir: Path, sample_fps: float, max_frames: int | None):
    cfg = load_config()
    return FieldTriageProcessor(cfg).process(
        path, out_dir, sample_fps=sample_fps, max_frames=max_frames, segmentation_backend="heuristic", skip_dino=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", required=True, help="Path to the original/reference source")
    parser.add_argument(
        "--candidate", action="append", default=[], metavar="NAME=PATH",
        help="A compressed candidate to compare, as name=path (repeatable)",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--work-dir", default=str(REPO_ROOT / "benchmarks" / "drift-work"))
    parser.add_argument("--out", default=None, help="Write machine-readable JSON here")
    parser.add_argument(
        "--heuristic-only", action="store_true", default=True,
        help="Use the heuristic (not SAM2/DINO) backend for fast, deterministic, CPU-only comparison (default: on)",
    )
    args = parser.parse_args()

    if not args.candidate:
        parser.error("at least one --candidate NAME=PATH is required")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running reference analysis on {args.original} ...")
    original_report = _run(Path(args.original), work_dir / "original", args.sample_fps, args.max_frames)

    results: dict[str, dict] = {}
    gate = DriftGate()
    for spec in args.candidate:
        if "=" not in spec:
            parser.error(f"--candidate must be NAME=PATH, got: {spec!r}")
        name, path_str = spec.split("=", 1)
        print(f"Running candidate '{name}' analysis on {path_str} ...")
        candidate_report = _run(Path(path_str), work_dir / name, args.sample_fps, args.max_frames)

        drift = compare_reports(original_report, candidate_report)
        ok, reasons = passes_gate(drift, gate)
        results[name] = {
            **drift.to_dict(),
            "gatePasses": ok,
            "gateFailureReasons": reasons,
            "originalBytes": Path(args.original).stat().st_size,
            "candidateBytes": Path(path_str).stat().st_size,
        }
        results[name]["compressionRatio"] = (
            1.0 - (results[name]["candidateBytes"] / results[name]["originalBytes"])
            if results[name]["originalBytes"]
            else 0.0
        )

    print("\n| Candidate | Crop MAE | Soil MAE | Type agree | Priority agree | HP recall | Mean IoU | Gate |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, r in results.items():
        gate_str = "PASS" if r["gatePasses"] else "FAIL: " + "; ".join(r["gateFailureReasons"])
        print(
            f"| {name} | {r['cropCoverageMae']:.4f} | {r['soilFractionMae']:.4f} | "
            f"{r['typeAgreement']:.1%} | {r['priorityAgreement']:.1%} | {r['highPriorityRecall']:.1%} | "
            f"{r['meanMatchedIou']:.3f} | {gate_str} |"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
