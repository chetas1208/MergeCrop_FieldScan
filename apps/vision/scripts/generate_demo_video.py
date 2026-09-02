#!/usr/bin/env python3
"""Synthetic Midwest-like field flight for offline demo (no copyrighted footage)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from cropmerge.eval.synthetic_field import make_synthetic_frame as make_frame


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=None)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--fps", type=float, default=10.0)
    args = p.parse_args()
    out = Path(args.output or Path(__file__).resolve().parents[1] / "samples" / "synthetic_field.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 360
    n = int(args.seconds * args.fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit("VideoWriter failed")
    for i in range(n):
        t = i / args.fps
        writer.write(make_frame(t, w, h))
    writer.release()
    print(f"Wrote {out} ({n} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
