#!/usr/bin/env python3
"""Publish a bundled sample video into MediaMTX as a real-time RTMP feed, so
the full live-mode pipeline (MediaMTX -> WHEP browser + RTSP CV worker) can
be exercised with zero DJI hardware. Controlled via
POST /vision/live/simulate/start|stop (cropmerge/live/router.py).

The UI must always label this "SIMULATED LIVE INPUT" — never presented as
indistinguishable from a real drone feed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, help="Path to the sample video to loop-publish")
    parser.add_argument("--rtmp-url", required=True, help="MediaMTX RTMP ingest URL, e.g. rtmp://127.0.0.1:1935/cropmerge/live")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not installed or not on PATH", file=sys.stderr)
        return 1

    cmd = [
        "ffmpeg",
        "-re",
        "-stream_loop", "-1",
        "-i", args.sample,
        "-c", "copy",
        "-f", "flv",
        args.rtmp_url,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
