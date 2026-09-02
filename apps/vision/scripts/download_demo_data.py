#!/usr/bin/env python3
"""Document public data sources; generate local synthetic demo (no large downloads by default)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


SOURCES = [
    {
        "name": "synthetic_field",
        "source": "generated locally by scripts/generate_demo_video.py",
        "license": "generated — no third-party copyright",
        "imagery_type": "synthetic RGB aerial-like",
        "resolution": "640x360",
        "crop_location_context": "stylized Midwest corn/soy canopy + soil patch",
        "why": "Offline reproducible demo without committing copyrighted drone video",
        "domain_gap": "Not real DJI Mini 2 footage; textures and lighting simplified",
    },
    {
        "name": "Agriculture-Vision",
        "source": "https://github.com/SHI-Labs/Agriculture-Vision",
        "license": "see upstream challenge terms",
        "imagery_type": "aerial farmland RGB (+ NIR available — do not use NIR for RGB-only claims)",
        "resolution": "varies",
        "crop_location_context": "U.S. agricultural fields",
        "why": "Benchmark for sensible RGB field behavior",
        "domain_gap": "Still imagery vs Mini-2 video; altitude/GSD differ",
    },
    {
        "name": "USDA Cropland Data Layer",
        "source": "https://www.nass.usda.gov/Research_and_Science/Cropland/Viewer/index.php",
        "license": "US government public data — verify current terms",
        "imagery_type": "georeferenced crop land-cover",
        "resolution": "30m class",
        "crop_location_context": "Illinois corn/soy relevant",
        "why": "Future geospatial hook — not used in V1 CV core",
        "domain_gap": "Not RGB drone video",
    },
]


def main() -> int:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    (SAMPLES / "DATA_SOURCES.json").write_text(json.dumps(SOURCES, indent=2))
    # Generate synthetic video
    from generate_demo_video import main as gen

    gen()
    print("Demo data ready. For real farm video, place files under samples/ (do not commit copyrighted media).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
