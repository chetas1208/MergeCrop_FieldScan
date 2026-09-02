#!/usr/bin/env python3
"""
Download CV weights for CropMerge vision engine.

Public (no auth):
  - SAM 2.1 Hiera-S  (Meta FB CDN)
  - DINOv2 ViT-B/14  (Meta FB CDN)

Gated (needs HF token + license accept):
  - SAM 3.1 multiplex
  - DINOv3 ViT-B/16

Usage:
  python scripts/download_weights.py
  HF_TOKEN=hf_... python scripts/download_weights.py --gated
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

PUBLIC = {
    "sam2.1_hiera_small.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "sha256_prefix": None,  # large; size check only
        "min_bytes": 150_000_000,
        "role": "segmentation",
        "backend": "sam2",
        "note": "Meta SAM 2.1 Hiera-S — public. Not SAM 3.1.",
    },
    "dinov2_vitb14_pretrain.pth": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "sha256_prefix": None,
        "min_bytes": 300_000_000,
        "role": "embeddings",
        "backend": "dinov2",
        "note": "Meta DINOv2 ViT-B/14 — public. Not DINOv3.",
    },
}

GATED = {
    "sam3.1_multiplex.pt": {
        "repo_id": "facebook/sam3.1",
        "filename": "sam3.1_multiplex.pt",
        "min_bytes": 2_000_000_000,
        "role": "segmentation",
        "backend": "sam3",
        "note": "Gated SAM 3.1 — accept license on HF + HF_TOKEN",
    },
    "dinov3_vitb16.safetensors": {
        "repo_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "filename": "model.safetensors",
        "min_bytes": 200_000_000,
        "role": "embeddings",
        "backend": "dinov3",
        "note": "Gated DINOv3 ViT-B/16 — accept license on HF + HF_TOKEN",
        "save_as": "dinov3_vitb16_pretrain.safetensors",
    },
}


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"↓ {url}\n  → {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "cropmerge-field-triage/0.1"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print(f"\r  {done/1e6:.0f}/{total/1e6:.0f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r  {done/1e6:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def _ok(path: Path, min_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def download_public(force: bool = False) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name, meta in PUBLIC.items():
        dest = MODELS / name
        if _ok(dest, meta["min_bytes"]) and not force:
            print(f"✓ exists {dest} ({dest.stat().st_size/1e6:.0f} MB)")
            out[name] = dest
            continue
        _download_url(meta["url"], dest)
        if not _ok(dest, meta["min_bytes"]):
            raise RuntimeError(f"Download incomplete: {dest}")
        print(f"✓ {name} ({dest.stat().st_size/1e6:.0f} MB) — {meta['note']}")
        out[name] = dest
    return out


def download_gated(force: bool = False) -> dict[str, Path]:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        print("✗ No HF_TOKEN — skip gated SAM3.1 / DINOv3")
        print("  1. Accept licenses:")
        print("     https://huggingface.co/facebook/sam3.1")
        print("     https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m")
        print("  2. export HF_TOKEN=hf_...")
        print("  3. python scripts/download_weights.py --gated")
        return {}

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub") from e

    out: dict[str, Path] = {}
    for name, meta in GATED.items():
        save_name = meta.get("save_as", name)
        dest = MODELS / save_name
        if _ok(dest, meta["min_bytes"]) and not force:
            print(f"✓ exists {dest}")
            out[save_name] = dest
            continue
        print(f"↓ HF {meta['repo_id']} / {meta['filename']}")
        try:
            path = hf_hub_download(
                repo_id=meta["repo_id"],
                filename=meta["filename"],
                token=token,
                local_dir=str(MODELS),
                local_dir_use_symlinks=False,
            )
            src = Path(path)
            if src.name != save_name:
                target = MODELS / save_name
                if src.resolve() != target.resolve():
                    target.write_bytes(src.read_bytes()) if not target.exists() else None
                    if src.exists() and src.name == meta["filename"] and save_name != meta["filename"]:
                        # keep both; prefer save_as path
                        if not target.exists():
                            src.rename(target)
                        dest = target
                    else:
                        dest = src if src.exists() else target
            if not _ok(Path(dest), meta["min_bytes"]):
                # hf may nest under models/repo
                candidates = list(MODELS.rglob(meta["filename"]))
                if candidates:
                    dest = candidates[0]
            print(f"✓ {dest} — {meta['note']}")
            out[save_name] = Path(dest)
        except Exception as e:
            print(f"✗ {name}: {e}")
            print(f"  Accept model license on HF, ensure token has access.")
    return out


def write_env_snippet(paths: dict[str, Path]) -> Path:
    env = MODELS / "weights.env"
    lines = [
        "# Auto-generated by scripts/download_weights.py",
        f"CROP_MERGE_MODELS_DIR={MODELS}",
    ]
    sam2 = paths.get("sam2.1_hiera_small.pt") or (MODELS / "sam2.1_hiera_small.pt")
    dino2 = paths.get("dinov2_vitb14_pretrain.pth") or (MODELS / "dinov2_vitb14_pretrain.pth")
    sam3 = MODELS / "sam3.1_multiplex.pt"
    dino3 = MODELS / "dinov3_vitb16_pretrain.safetensors"
    if sam3.is_file():
        lines.append("CROP_MERGE_SEGMENTATION_BACKEND=sam3")
        lines.append(f"SAM3_CHECKPOINT={sam3}")
    elif sam2.is_file() if isinstance(sam2, Path) else False:
        lines.append("CROP_MERGE_SEGMENTATION_BACKEND=sam2")
        lines.append(f"SAM2_CHECKPOINT={sam2}")
    if dino3.is_file():
        lines.append("CROP_MERGE_DINO_BACKEND=dinov3")
        lines.append(f"DINOV3_CHECKPOINT={dino3}")
    elif isinstance(dino2, Path) and dino2.is_file():
        lines.append("CROP_MERGE_DINO_BACKEND=dinov2")
        lines.append(f"DINOV2_CHECKPOINT={dino2}")
    env.write_text("\n".join(lines) + "\n")
    print(f"wrote {env}")
    return env


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gated", action="store_true", help="Also try SAM3.1 + DINOv3 via HF")
    p.add_argument("--force", action="store_true")
    p.add_argument("--gated-only", action="store_true")
    args = p.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if not args.gated_only:
        paths.update(download_public(force=args.force))
    if args.gated or args.gated_only:
        paths.update(download_gated(force=args.force))
    write_env_snippet(paths)
    print("\nDone. Source env before serving:")
    print(f"  set -a; source {MODELS}/weights.env; set +a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
