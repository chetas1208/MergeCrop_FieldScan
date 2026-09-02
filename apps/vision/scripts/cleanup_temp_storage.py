#!/usr/bin/env python3
"""
Report (and, only with --apply, delete) orphaned temporary upload/archival
state under the vision data directory.

Phase 11 of the storage integration campaign. Default is DRY RUN — this
script only prints what it *would* remove until --apply is passed. Never
infers orphan status from file age alone: chunked-upload staging
directories are cross-checked against their own upload.json record before
being flagged, and a directory whose record shows status="completed" is
reported separately as an anomaly (it should already have been removed by
UploadStore.complete()) rather than silently deleted.

Usage:
  python scripts/cleanup_temp_storage.py                  # dry run, defaults
  python scripts/cleanup_temp_storage.py --ttl-hours 6     # shorter TTL
  python scripts/cleanup_temp_storage.py --apply           # actually delete
  python scripts/cleanup_temp_storage.py --limit 20        # cap report size
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OrphanCandidate:
    path: Path
    age_hours: float
    size_bytes: int
    reason: str
    anomaly: bool = False  # True = flagged for human review, never auto-deleted


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _latest_activity_mtime(path: Path) -> float:
    """Most recent mtime among all files under `path` (or of `path` itself
    if it's a file). More reliable than a directory's own mtime, which on
    POSIX only updates when a *direct* child is added/removed/renamed, not
    on writes to files nested deeper or on explicit mtime changes to an
    existing child -- exactly the case for a staging directory whose chunk
    files get written/rewritten in place."""
    if path.is_file():
        return path.stat().st_mtime
    latest: float | None = None
    for p in path.rglob("*"):
        if p.is_file():
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            latest = mtime if latest is None else max(latest, mtime)
    return latest if latest is not None else path.stat().st_mtime


def _scan_incomplete_uploads(uploads_dir: Path, ttl_hours: float, now: float) -> list[OrphanCandidate]:
    """Cross-checks each staging directory's own upload.json record rather
    than trusting mtime alone."""
    incomplete_root = uploads_dir / ".incomplete"
    if not incomplete_root.is_dir():
        return []

    candidates: list[OrphanCandidate] = []
    for entry in sorted(incomplete_root.iterdir()):
        if not entry.is_dir():
            continue
        record_path = entry / "upload.json"
        age_hours = (now - _latest_activity_mtime(entry)) / 3600.0
        size_bytes = _dir_size_bytes(entry)

        if not record_path.is_file():
            if age_hours >= ttl_hours:
                candidates.append(
                    OrphanCandidate(entry, age_hours, size_bytes, "no upload.json record (crashed/partial upload)")
                )
            continue

        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            status = record.get("status", "unknown")
        except (OSError, json.JSONDecodeError):
            if age_hours >= ttl_hours:
                candidates.append(OrphanCandidate(entry, age_hours, size_bytes, "unreadable upload.json record"))
            continue

        if status == "completed":
            # UploadStore.complete() already shutil.rmtree()s this directory
            # on success -- if it's still here, something is wrong (crash
            # between assembly and cleanup). Flag for review, never
            # auto-delete: a completed record pointing at data we can't
            # otherwise verify is exactly the "don't infer orphanhood from
            # age alone" case.
            candidates.append(OrphanCandidate(entry, age_hours, size_bytes, "status=completed but not cleaned up", anomaly=True))
        elif status == "uploading" and age_hours >= ttl_hours:
            candidates.append(OrphanCandidate(entry, age_hours, size_bytes, f"abandoned upload (status={status})"))

    return candidates


def _scan_jxl_archive_tmp(data_dir: Path, ttl_hours: float, now: float) -> list[OrphanCandidate]:
    """cropmerge/storage/jxl_archive.py's staging area
    (data/tmp/jxl-archive/<upload_id>.jxl) is cleaned up in a `finally`
    block on every normal or exceptional path already -- a file surviving
    here past the TTL implies the process itself was killed mid-archival,
    not a code-path bug. Always safe to remove: nothing else ever reads
    from this directory (the verified blob is copied into the CAS blob
    store before this temp file would be cleaned up)."""
    jxl_tmp = data_dir / "tmp" / "jxl-archive"
    if not jxl_tmp.is_dir():
        return []
    out = []
    for f in jxl_tmp.glob("*.jxl"):
        age_hours = (now - f.stat().st_mtime) / 3600.0
        if age_hours >= ttl_hours:
            out.append(OrphanCandidate(f, age_hours, f.stat().st_size, "orphaned JXL archival staging file"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"), help="Vision data dir (default: %(default)s)")
    parser.add_argument("--uploads-dir", default=None, help="Defaults to <data-dir>/uploads")
    parser.add_argument("--ttl-hours", type=float, default=24.0, help="Age threshold in hours (default: 24)")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of candidates reported")
    parser.add_argument("--apply", action="store_true", help="Actually delete (default: dry run, report only)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    uploads_dir = Path(args.uploads_dir) if args.uploads_dir else data_dir / "uploads"
    now = time.time()

    candidates = _scan_incomplete_uploads(uploads_dir, args.ttl_hours, now)
    candidates += _scan_jxl_archive_tmp(data_dir, args.ttl_hours, now)

    deletable = [c for c in candidates if not c.anomaly]
    anomalies = [c for c in candidates if c.anomaly]

    if args.limit is not None:
        deletable = deletable[: args.limit]
        anomalies = anomalies[: args.limit]

    print(f"{'DRY RUN' if not args.apply else 'APPLY'} — TTL={args.ttl_hours}h, scanned under {data_dir}\n")

    if anomalies:
        print(f"ANOMALIES (never auto-deleted, needs human review) — {len(anomalies)}:")
        for c in anomalies:
            print(f"  {c.path}  age={c.age_hours:.1f}h  size={c.size_bytes}B  reason={c.reason}")
        print()

    total_bytes = sum(c.size_bytes for c in deletable)
    print(f"DELETABLE CANDIDATES — {len(deletable)} ({total_bytes} bytes total):")
    for c in deletable:
        print(f"  {c.path}  age={c.age_hours:.1f}h  size={c.size_bytes}B  reason={c.reason}")

    if not args.apply:
        print("\nDry run only — nothing was deleted. Re-run with --apply to delete the candidates above.")
        return 0

    removed = 0
    for c in deletable:
        try:
            if c.path.is_dir():
                shutil.rmtree(c.path)
            else:
                c.path.unlink()
            removed += 1
        except OSError as exc:
            print(f"  FAILED to remove {c.path}: {exc}")
    print(f"\nRemoved {removed}/{len(deletable)} candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
