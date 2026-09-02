"""Materialize a logical source through the storage abstraction.

Phase 9 of the storage integration campaign -- the deliberately-deferred
"riskier half" of CAS upload integration (see api/main.py's
_register_upload_blob docstring). Analysis code should call
materialize_source() rather than reaching for UploadRecord.completed_path
directly, so a future storage representation change (e.g. reversible JPEG
XL becoming the on-disk source, or a lossless COG derivative) doesn't need
every caller updated -- this is the one seam.

Correctness constraint that shaped this design: decoders (cv2.VideoCapture,
some FFmpeg builds, PIL in edge cases) can be sensitive to file extension,
not just magic bytes. A content-addressed blob's on-disk name is a bare
SHA-256 hex string with NO extension (see blob_store.py). Returning that
path directly to a video decoder would be a real, silent correctness risk.
So materialization never hands back the bare blob path: it produces (via
hardlink -- same bytes, zero copy, same filesystem) a same-content file
that keeps the original suffix, inside a dedicated materialized-sources
directory. Hardlink falls back to symlink, then to a full copy, if the
blob store and the materialized-sources directory ever end up on different
filesystems (e.g. a future deployment mounts them separately).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from cropmerge.storage.manifest import Manifest

log = logging.getLogger("cropmerge.storage.materialize")


def materialize_blob(blob_store, sha256: str, work_dir: Path, suffix: str) -> Path:
    """Materialize one CAS blob directly by its hash into a decoder-safe,
    suffix-preserving path -- the core hardlink/symlink/copy logic shared
    by materialize_source() below. Unlike materialize_source(), this RAISES
    on failure (FileNotFoundError if the blob is missing, or an OSError from
    a failed copy) rather than silently returning a fallback -- callers
    without a meaningful fallback path (e.g. a collection image whose only
    ever representation was its CAS blob) should catch and handle this
    themselves rather than have a fallback fabricated for them.
    """
    blob_path = blob_store.get(sha256)  # raises FileNotFoundError if missing
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    materialized_path = work_dir / f"{sha256}{suffix}"

    if materialized_path.is_file():
        return materialized_path

    try:
        os.link(blob_path, materialized_path)
    except OSError:
        try:
            materialized_path.symlink_to(blob_path)
        except OSError:
            shutil.copyfile(blob_path, materialized_path)
    return materialized_path


def materialize_source(
    manifest: Manifest,
    logical_id: str,
    fallback_path: Path,
    work_dir: Path,
    suffix: str | None = None,
) -> Path:
    """Return a real, decoder-safe file path for `logical_id`'s content.

    If `logical_id` has no manifest entry (registration was disabled,
    failed, or simply never ran for this object), or the blob is missing on
    disk, or materialization fails for any reason, `fallback_path` is
    returned unchanged and a warning is logged -- this function must never
    raise, and must never cause an analysis to read from a nonexistent
    file. `fallback_path` is expected to already exist (e.g.
    UploadRecord.completed_path) and is the caller's own responsibility.
    """
    fallback_path = Path(fallback_path)
    try:
        entry = manifest.get(logical_id)
        if entry is None:
            return fallback_path

        effective_suffix = suffix if suffix is not None else fallback_path.suffix
        return materialize_blob(manifest.blob_store, entry.sha256, work_dir, effective_suffix)
    except Exception:
        log.exception(
            "Materialization failed for logical_id=%s, falling back to %s", logical_id, fallback_path
        )
        return fallback_path
