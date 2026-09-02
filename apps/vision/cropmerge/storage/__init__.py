"""Content-addressed storage subsystem (Phase 1: foundational, non-destructive).

This package is intentionally standalone: it is not wired into the existing
upload/analysis pipeline (`api/main.py`, `cropmerge/pipeline/processor.py`).
That integration is a separate, riskier future phase.

Modules:
    hashing    - streaming SHA-256 with bounded memory use.
    blob_store - content-addressed blob storage with atomic writes + dedup.
    manifest   - logical-object -> blob mapping with reference counting.
    policies   - storage-class enum describing compression policy per class.
    retention  - config-driven TTL surface + pure orphaned-temp-file finder.

Nothing in this package performs lossy transcoding of scientific source
imagery. See docs/STORAGE_COMPRESSION_RESEARCH.md for the rationale.
"""

from __future__ import annotations

from cropmerge.storage.blob_store import BlobRef, BlobStore
from cropmerge.storage.hashing import hash_file, hash_stream
from cropmerge.storage.manifest import Manifest, ManifestEntry
from cropmerge.storage.policies import StorageClass

__all__ = [
    "BlobRef",
    "BlobStore",
    "Manifest",
    "ManifestEntry",
    "StorageClass",
    "hash_file",
    "hash_stream",
]
