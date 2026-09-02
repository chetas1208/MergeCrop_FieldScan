"""Streaming SHA-256 hashing with bounded memory use.

Scientific-source media (drone video/imagery) can be multiple gigabytes.
Loading a whole file into memory to hash it is both wasteful and, on
constrained deployment hosts, a real OOM risk. Everything here reads in
fixed-size chunks so peak memory is O(chunk_size), not O(file_size).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

# 1 MiB chunks: large enough to avoid syscall overhead dominating, small
# enough that memory use stays flat regardless of input size.
DEFAULT_CHUNK_SIZE = 1024 * 1024


def hash_stream(stream: BinaryIO, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Compute the SHA-256 hex digest of a binary stream, reading in chunks.

    The stream is consumed from its current position to EOF. Callers that
    need the stream re-usable afterwards should seek it back themselves.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def hash_file(path: str | Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Compute the SHA-256 hex digest of a file on disk without loading it fully into RAM."""
    path = Path(path)
    with path.open("rb") as f:
        return hash_stream(f, chunk_size=chunk_size)


def is_valid_sha256_hex(value: str) -> bool:
    """True if `value` is a syntactically valid lowercase SHA-256 hex digest.

    Used to defend against path-traversal before any hash string is used to
    build a filesystem path (e.g. in blob_store.py).
    """
    if len(value) != 64:
        return False
    return all(c in "0123456789abcdef" for c in value)
