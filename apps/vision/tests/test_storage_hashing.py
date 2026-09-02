from __future__ import annotations

import hashlib
import resource

import pytest

from cropmerge.storage.hashing import (
    DEFAULT_CHUNK_SIZE,
    hash_file,
    hash_stream,
    is_valid_sha256_hex,
)

LARGE_FILE_SIZE = 60 * 1024 * 1024  # 60 MiB — big enough to expose a full-read regression


def _write_random_file(path, size: int, seed: int = 1234) -> None:
    import random

    rng = random.Random(seed)
    remaining = size
    chunk = 1024 * 1024
    with path.open("wb") as f:
        while remaining > 0:
            n = min(chunk, remaining)
            f.write(rng.randbytes(n))
            remaining -= n


def test_hash_file_matches_hashlib_full_read(tmp_path):
    path = tmp_path / "large.bin"
    _write_random_file(path, LARGE_FILE_SIZE)

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = hash_file(path)

    assert actual == expected


def test_hash_stream_matches_hash_file(tmp_path):
    path = tmp_path / "small.bin"
    _write_random_file(path, 1024 * 500, seed=7)

    via_file = hash_file(path)
    with path.open("rb") as f:
        via_stream = hash_stream(f)

    assert via_file == via_stream


def test_hash_file_empty_file(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert hash_file(path) == hashlib.sha256(b"").hexdigest()


def test_hash_stream_rejects_non_positive_chunk_size(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    with path.open("rb") as f, pytest.raises(ValueError):
        hash_stream(f, chunk_size=0)


def test_is_valid_sha256_hex():
    assert is_valid_sha256_hex("a" * 64) is True
    assert is_valid_sha256_hex("A" * 64) is False  # must be lowercase
    assert is_valid_sha256_hex("a" * 63) is False
    assert is_valid_sha256_hex("../../etc/passwd") is False
    assert is_valid_sha256_hex("g" * 64) is False  # non-hex char


@pytest.mark.skipif(not hasattr(resource, "getrusage"), reason="resource module unavailable")
def test_hash_file_uses_bounded_memory_not_full_file_read(tmp_path):
    """Hashing a large file must not require loading it fully into RAM.

    Heuristic: compare the process's resident-memory high-water mark before
    and after streaming-hashing a large file. A naive `f.read()` of the
    whole file would grow RSS by roughly the file size; chunked reading
    (DEFAULT_CHUNK_SIZE) should grow it by a small fraction of that.
    """
    path = tmp_path / "huge.bin"
    _write_random_file(path, LARGE_FILE_SIZE)

    # Let allocator settle and get a baseline.
    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    digest = hash_file(path, chunk_size=DEFAULT_CHUNK_SIZE)

    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    grown_kb = max(0, after_kb - before_kb)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    # ru_maxrss is in KiB on Linux. A full-file read would grow RSS by
    # roughly LARGE_FILE_SIZE (60MB+). Bounded chunked reads should stay
    # well under half that — generous slack for allocator/GC noise.
    file_size_kb = LARGE_FILE_SIZE / 1024
    assert grown_kb < file_size_kb * 0.5, (
        f"RSS grew by {grown_kb} KiB while hashing a {file_size_kb:.0f} KiB file; "
        "hashing appears to be loading the whole file into memory instead of streaming it"
    )
