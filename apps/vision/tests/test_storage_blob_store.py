from __future__ import annotations

import hashlib
import io

import pytest

from cropmerge.storage.blob_store import BlobStore, InvalidHashError


def test_put_and_get_roundtrip(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    content = b"hello field scan" * 1000
    src = tmp_path / "src.bin"
    src.write_bytes(content)

    ref = store.put(src)

    assert ref.sha256 == hashlib.sha256(content).hexdigest()
    assert ref.size_bytes == len(content)

    got_path = store.get(ref.sha256)
    assert got_path.is_file()
    assert got_path.read_bytes() == content


def test_put_from_stream(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    content = b"stream content" * 500
    ref = store.put(io.BytesIO(content))
    assert store.get(ref.sha256).read_bytes() == content


def test_dedup_identical_content_written_once(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    content = b"duplicate me" * 2000
    src1 = tmp_path / "a.bin"
    src2 = tmp_path / "b.bin"
    src1.write_bytes(content)
    src2.write_bytes(content)

    ref1 = store.put(src1)
    path1 = store.get(ref1.sha256)
    inode1 = path1.stat().st_ino
    mtime1 = path1.stat().st_mtime_ns

    ref2 = store.put(src2)
    path2 = store.get(ref2.sha256)

    assert ref1.sha256 == ref2.sha256
    assert path1 == path2
    # The second put must not have rewritten the blob.
    assert path2.stat().st_ino == inode1
    assert path2.stat().st_mtime_ns == mtime1

    # Only one blob file should exist on disk for this content (exclude
    # the store's own .tmp staging directory from the count).
    blob_files = [
        p
        for p in (tmp_path / "blobs").rglob("*")
        if p.is_file() and ".tmp" not in p.relative_to(tmp_path / "blobs").parts
    ]
    assert len(blob_files) == 1


def test_shard_layout(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    ref = store.put(io.BytesIO(b"shard test"))
    expected = tmp_path / "blobs" / ref.sha256[:2] / ref.sha256[2:4] / ref.sha256
    assert expected.is_file()


def test_get_missing_blob_raises(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    fake_hash = "0" * 64
    with pytest.raises(FileNotFoundError):
        store.get(fake_hash)


def test_path_for_rejects_invalid_hash(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    for bad in ["../../etc/passwd", "not-a-hash", "a" * 63, "A" * 64, ""]:
        with pytest.raises(InvalidHashError):
            store.path_for(bad)


def test_exists(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    ref = store.put(io.BytesIO(b"exists check"))
    assert store.exists(ref.sha256) is True
    assert store.exists("f" * 64) is False


def test_atomic_write_no_partial_blob_visible_on_failure(tmp_path):
    """If reading the source fails partway through, no corrupt blob must
    ever become visible at its content-addressed final path."""

    class FlakyStream:
        def __init__(self, good_chunk: bytes) -> None:
            self._good_chunk = good_chunk
            self._served = False

        def read(self, n: int = -1) -> bytes:
            if not self._served:
                self._served = True
                return self._good_chunk
            raise OSError("simulated mid-write failure")

    store = BlobStore(tmp_path / "blobs")
    blobs_dir = tmp_path / "blobs"

    with pytest.raises(OSError):
        store.put(FlakyStream(b"partial data that should never be committed"))

    # No blob should be visible anywhere under the store.
    visible_blobs = [p for p in blobs_dir.rglob("*") if p.is_file()]
    assert visible_blobs == []

    # No leftover temp files either.
    tmp_dir = blobs_dir / ".tmp"
    if tmp_dir.is_dir():
        assert list(tmp_dir.iterdir()) == []


def test_delete_removes_blob(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    ref = store.put(io.BytesIO(b"to be deleted"))
    assert store.exists(ref.sha256)

    removed = store.delete(ref.sha256)
    assert removed is True
    assert store.exists(ref.sha256) is False

    # Deleting again is a no-op, not an error.
    assert store.delete(ref.sha256) is False


def test_copy_out(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    content = b"copy me out"
    ref = store.put(io.BytesIO(content))
    dest = tmp_path / "exported" / "out.bin"
    result = store.copy_out(ref.sha256, dest)
    assert result == dest
    assert dest.read_bytes() == content
