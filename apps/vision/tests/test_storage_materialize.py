from __future__ import annotations

from pathlib import Path

from cropmerge.storage.blob_store import BlobStore
from cropmerge.storage.manifest import Manifest
from cropmerge.storage.materialize import materialize_source


def _manifest(tmp_path: Path) -> Manifest:
    blob_store = BlobStore(tmp_path / "blobs")
    return Manifest(tmp_path / "manifest.sqlite", blob_store=blob_store)


def test_materialize_falls_back_when_no_manifest_entry(tmp_path: Path):
    manifest = _manifest(tmp_path)
    fallback = tmp_path / "original.jpg"
    fallback.write_bytes(b"hello")

    result = materialize_source(manifest, "upload:missing", fallback, tmp_path / "materialized")

    assert result == fallback


def test_materialize_returns_hardlinked_copy_with_original_suffix(tmp_path: Path):
    manifest = _manifest(tmp_path)
    fallback = tmp_path / "original.jpg"
    content = b"real jpeg bytes here"
    fallback.write_bytes(content)
    blob_ref = manifest.blob_store.put(fallback)
    manifest.add("upload:abc", blob_ref, storage_class="scientific_source")

    result = materialize_source(manifest, "upload:abc", fallback, tmp_path / "materialized")

    assert result != fallback
    assert result.suffix == ".jpg"
    assert result.read_bytes() == content
    # Hardlink (or symlink/copy fallback) means this is the SAME bytes, not
    # a decoder-unsafe extensionless blob path.
    assert result.name == f"{blob_ref.sha256}.jpg"


def test_materialize_is_idempotent_on_repeated_calls(tmp_path: Path):
    manifest = _manifest(tmp_path)
    fallback = tmp_path / "original.mp4"
    fallback.write_bytes(b"video bytes")
    blob_ref = manifest.blob_store.put(fallback)
    manifest.add("upload:vid1", blob_ref, storage_class="scientific_source")

    first = materialize_source(manifest, "upload:vid1", fallback, tmp_path / "materialized")
    second = materialize_source(manifest, "upload:vid1", fallback, tmp_path / "materialized")

    assert first == second
    assert first.is_file()


def test_materialize_uses_explicit_suffix_over_fallback_suffix(tmp_path: Path):
    manifest = _manifest(tmp_path)
    fallback = tmp_path / "original"  # no suffix at all
    fallback.write_bytes(b"content")
    blob_ref = manifest.blob_store.put(fallback)
    manifest.add("upload:x", blob_ref, storage_class="scientific_source")

    result = materialize_source(manifest, "upload:x", fallback, tmp_path / "materialized", suffix=".mp4")

    assert result.suffix == ".mp4"


def test_materialize_never_raises_and_falls_back_on_missing_blob(tmp_path: Path):
    manifest = _manifest(tmp_path)
    fallback = tmp_path / "original.jpg"
    fallback.write_bytes(b"content")
    blob_ref = manifest.blob_store.put(fallback)
    manifest.add("upload:y", blob_ref, storage_class="scientific_source")

    # Simulate the blob having vanished from disk (e.g. manual tampering,
    # corrupted deploy) without touching the manifest row.
    manifest.blob_store.delete(blob_ref.sha256)

    result = materialize_source(manifest, "upload:y", fallback, tmp_path / "materialized")

    assert result == fallback
