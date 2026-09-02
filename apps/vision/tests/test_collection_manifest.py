from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

from cropmerge.storage.blob_store import BlobStore
from cropmerge.storage.collection_manifest import ingest_zip_collection
from cropmerge.storage.manifest import Manifest
from cropmerge.storage.zip_collection import ZipLimits


def _manifest(tmp_path: Path) -> Manifest:
    blob_store = BlobStore(tmp_path / "blobs")
    return Manifest(tmp_path / "manifest.sqlite", blob_store=blob_store)


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _fake_jpeg(n: int = 200) -> bytes:
    return b"\xff\xd8\xff\xe0" + bytes(n)


def test_ingests_flat_collection_and_registers_in_cas(tmp_path: Path):
    manifest = _manifest(tmp_path)
    zip_bytes = _make_zip({"a.jpg": _fake_jpeg(100), "b.jpg": _fake_jpeg(200)})

    result = ingest_zip_collection(zip_bytes, manifest, collection_id="col1")

    assert result.archive_rejected is None
    assert len(result.valid_images) == 2
    assert result.duplicate_images == []
    for img in result.valid_images:
        entry = manifest.get(f"collection:col1:image:{img.id}")
        assert entry is not None
        assert entry.sha256 == img.source_sha256
        assert manifest.blob_store.get(entry.sha256).is_file()


def test_duplicate_content_within_one_zip_is_marked_and_shares_one_blob(tmp_path: Path):
    manifest = _manifest(tmp_path)
    content = _fake_jpeg(150)
    zip_bytes = _make_zip({"a.jpg": content, "b.jpg": content})  # byte-identical

    result = ingest_zip_collection(zip_bytes, manifest, collection_id="col1")

    assert len(result.valid_images) == 1
    assert len(result.duplicate_images) == 1
    assert result.duplicate_images[0].duplicate_of == result.valid_images[0].id
    assert result.duplicate_images[0].source_sha256 == result.valid_images[0].source_sha256

    # Two logical manifest entries, one physical blob.
    sha = result.valid_images[0].source_sha256
    assert manifest.ref_count(sha) == 2
    shard_dir = tmp_path / "blobs" / sha[:2] / sha[2:4]
    assert len(list(shard_dir.glob("*"))) == 1


def test_duplicate_against_preexisting_cas_content_is_detected(tmp_path: Path):
    manifest = _manifest(tmp_path)
    content = _fake_jpeg(100)
    # Pre-register the same content under a different logical id (e.g. from
    # a prior single-image upload, before this collection was ever ingested).
    blob_ref = manifest.blob_store.put(io.BytesIO(content))
    manifest.add("upload:preexisting", blob_ref, storage_class="scientific_source")

    zip_bytes = _make_zip({"a.jpg": content})
    result = ingest_zip_collection(zip_bytes, manifest, collection_id="col1")

    assert len(result.images) == 1
    assert result.images[0].status == "duplicate"
    assert result.images[0].source_sha256 == hashlib.sha256(content).hexdigest()


def test_preserves_group_metadata_from_folder_structure(tmp_path: Path):
    manifest = _manifest(tmp_path)
    zip_bytes = _make_zip(
        {"variation-a/img1.jpg": _fake_jpeg(50), "variation-b/img1.jpg": _fake_jpeg(60)}
    )

    result = ingest_zip_collection(zip_bytes, manifest, collection_id="col1")

    groups = {img.archive_path: img.relative_group for img in result.valid_images}
    assert groups == {"variation-a/img1.jpg": "variation-a", "variation-b/img1.jpg": "variation-b"}


def test_rejected_members_are_reported_not_silently_dropped(tmp_path: Path):
    manifest = _manifest(tmp_path)
    zip_bytes = _make_zip({"good.jpg": _fake_jpeg(50), "../escape.jpg": _fake_jpeg(50)})

    result = ingest_zip_collection(zip_bytes, manifest, collection_id="col1")

    assert len(result.valid_images) == 1
    assert len(result.rejected) == 1
    assert "unsafe path" in result.rejected[0].reason


def test_whole_archive_rejection_produces_no_images(tmp_path: Path):
    manifest = _manifest(tmp_path)
    zip_bytes = _make_zip({f"img{i}.jpg": _fake_jpeg(10) for i in range(5)})
    limits = ZipLimits(max_member_count=2)

    result = ingest_zip_collection(zip_bytes, manifest, collection_id="col1", limits=limits)

    assert result.archive_rejected is not None
    assert result.images == []


def test_failed_member_extraction_is_recorded_not_raised(tmp_path: Path, monkeypatch):
    manifest = _manifest(tmp_path)
    zip_bytes = _make_zip({"a.jpg": _fake_jpeg(50)})

    monkeypatch.setattr(
        "cropmerge.storage.collection_manifest.extract_member_bytes",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("corrupt member")),
    )

    result = ingest_zip_collection(zip_bytes, manifest, collection_id="col1")

    assert len(result.images) == 1
    assert result.images[0].status == "failed"
    assert "corrupt member" in result.images[0].error


def test_auto_generates_collection_id_when_not_provided(tmp_path: Path):
    manifest = _manifest(tmp_path)
    zip_bytes = _make_zip({"a.jpg": _fake_jpeg(50)})

    result = ingest_zip_collection(zip_bytes, manifest)

    assert result.collection_id
    assert len(result.collection_id) == 12
