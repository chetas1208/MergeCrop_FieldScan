from __future__ import annotations

import io

import pytest

from cropmerge.storage.blob_store import BlobStore
from cropmerge.storage.manifest import Manifest


@pytest.fixture
def store(tmp_path):
    return BlobStore(tmp_path / "blobs")


@pytest.fixture
def manifest(tmp_path, store):
    m = Manifest(tmp_path / "manifest.sqlite", blob_store=store)
    yield m
    m.close()


def test_add_and_get(manifest, store):
    ref = store.put(io.BytesIO(b"content A"))
    entry = manifest.add("upload:1", ref, storage_class="scientific_source")

    fetched = manifest.get("upload:1")
    assert fetched is not None
    assert fetched.sha256 == ref.sha256
    assert fetched.size_bytes == ref.size_bytes
    assert fetched.storage_class == "scientific_source"
    assert entry == fetched


def test_get_missing_returns_none(manifest):
    assert manifest.get("does-not-exist") is None


def test_multiple_logical_objects_can_share_one_blob(manifest, store):
    ref = store.put(io.BytesIO(b"shared content"))
    manifest.add("upload:1", ref)
    manifest.add("upload:2", ref)

    assert manifest.ref_count(ref.sha256) == 2
    assert store.exists(ref.sha256)


def test_removing_one_reference_keeps_blob_if_still_referenced(manifest, store):
    ref = store.put(io.BytesIO(b"shared content 2"))
    manifest.add("upload:1", ref)
    manifest.add("upload:2", ref)

    removed = manifest.remove("upload:1")

    assert removed is True
    assert manifest.get("upload:1") is None
    assert manifest.get("upload:2") is not None
    assert manifest.ref_count(ref.sha256) == 1
    # Blob must still exist — upload:2 still references it.
    assert store.exists(ref.sha256) is True


def test_removing_last_reference_deletes_blob(manifest, store):
    ref = store.put(io.BytesIO(b"only referenced once"))
    manifest.add("upload:1", ref)

    manifest.remove("upload:1")

    assert manifest.ref_count(ref.sha256) == 0
    assert store.exists(ref.sha256) is False


def test_remove_nonexistent_logical_id_returns_false(manifest):
    assert manifest.remove("never-added") is False


def test_manifest_without_blob_store_does_not_touch_blobs(tmp_path, store):
    ref = store.put(io.BytesIO(b"no blob store wired"))
    m = Manifest(tmp_path / "manifest2.sqlite", blob_store=None)
    m.add("upload:x", ref)
    m.remove("upload:x")
    m.close()

    # Manifest had no blob_store, so it must never have attempted deletion;
    # the blob is orphaned but intact, not corrupted or half-deleted.
    assert store.exists(ref.sha256) is True


def test_reassigning_logical_id_to_new_blob_drops_old_reference(manifest, store):
    ref_a = store.put(io.BytesIO(b"version A"))
    ref_b = store.put(io.BytesIO(b"version B"))

    manifest.add("doc:1", ref_a)
    assert store.exists(ref_a.sha256)

    manifest.add("doc:1", ref_b)  # re-point doc:1 at a new blob

    fetched = manifest.get("doc:1")
    assert fetched.sha256 == ref_b.sha256
    # ref_a had exactly one reference (doc:1), now reassigned -> deleted.
    assert store.exists(ref_a.sha256) is False
    assert store.exists(ref_b.sha256) is True


def test_reassigning_logical_id_keeps_old_blob_if_still_referenced_elsewhere(manifest, store):
    ref_a = store.put(io.BytesIO(b"shared version A"))
    ref_b = store.put(io.BytesIO(b"version B distinct"))

    manifest.add("doc:1", ref_a)
    manifest.add("doc:2", ref_a)  # second reference to ref_a
    manifest.add("doc:1", ref_b)  # doc:1 moves to ref_b

    assert manifest.get("doc:2").sha256 == ref_a.sha256
    assert store.exists(ref_a.sha256) is True  # doc:2 still needs it


def test_list_by_blob_and_list_all(manifest, store):
    ref = store.put(io.BytesIO(b"listing test"))
    manifest.add("a", ref)
    manifest.add("b", ref)

    by_blob = manifest.list_by_blob(ref.sha256)
    assert {e.logical_id for e in by_blob} == {"a", "b"}

    all_entries = manifest.list_all()
    assert {e.logical_id for e in all_entries} == {"a", "b"}


def test_manifest_persists_across_reopen(tmp_path, store):
    db_path = tmp_path / "persist.sqlite"
    ref = store.put(io.BytesIO(b"persisted content"))

    m1 = Manifest(db_path, blob_store=store)
    m1.add("upload:persist", ref)
    m1.close()

    m2 = Manifest(db_path, blob_store=store)
    entry = m2.get("upload:persist")
    m2.close()

    assert entry is not None
    assert entry.sha256 == ref.sha256
