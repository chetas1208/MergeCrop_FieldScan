from __future__ import annotations

from pathlib import Path

from api.main import UploadRecord, _materialize_upload_source, _register_upload_blob, _storage


def _configure_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_SHARED_SECRET", "0123456789abcdef0123456789abcdef")
    monkeypatch.setenv("VISION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VISION_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VISION_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VISION_JOB_DATABASE_PATH", str(tmp_path / "data" / "jobs.sqlite"))


def _completed_upload(tmp_path: Path, content: bytes, suffix: str = ".mp4") -> UploadRecord:
    completed_dir = tmp_path / "uploads" / "completed"
    completed_dir.mkdir(parents=True, exist_ok=True)
    target = completed_dir / f"upload1{suffix}"
    target.write_bytes(content)
    record = UploadRecord(
        id="upload1", filename=f"f{suffix}", suffix=suffix, size=len(content), chunk_size=len(content),
        total_chunks=1, status="completed", created_at="t", completed_path=str(target),
    )
    return record


def test_materializes_to_a_hardlinked_copy_with_same_content(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    content = b"real video bytes"
    record = _completed_upload(tmp_path, content)
    record = _register_upload_blob(record)
    assert record.blob_sha256 is not None

    materialized = _materialize_upload_source(record)

    assert materialized != Path(record.completed_path)
    assert materialized.suffix == ".mp4"
    assert materialized.read_bytes() == content


def test_falls_back_to_completed_path_when_registration_never_happened(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    record = _completed_upload(tmp_path, b"content")
    # deliberately skip _register_upload_blob -- no manifest entry exists

    materialized = _materialize_upload_source(record)

    assert materialized == Path(record.completed_path)


def test_falls_back_when_storage_raises(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    record = _completed_upload(tmp_path, b"content")
    record = _register_upload_blob(record)

    monkeypatch.setattr("api.main._storage", lambda: (_ for _ in ()).throw(RuntimeError("disk full")))

    materialized = _materialize_upload_source(record)

    assert materialized == Path(record.completed_path)
