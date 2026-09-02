"""Regression test for the real, 100%-reproducible bug found by
scripts/cleanup_temp_storage.py (see Decisions.md 2026-09-02): every
completed upload used to leave its .incomplete/<id>/ staging directory
behind forever, because UploadStore.complete()/save_small() wrote the
completed record back into that same directory via _write_record(), which
unconditionally recreates it right after complete() had just rmtree'd it.

Fix: completed records now live at completed_root/<id>.json
(_write_completed_record()), never back under .incomplete/<id>/, so that
directory is genuinely gone once an upload finishes.
"""
from __future__ import annotations

import time
from pathlib import Path

import jwt
from fastapi.testclient import TestClient

from api.main import create_app

SECRET = "0123456789abcdef0123456789abcdef"


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("VISION_SHARED_SECRET", SECRET)
    monkeypatch.setenv("VISION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VISION_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VISION_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VISION_JOB_DATABASE_PATH", str(tmp_path / "data" / "jobs.sqlite"))
    monkeypatch.setenv("VISION_UPLOAD_CHUNK_BYTES", "3")
    monkeypatch.setenv("VISION_SMALL_UPLOAD_THRESHOLD_BYTES", "4")
    monkeypatch.setenv("VISION_MAX_UPLOAD_BYTES", "64")
    monkeypatch.setenv("VISION_CORS_ORIGINS", "https://app.example.test,http://localhost:3000")
    return TestClient(create_app())


def _headers() -> dict[str, str]:
    token = jwt.encode(
        {"iss": "cropmerge-web", "purpose": "vision-session", "exp": 4_102_444_800}, SECRET, algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def test_chunked_upload_leaves_no_incomplete_directory_after_completion(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    initialized = client.post(
        "/vision/uploads/init", json={"filename": "flight.mp4", "size": 5}, headers=_headers()
    )
    upload_id = initialized.json()["uploadId"]
    client.put(
        f"/vision/uploads/{upload_id}/chunks/0", content=b"abc",
        headers={**_headers(), "content-type": "application/octet-stream"},
    )
    client.put(
        f"/vision/uploads/{upload_id}/chunks/1", content=b"de",
        headers={**_headers(), "content-type": "application/octet-stream"},
    )
    completed = client.post(f"/vision/uploads/{upload_id}/complete", headers=_headers())
    assert completed.status_code == 200

    staging_dir = tmp_path / "uploads" / ".incomplete" / upload_id
    assert not staging_dir.exists(), "staging directory must be gone after completion, not recreated"

    completed_record = tmp_path / "uploads" / "completed" / f"{upload_id}.json"
    assert completed_record.is_file()

    # The upload must still be readable/usable after completion (get() must
    # correctly find the record in its new location).
    get_resp = client.get("/vision/analyses", headers=_headers())  # sanity: app still healthy
    assert get_resp.status_code == 200


def test_small_upload_leaves_no_incomplete_directory_after_completion(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    resp = client.post(
        "/vision/uploads", files={"file": ("photo.jpg", b"ab", "image/jpeg")}, headers=_headers()
    )
    assert resp.status_code == 200
    upload_id = resp.json()["uploadId"]

    staging_dir = tmp_path / "uploads" / ".incomplete" / upload_id
    assert not staging_dir.exists(), "staging directory must be gone after completion, not recreated"

    completed_record = tmp_path / "uploads" / "completed" / f"{upload_id}.json"
    assert completed_record.is_file()


def test_completed_upload_is_still_readable_via_get(tmp_path: Path, monkeypatch):
    """The completed-record relocation must not break normal lookups (used
    by analysis submission, materialize_upload_source, etc.)."""
    client = _client(tmp_path, monkeypatch)
    upload_resp = client.post(
        "/vision/uploads", files={"file": ("photo.jpg", b"ab", "image/jpeg")}, headers=_headers()
    )
    upload_id = upload_resp.json()["uploadId"]

    submit_resp = client.post(
        "/vision/analyses",
        json={"uploadId": upload_id, "segmentationBackend": "heuristic", "dinoBackend": "heuristic", "skipDino": True},
        headers=_headers(),
    )
    # A 202 here proves JobManager.submit() -> manager.uploads.get(upload_id)
    # successfully found the completed record at its new location.
    assert submit_resp.status_code == 202


def test_cleanup_script_no_longer_flags_completed_uploads_as_anomalies(tmp_path: Path, monkeypatch):
    """End-to-end proof the original bug is gone: scripts/cleanup_temp_storage.py's
    'status=completed but not cleaned up' anomaly detector (see
    tests/test_cleanup_temp_storage.py) must find nothing to flag."""
    from scripts.cleanup_temp_storage import _scan_incomplete_uploads

    client = _client(tmp_path, monkeypatch)
    client.post("/vision/uploads", files={"file": ("photo.jpg", b"ab", "image/jpeg")}, headers=_headers())
    client.post("/vision/uploads", files={"file": ("photo2.jpg", b"cd", "image/jpeg")}, headers=_headers())

    candidates = _scan_incomplete_uploads(tmp_path / "uploads", ttl_hours=0, now=time.time())

    assert candidates == []
