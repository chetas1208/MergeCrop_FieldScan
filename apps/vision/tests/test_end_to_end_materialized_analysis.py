"""Real end-to-end proof that Phase 9 (analysis reads through the storage
materialization seam instead of completed_path directly) does not break a
real analysis job. This is the one genuinely live-path-affecting change in
the storage integration campaign, so it gets a real HTTP-level test rather
than only unit-level coverage of materialize_source() in isolation.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import jwt
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.main import _storage, create_app

SECRET = "0123456789abcdef0123456789abcdef"


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("VISION_SHARED_SECRET", SECRET)
    monkeypatch.setenv("VISION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VISION_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VISION_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VISION_JOB_DATABASE_PATH", str(tmp_path / "data" / "jobs.sqlite"))
    monkeypatch.setenv("VISION_SMALL_UPLOAD_THRESHOLD_BYTES", str(2 * 1024 * 1024))
    monkeypatch.setenv("VISION_UPLOAD_CHUNK_BYTES", str(1024 * 1024))
    monkeypatch.setenv("VISION_MAX_UPLOAD_BYTES", str(4 * 1024 * 1024))
    monkeypatch.setenv("VISION_CORS_ORIGINS", "https://app.example.test,http://localhost:3000")
    return TestClient(create_app())


def _headers() -> dict[str, str]:
    token = jwt.encode(
        {"iss": "cropmerge-web", "purpose": "vision-session", "exp": 4_102_444_800}, SECRET, algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def _real_jpeg_bytes() -> bytes:
    from scripts.generate_demo_video import make_frame

    frame = make_frame(0.5)  # a real synthetic-field frame, not noise
    buf = io.BytesIO()
    Image.fromarray(frame[:, :, ::-1]).save(buf, format="JPEG", quality=90)  # BGR->RGB
    return buf.getvalue()


def test_analysis_completes_successfully_reading_the_materialized_source(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    content = _real_jpeg_bytes()

    upload_resp = client.post(
        "/vision/uploads", files={"file": ("field.jpg", content, "image/jpeg")}, headers=_headers()
    )
    assert upload_resp.status_code == 200
    upload_id = upload_resp.json()["uploadId"]

    # Confirm materialization actually has something to read through before
    # asserting the job succeeds -- otherwise this test would silently pass
    # via the completed_path fallback and prove nothing about the new path.
    assert _storage().get(f"upload:{upload_id}") is not None

    submit_resp = client.post(
        "/vision/analyses",
        json={"uploadId": upload_id, "segmentationBackend": "heuristic", "dinoBackend": "heuristic", "skipDino": True},
        headers=_headers(),
    )
    assert submit_resp.status_code == 202
    job_id = submit_resp.json()["id"]

    deadline = time.time() + 90
    status = None
    while time.time() < deadline:
        poll = client.get(f"/vision/analyses/{job_id}", headers=_headers())
        status = poll.json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert status == "completed", f"job did not complete in time (last status={status})"
    result = poll.json()
    assert result["report"]["sourceSha256"] == _storage().get(f"upload:{upload_id}").sha256
