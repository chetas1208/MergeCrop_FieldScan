from __future__ import annotations

import io
from pathlib import Path

import jwt
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import _storage, create_app
from cropmerge.storage.jxl_archive import jxl_available, reconstruct_jpeg_bytes

pytestmark = pytest.mark.skipif(not jxl_available(), reason="pillow-jxl-plugin not installed")

SECRET = "0123456789abcdef0123456789abcdef"


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("VISION_SHARED_SECRET", SECRET)
    monkeypatch.setenv("VISION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VISION_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VISION_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VISION_JOB_DATABASE_PATH", str(tmp_path / "data" / "jobs.sqlite"))
    monkeypatch.setenv("VISION_UPLOAD_CHUNK_BYTES", str(1024 * 1024))
    monkeypatch.setenv("VISION_SMALL_UPLOAD_THRESHOLD_BYTES", str(2 * 1024 * 1024))
    monkeypatch.setenv("VISION_MAX_UPLOAD_BYTES", str(4 * 1024 * 1024))
    monkeypatch.setenv("VISION_CORS_ORIGINS", "https://app.example.test,http://localhost:3000")
    return TestClient(create_app())


def _headers() -> dict[str, str]:
    token = jwt.encode(
        {"iss": "cropmerge-web", "purpose": "vision-session", "exp": 4_102_444_800},
        SECRET,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def _real_jpeg_bytes() -> bytes:
    rng = np.random.default_rng(1)
    arr = (rng.random((400, 600, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def test_jpeg_upload_gets_a_verified_reversible_jxl_archive(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    content = _real_jpeg_bytes()

    response = client.post(
        "/vision/uploads", files={"file": ("photo.jpg", content, "image/jpeg")}, headers=_headers()
    )
    assert response.status_code == 200
    upload_id = response.json()["uploadId"]

    entry = _storage().get(f"upload:{upload_id}:jxl_archive")
    assert entry is not None, "expected a verified JXL archive to be registered"
    assert entry.storage_class == "reversible_source"

    jxl_blob_path = _storage().blob_store.get(entry.sha256)
    reconstructed = reconstruct_jpeg_bytes(jxl_blob_path)
    assert reconstructed == content

    # Temp working file must be cleaned up -- only the CAS blob remains.
    tmp_jxl_dir = tmp_path / "data" / "tmp" / "jxl-archive"
    assert not tmp_jxl_dir.exists() or list(tmp_jxl_dir.iterdir()) == []


def test_jxl_archival_failure_does_not_fail_the_upload(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "cropmerge.storage.jxl_archive.archive_jpeg_reversible",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("encoder crashed")),
    )

    response = client.post(
        "/vision/uploads",
        files={"file": ("photo.jpg", _real_jpeg_bytes(), "image/jpeg")},
        headers=_headers(),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_non_jpeg_upload_gets_no_jxl_archive(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    arr = (np.random.default_rng(2).random((100, 100, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")

    response = client.post(
        "/vision/uploads", files={"file": ("photo.png", buf.getvalue(), "image/png")}, headers=_headers()
    )
    assert response.status_code == 200
    upload_id = response.json()["uploadId"]
    assert _storage().get(f"upload:{upload_id}:jxl_archive") is None
