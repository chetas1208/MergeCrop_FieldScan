from __future__ import annotations

import hashlib
from pathlib import Path

import jwt
from fastapi.testclient import TestClient

from api.main import _encode_artifact_token, _storage, create_app


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
        {"iss": "cropmerge-web", "purpose": "vision-session", "exp": 4_102_444_800},
        SECRET,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def test_health_is_public_and_does_not_expose_server_paths(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/vision/health")

    assert response.status_code == 200
    assert response.json()["service"] == "cropmerge-vision"
    assert "gpuBusy" in response.json()
    assert "queueDepth" in response.json()
    assert "databaseUrl" not in response.json()
    assert response.headers["cache-control"] == "no-store"


def test_chunked_upload_is_authenticated_and_assembled_atomically(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    assert client.post("/vision/uploads/init", json={"filename": "flight.mp4", "size": 5}).status_code == 401

    initialized = client.post(
        "/vision/uploads/init",
        json={"filename": "flight.mp4", "size": 5},
        headers=_headers(),
    )
    assert initialized.status_code == 200
    payload = initialized.json()
    assert payload["chunkSize"] == 3
    assert payload["totalChunks"] == 2

    upload_id = payload["uploadId"]
    assert client.put(
        f"/vision/uploads/{upload_id}/chunks/0",
        content=b"abc",
        headers={**_headers(), "content-type": "application/octet-stream"},
    ).status_code == 204
    assert client.put(
        f"/vision/uploads/{upload_id}/chunks/1",
        content=b"de",
        headers={**_headers(), "content-type": "application/octet-stream"},
    ).status_code == 204

    completed = client.post(f"/vision/uploads/{upload_id}/complete", headers=_headers())

    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert (tmp_path / "uploads" / "completed" / f"{upload_id}.mp4").read_bytes() == b"abcde"


def test_completed_upload_is_registered_in_content_addressed_storage(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    content = b"ab"  # <= VISION_SMALL_UPLOAD_THRESHOLD_BYTES (4) from _client()

    response = client.post(
        "/vision/uploads",
        files={"file": ("photo.jpg", content, "image/jpeg")},
        headers=_headers(),
    )
    assert response.status_code == 200
    upload_id = response.json()["uploadId"]

    expected_sha256 = hashlib.sha256(content).hexdigest()
    entry = _storage().get(f"upload:{upload_id}")
    assert entry is not None
    assert entry.sha256 == expected_sha256
    assert entry.size_bytes == len(content)
    # The blob itself is real, on disk, and byte-identical to the upload —
    # not just a manifest row pointing at nothing.
    assert _storage().blob_store.get(expected_sha256).read_bytes() == content


def test_duplicate_upload_content_dedupes_to_one_physical_blob(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    content = b"ab"

    first = client.post(
        "/vision/uploads", files={"file": ("a.jpg", content, "image/jpeg")}, headers=_headers()
    ).json()
    second = client.post(
        "/vision/uploads", files={"file": ("b.jpg", content, "image/jpeg")}, headers=_headers()
    ).json()
    assert first["uploadId"] != second["uploadId"]

    # Two separate on-disk completed files for now (Phase 6 only adds CAS
    # bookkeeping — it does not yet make the read path resolve through the
    # blob store), but exactly one physical blob and a refcount of 2.
    completed_dir = tmp_path / "uploads" / "completed"
    assert len(list(completed_dir.glob("*.jpg"))) == 2

    expected_sha256 = hashlib.sha256(content).hexdigest()
    assert _storage().ref_count(expected_sha256) == 2
    blob_shard_dir = tmp_path / "data" / "blobs" / expected_sha256[:2] / expected_sha256[2:4]
    assert len(list(blob_shard_dir.glob("*"))) == 1


def test_blob_registration_failure_does_not_fail_the_upload(tmp_path: Path, monkeypatch) -> None:
    """A storage/manifest problem must never take down the actual upload —
    see _register_upload_blob()'s docstring in api/main.py."""
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr("api.main._storage", lambda: (_ for _ in ()).throw(RuntimeError("disk full")))

    response = client.post(
        "/vision/uploads", files={"file": ("a.jpg", b"ab", "image/jpeg")}, headers=_headers()
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_signed_artifact_urls_are_path_bound_and_support_ranges(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    run_id = "a1b2c3d4e5f6"
    artifact = tmp_path / "outputs" / run_id / "annotated_video.mp4"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"abcdef")

    token = _encode_artifact_token(run_id, "annotated_video.mp4")
    ranged = client.get(
        f"/vision/artifacts/{run_id}/annotated_video.mp4",
        params={"token": token},
        headers={"Range": "bytes=1-3"},
    )

    assert ranged.status_code == 206
    assert ranged.content == b"bcd"
    assert ranged.headers["cache-control"] == "private, no-store"

    denied = client.get(
        f"/vision/artifacts/{run_id}/metrics.json",
        params={"token": token},
    )
    assert denied.status_code == 403


def test_cors_only_allows_configured_browser_origins(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.options(
        "/vision/uploads/init",
        headers={
            "Origin": "https://app.example.test",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://app.example.test"
