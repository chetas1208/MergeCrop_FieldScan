from __future__ import annotations

import io
import zipfile
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
    monkeypatch.setenv("VISION_CORS_ORIGINS", "https://app.example.test,http://localhost:3000")
    return TestClient(create_app())


def _headers() -> dict[str, str]:
    token = jwt.encode(
        {"iss": "cropmerge-web", "purpose": "vision-session", "exp": 4_102_444_800}, SECRET, algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _fake_jpeg(n: int = 200) -> bytes:
    return b"\xff\xd8\xff\xe0" + bytes(n)


def test_upload_requires_auth(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    zip_bytes = _make_zip({"a.jpg": _fake_jpeg()})

    response = client.post("/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")})

    assert response.status_code == 401


def test_upload_ingests_and_returns_summary(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    zip_bytes = _make_zip(
        {
            "variation-a/img1.jpg": _fake_jpeg(100),
            "variation-a/img2.jpg": _fake_jpeg(150),
            "variation-b/img1.jpg": _fake_jpeg(200),
            "notes.txt": b"not an image",
        }
    )

    response = client.post(
        "/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")}, headers=_headers()
    )

    assert response.status_code == 201
    body = response.json()
    assert body["summary"]["validImages"] == 3
    assert body["summary"]["rejectedMemberCount"] == 1
    assert sorted(body["summary"]["groups"]) == ["variation-a", "variation-b"]
    assert len(body["collectionId"]) == 12


def test_upload_rejects_path_traversal_archive_but_still_ingests_good_members(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    zip_bytes = _make_zip({"good.jpg": _fake_jpeg(), "../evil.jpg": _fake_jpeg()})

    response = client.post(
        "/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")}, headers=_headers()
    )

    assert response.status_code == 201
    body = response.json()
    assert body["summary"]["validImages"] == 1
    assert body["summary"]["rejectedMemberCount"] == 1
    assert "unsafe path" in body["rejectedMembers"][0]["reason"]


def test_malformed_zip_returns_400(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/vision/collections",
        files={"file": ("photos.zip", b"not a zip", "application/zip")},
        headers=_headers(),
    )

    assert response.status_code == 400


def test_get_collection_returns_persisted_record(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    zip_bytes = _make_zip({"a.jpg": _fake_jpeg()})
    created = client.post(
        "/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")}, headers=_headers()
    ).json()

    fetched = client.get(f"/vision/collections/{created['collectionId']}", headers=_headers())

    assert fetched.status_code == 200
    assert fetched.json()["collectionId"] == created["collectionId"]
    assert fetched.json()["summary"]["validImages"] == 1


def test_get_unknown_collection_returns_404(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.get("/vision/collections/aaaaaaaaaaaa", headers=_headers())

    assert response.status_code == 404


def test_get_collection_rejects_invalid_id_format(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.get("/vision/collections/not-a-valid-id", headers=_headers())

    assert response.status_code == 400


def test_duplicate_photo_within_zip_is_reflected_in_summary(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    same_content = _fake_jpeg(500)
    zip_bytes = _make_zip({"a.jpg": same_content, "b.jpg": same_content})

    response = client.post(
        "/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")}, headers=_headers()
    )

    body = response.json()
    assert body["summary"]["validImages"] == 1
    assert body["summary"]["duplicateImages"] == 1


def test_no_temp_zip_files_left_behind_after_ingestion(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    zip_bytes = _make_zip({"a.jpg": _fake_jpeg()})

    client.post(
        "/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")}, headers=_headers()
    )

    tmp_dir = tmp_path / "data" / "tmp" / "collection-uploads"
    assert not tmp_dir.exists() or list(tmp_dir.iterdir()) == []
