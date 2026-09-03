from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import jwt
from fastapi.testclient import TestClient

from api.main import create_app
from cropmerge.eval.synthetic_field import make_synthetic_frame

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


def _synthetic_jpeg_bytes(t: float) -> bytes:
    frame = make_synthetic_frame(t)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_full_collection_flow_upload_analyze_get(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    zip_bytes = _make_zip(
        {
            "variation-a/img1.jpg": _synthetic_jpeg_bytes(0.3),
            "variation-a/img2.jpg": _synthetic_jpeg_bytes(0.7),
            "variation-b/img1.jpg": _synthetic_jpeg_bytes(1.1),
        }
    )

    upload_resp = client.post(
        "/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")}, headers=_headers()
    )
    assert upload_resp.status_code == 201
    collection_id = upload_resp.json()["collectionId"]
    assert upload_resp.json()["summary"]["validImages"] == 3

    analyze_resp = client.post(f"/vision/collections/{collection_id}/analyze", json={}, headers=_headers())
    assert analyze_resp.status_code == 200
    analysis = analyze_resp.json()
    assert analysis["summary"]["analyzedCount"] == 3
    assert analysis["summary"]["failedCount"] == 0

    groups = {g["group"] for g in analysis["groupStatistics"]}
    assert groups == {"variation-a", "variation-b"}
    for g in analysis["groupStatistics"]:
        assert g["cropCoverageMedian"] is not None
        assert g["cropCoverageIqr"] is not None

    fetched = client.get(f"/vision/collections/{collection_id}/analysis", headers=_headers())
    assert fetched.status_code == 200
    assert fetched.json()["collectionId"] == collection_id


def test_get_analysis_before_analyzing_returns_404(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    zip_bytes = _make_zip({"a.jpg": _synthetic_jpeg_bytes(0.5)})
    upload = client.post(
        "/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")}, headers=_headers()
    ).json()

    response = client.get(f"/vision/collections/{upload['collectionId']}/analysis", headers=_headers())

    assert response.status_code == 404


def test_analyze_requires_auth(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.post("/vision/collections/aaaaaaaaaaaa/analyze", json={})

    assert response.status_code == 401


def test_analyze_unknown_collection_returns_404(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.post("/vision/collections/aaaaaaaaaaaa/analyze", json={}, headers=_headers())

    assert response.status_code == 404


def test_collection_image_detail_and_artifacts_are_reachable(tmp_path: Path, monkeypatch):
    """The per-image detail view (Phase: expose ZIP collections in the UI)
    needs a real, per-image report + signed artifact URLs -- collection
    images are analyzed directly via FieldTriageProcessor, not through the
    JobManager, so they have no JobStore-backed run_id and need their own
    route (see api/main.py's _safe_collection_image_artifact_path)."""
    client = _client(tmp_path, monkeypatch)
    zip_bytes = _make_zip({"a/img1.jpg": _synthetic_jpeg_bytes(0.3)})

    upload = client.post(
        "/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")}, headers=_headers()
    ).json()
    collection_id = upload["collectionId"]

    analysis = client.post(f"/vision/collections/{collection_id}/analyze", json={}, headers=_headers()).json()
    image_result = analysis["imageResults"][0]
    assert image_result["status"] == "analyzed"
    image_id = image_result["imageId"]

    detail_resp = client.get(f"/vision/collections/{collection_id}/images/{image_id}", headers=_headers())
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["runId"]
    assert "source" not in detail or "path" not in (detail.get("source") or {})
    assert "heatmap.png" in detail["artifactUrls"]

    artifact_url = detail["artifactUrls"]["heatmap.png"]
    path_and_query = artifact_url.split(str(client.base_url), 1)[-1]
    artifact_resp = client.get(path_and_query)
    assert artifact_resp.status_code == 200
    assert artifact_resp.headers["content-type"].startswith("image/")


def test_collection_image_detail_requires_auth(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.get("/vision/collections/aaaaaaaaaaaa/images/bbbbbbbbbbbb")

    assert response.status_code == 401


def test_collection_image_detail_404_before_analysis(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    zip_bytes = _make_zip({"img1.jpg": _synthetic_jpeg_bytes(0.3)})
    upload = client.post(
        "/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")}, headers=_headers()
    ).json()

    response = client.get(
        f"/vision/collections/{upload['collectionId']}/images/aaaaaaaaaaaa", headers=_headers()
    )

    assert response.status_code == 404


def test_collection_image_artifact_rejects_wrong_token(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    zip_bytes = _make_zip({"img1.jpg": _synthetic_jpeg_bytes(0.3)})
    upload = client.post(
        "/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")}, headers=_headers()
    ).json()
    collection_id = upload["collectionId"]
    analysis = client.post(f"/vision/collections/{collection_id}/analyze", json={}, headers=_headers()).json()
    image_id = analysis["imageResults"][0]["imageId"]

    response = client.get(
        f"/vision/collections/{collection_id}/images/{image_id}/artifacts/heatmap.png?token=not-a-real-token"
    )

    assert response.status_code == 401


def test_collection_image_artifact_rejects_path_traversal(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    zip_bytes = _make_zip({"img1.jpg": _synthetic_jpeg_bytes(0.3)})
    upload = client.post(
        "/vision/collections", files={"file": ("photos.zip", zip_bytes, "application/zip")}, headers=_headers()
    ).json()
    collection_id = upload["collectionId"]
    analysis = client.post(f"/vision/collections/{collection_id}/analyze", json={}, headers=_headers()).json()
    image_id = analysis["imageResults"][0]["imageId"]
    detail = client.get(f"/vision/collections/{collection_id}/images/{image_id}", headers=_headers()).json()
    token = detail["artifactUrls"]["heatmap.png"].split("token=", 1)[-1]

    response = client.get(
        f"/vision/collections/{collection_id}/images/{image_id}/artifacts/../../../etc/passwd?token={token}"
    )

    assert response.status_code in (400, 401, 404)
