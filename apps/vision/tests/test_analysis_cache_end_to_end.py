"""Real end-to-end proof that Phase 17 (skip re-running the heavy pipeline
for an identical source+config) works through the actual HTTP job flow, not
just the pure compute_cache_key()/AnalysisCache unit tests."""
from __future__ import annotations

import io
import time
from pathlib import Path
from urllib.parse import urlsplit

import jwt
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app

SECRET = "0123456789abcdef0123456789abcdef"


def _client(tmp_path: Path, monkeypatch, cache_enabled: bool) -> TestClient:
    monkeypatch.setenv("VISION_SHARED_SECRET", SECRET)
    monkeypatch.setenv("VISION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VISION_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VISION_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VISION_JOB_DATABASE_PATH", str(tmp_path / "data" / "jobs.sqlite"))
    monkeypatch.setenv("VISION_SMALL_UPLOAD_THRESHOLD_BYTES", str(2 * 1024 * 1024))
    monkeypatch.setenv("VISION_UPLOAD_CHUNK_BYTES", str(1024 * 1024))
    monkeypatch.setenv("VISION_MAX_UPLOAD_BYTES", str(4 * 1024 * 1024))
    monkeypatch.setenv("VISION_CORS_ORIGINS", "https://app.example.test,http://localhost:3000")
    monkeypatch.setenv("CROPMERGE_ANALYSIS_CACHE_ENABLED", "true" if cache_enabled else "false")
    return TestClient(create_app())


def _headers() -> dict[str, str]:
    token = jwt.encode(
        {"iss": "cropmerge-web", "purpose": "vision-session", "exp": 4_102_444_800}, SECRET, algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def _real_jpeg_bytes() -> bytes:
    from scripts.generate_demo_video import make_frame

    frame = make_frame(0.5)
    buf = io.BytesIO()
    Image.fromarray(frame[:, :, ::-1]).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _upload_and_analyze(client: TestClient, content: bytes) -> tuple[str, dict]:
    upload_resp = client.post(
        "/vision/uploads", files={"file": ("field.jpg", content, "image/jpeg")}, headers=_headers()
    )
    assert upload_resp.status_code == 200
    upload_id = upload_resp.json()["uploadId"]

    submit_resp = client.post(
        "/vision/analyses",
        json={"uploadId": upload_id, "segmentationBackend": "heuristic", "dinoBackend": "heuristic", "skipDino": True},
        headers=_headers(),
    )
    assert submit_resp.status_code == 202
    job_id = submit_resp.json()["id"]

    deadline = time.time() + 90
    result = None
    while time.time() < deadline:
        poll = client.get(f"/vision/analyses/{job_id}", headers=_headers())
        result = poll.json()
        if result["status"] in ("completed", "failed"):
            break
        time.sleep(0.3)
    assert result["status"] == "completed", f"job did not complete (status={result['status']})"
    return job_id, result


def _artifact_url_paths(artifact_urls: dict[str, str]) -> dict[str, str]:
    """Compare the URL path (which embeds run_id and artifact name), not
    the full URL: the ?token=... query string is a freshly-minted JWT with
    a per-request iat/exp timestamp (see api/main.py::_encode_artifact_token),
    so two otherwise-identical artifactUrls responses legitimately differ
    in their token bytes even a few hundred milliseconds apart -- comparing
    full URLs is inherently flaky, not a signal of an actual cache miss."""
    return {name: urlsplit(url).path for name, url in artifact_urls.items()}


def test_second_identical_analysis_hits_cache_and_reuses_the_run(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch, cache_enabled=True)
    content = _real_jpeg_bytes()

    first_job_id, first_result = _upload_and_analyze(client, content)
    second_job_id, second_result = _upload_and_analyze(client, content)  # same bytes -> same blob_sha256

    assert second_job_id != first_job_id
    # The cache hit's report intentionally keeps the ORIGINAL run's id so
    # artifact URLs resolve to the already-existing files (see
    # JobManager._try_analysis_cache_hit's docstring).
    assert second_result["report"]["runId"] == first_result["report"]["runId"] == first_job_id
    assert second_result["report"]["sourceSha256"] == first_result["report"]["sourceSha256"]
    assert _artifact_url_paths(second_result["artifactUrls"]) == _artifact_url_paths(first_result["artifactUrls"])


def test_analysis_cache_disabled_by_default_runs_pipeline_twice(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch, cache_enabled=False)
    content = _real_jpeg_bytes()

    first_job_id, first_result = _upload_and_analyze(client, content)
    second_job_id, second_result = _upload_and_analyze(client, content)

    assert second_result["report"]["runId"] == second_job_id  # its own run, not reused
    assert first_result["report"]["runId"] == first_job_id


def test_different_config_options_are_not_cache_hits(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch, cache_enabled=True)
    content = _real_jpeg_bytes()

    upload_resp = client.post(
        "/vision/uploads", files={"file": ("field.jpg", content, "image/jpeg")}, headers=_headers()
    )
    upload_id = upload_resp.json()["uploadId"]

    submit_a = client.post(
        "/vision/analyses",
        json={"uploadId": upload_id, "segmentationBackend": "heuristic", "dinoBackend": "heuristic", "skipDino": True, "sampleFps": 1},
        headers=_headers(),
    )
    job_a = submit_a.json()["id"]

    upload_resp2 = client.post(
        "/vision/uploads", files={"file": ("field.jpg", content, "image/jpeg")}, headers=_headers()
    )
    upload_id2 = upload_resp2.json()["uploadId"]
    submit_b = client.post(
        "/vision/analyses",
        json={"uploadId": upload_id2, "segmentationBackend": "heuristic", "dinoBackend": "heuristic", "skipDino": True, "sampleFps": 2},
        headers=_headers(),
    )
    job_b = submit_b.json()["id"]

    def _wait(job_id: str) -> dict:
        deadline = time.time() + 90
        result = None
        while time.time() < deadline:
            result = client.get(f"/vision/analyses/{job_id}", headers=_headers()).json()
            if result["status"] in ("completed", "failed"):
                break
            time.sleep(0.3)
        assert result["status"] == "completed"
        return result

    result_a = _wait(job_a)
    result_b = _wait(job_b)

    # Different sampleFps -> different cache key -> job_b must run its OWN pipeline.
    assert result_b["report"]["runId"] == job_b
    assert result_a["report"]["runId"] == job_a
