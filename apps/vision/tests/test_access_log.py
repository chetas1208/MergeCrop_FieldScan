from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from api.access_log import ACCESS_LOGGER
from api.main import create_app


@pytest.fixture
def access_log_capture():
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector()
    ACCESS_LOGGER.addHandler(handler)
    ACCESS_LOGGER.setLevel(logging.INFO)
    try:
        yield records
    finally:
        ACCESS_LOGGER.removeHandler(handler)


def test_access_log_middleware_skips_health_but_sets_request_id(monkeypatch, access_log_capture) -> None:
    monkeypatch.setenv("VISION_SHARED_SECRET", "0123456789abcdef0123456789abcdef")
    client = TestClient(create_app())
    response = client.get(
        "/vision/health",
        headers={"CF-Ray": "test-ray-id"},
    )

    assert response.status_code == 200
    assert response.headers.get("x-request-id")
    assert not access_log_capture


def test_access_log_middleware_logs_cloudflare_analyses_route(monkeypatch, access_log_capture) -> None:
    monkeypatch.setenv("VISION_SHARED_SECRET", "0123456789abcdef0123456789abcdef")
    client = TestClient(create_app())
    client.get(
        "/vision/analyses",
        headers={
            "CF-Connecting-IP": "203.0.113.10",
            "CF-Ray": "abc123",
            "Origin": "https://cropmerge-field-triage.vercel.app",
        },
    )

    assert access_log_capture
    hit = json.loads(access_log_capture[-1].message)
    assert hit["event"] == "api_access"
    assert hit["path"] == "/vision/analyses"
    assert hit["via_cloudflare"] is True
    assert hit["cf_ray"] == "abc123"
    assert hit["client_ip"] == "203.0.113.10"
