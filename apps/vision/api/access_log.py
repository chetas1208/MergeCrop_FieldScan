from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

ACCESS_LOGGER = logging.getLogger("cropmerge.api.access")

# Uvicorn already prints generic lines; keep this focused on product API routes.
LOGGED_PREFIXES = ("/vision/",)


def _client_ip(request: Request) -> str:
    # Cloudflare Tunnel / proxy headers (never trust for auth — logging only).
    for header in ("cf-connecting-ip", "x-forwarded-for", "x-real-ip"):
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "-"


def _route_hint(path: str) -> str | None:
    parts = path.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "vision":
        if parts[1] == "analyses" and len(parts) >= 3:
            return parts[2]
        if parts[1] == "uploads" and len(parts) >= 3:
            return parts[2]
        if parts[1] == "artifacts" and len(parts) >= 3:
            return parts[2]
    return None


def _should_log(request: Request) -> bool:
    path = request.url.path
    if not path.startswith(LOGGED_PREFIXES):
        return False
    if path == "/vision/health" and request.method == "GET":
        return request.headers.get("x-cropmerge-access-health", "").lower() == "true"
    return True


def log_access_event(fields: dict[str, Any]) -> None:
    ACCESS_LOGGER.info(json.dumps(fields, separators=(",", ":"), default=str))


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        if not _should_log(request):
            response = await call_next(request)
            if request.url.path.startswith(LOGGED_PREFIXES):
                response.headers["X-Request-Id"] = request_id
            return response

        started = time.perf_counter()
        status_code = 500
        error_class: str | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-Id"] = request_id
            return response
        except Exception as exc:
            error_class = type(exc).__name__
            raise
        finally:
            if _should_log(request):
                duration_ms = round((time.perf_counter() - started) * 1000, 2)
                path = request.url.path
                fields: dict[str, Any] = {
                    "event": "api_access",
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status": status_code,
                    "duration_ms": duration_ms,
                    "client_ip": _client_ip(request),
                    "cf_ray": request.headers.get("cf-ray"),
                    "cf_visitor": request.headers.get("cf-visitor"),
                    "origin": request.headers.get("origin"),
                    "referer": request.headers.get("referer"),
                    "user_agent": request.headers.get("user-agent"),
                    "content_length": request.headers.get("content-length"),
                    "via_cloudflare": bool(
                        request.headers.get("cf-ray") or request.headers.get("cf-connecting-ip")
                    ),
                    "resource_id": _route_hint(path),
                }
                if error_class:
                    fields["error_class"] = error_class
                log_access_event(fields)
