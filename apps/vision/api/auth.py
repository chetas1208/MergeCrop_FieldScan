"""Shared vision-session JWT auth, split out of api/main.py so
cropmerge/live/router.py can depend on it without an import cycle
(api.main imports and mounts the live router)."""

from __future__ import annotations

import os
from typing import Annotated, Any

import jwt
from fastapi import Depends, Header, HTTPException


def _require_secret() -> str:
    secret = os.environ.get("VISION_SHARED_SECRET", "")
    if len(secret.encode("utf-8")) < 32:
        raise HTTPException(
            status_code=503,
            detail="Vision authentication is not configured. Set VISION_SHARED_SECRET to a 32-byte secret.",
        )
    return secret


def decode_session(authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    try:
        claims = jwt.decode(
            authorization.removeprefix("Bearer ").strip(),
            _require_secret(),
            algorithms=["HS256"],
            issuer="cropmerge-web",
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Session token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid session token") from exc
    if claims.get("purpose") != "vision-session":
        raise HTTPException(status_code=403, detail="Token purpose is not valid for vision API access")
    return claims


def decode_session_token(token: str) -> dict[str, Any]:
    """Same validation as decode_session, for transports (WebSocket) that
    can't set an Authorization header and pass the token another way."""
    if not token:
        raise HTTPException(status_code=401, detail="Session token required")
    try:
        claims = jwt.decode(token, _require_secret(), algorithms=["HS256"], issuer="cropmerge-web")
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Session token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid session token") from exc
    if claims.get("purpose") != "vision-session":
        raise HTTPException(status_code=403, detail="Token purpose is not valid for vision API access")
    return claims


SessionClaims = Annotated[dict[str, Any], Depends(decode_session)]
