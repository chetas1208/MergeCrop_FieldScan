from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from api.auth import SessionClaims, decode_session_token
from cropmerge.config import load_config
from cropmerge.live.export import build_live_exports
from cropmerge.live.hub import LiveEventHub
from cropmerge.live.rtsp_worker import LiveRtspWorker
from cropmerge.live.schemas import LiveSessionStatus, zone_from_camel_dict
from cropmerge.live.session_store import LiveSessionStore
from cropmerge.telemetry.association import nearest_telemetry
from cropmerge.telemetry.network_info import candidate_lan_addresses

log = logging.getLogger("cropmerge.live.router")

SESSION_ID_PREFIX_LEN = 12


def _live_settings() -> dict[str, Any]:
    stream_path = os.environ.get("LIVE_STREAM_PATH", "cropmerge/live")
    return {
        "stream_path": stream_path,
        "rtsp_url": os.environ.get("MEDIAMTX_RTSP_URL", f"rtsp://127.0.0.1:8554/{stream_path}"),
        "rtmp_url": os.environ.get("MEDIAMTX_RTMP_URL", f"rtmp://127.0.0.1:1935/{stream_path}"),
        "whep_public_url": os.environ.get("LIVE_WHEP_PUBLIC_URL", "http://127.0.0.1:8889").rstrip("/"),
        "sample_fps_default": float(os.environ.get("LIVE_SAMPLE_FPS_DEFAULT", "2")),
        "sample_fps_max": float(os.environ.get("LIVE_SAMPLE_FPS_MAX", "5")),
        "sample_video": os.environ.get(
            "LIVE_SAMPLE_VIDEO_PATH",
            str(Path(__file__).resolve().parents[4] / "apps" / "web" / "public" / "samples"),
        ),
    }


_store = LiveSessionStore()
_hub = LiveEventHub()
_workers: dict[str, LiveRtspWorker] = {}
_workers_lock = threading.Lock()
_simulate_process: subprocess.Popen | None = None
_simulate_lock = threading.Lock()


def _find_sample_video() -> Path | None:
    samples_dir = Path(_live_settings()["sample_video"])
    if not samples_dir.is_dir():
        return None
    for candidate in sorted(samples_dir.glob("*.mp4")):
        return candidate
    return None


def _output_dir(session_id: str) -> Path:
    from api.main import settings  # local import: avoid a module-load-time cycle with api.main

    return settings().output_dir / session_id


def create_live_router() -> APIRouter:
    router = APIRouter(prefix="/vision/live", tags=["live"])

    @router.get("/network-info")
    def network_info(_: SessionClaims):
        cfg = _live_settings()
        return {
            "candidateAddresses": candidate_lan_addresses(),
            "mqttPort": int(os.environ.get("MQTT_BROKER_PORT", "1883")),
            "rtmpPort": 1935,
            "rtspPort": 8554,
            "streamPath": cfg["stream_path"],
            "note": "Candidates only — confirm the address the RC Pro Enterprise can actually reach.",
        }

    @router.post("/sessions", status_code=201)
    def create_session(_: SessionClaims, sample_fps: float | None = None, simulated: bool = False):
        cfg = _live_settings()
        fps = sample_fps or cfg["sample_fps_default"]
        fps = max(0.1, min(fps, cfg["sample_fps_max"]))
        session = _store.create(
            stream_path=cfg["stream_path"], sample_fps=fps, simulated=simulated, whep_url=cfg["whep_public_url"]
        )
        return session.to_camel_dict()

    @router.get("/sessions")
    def list_sessions(_: SessionClaims, limit: int = 20):
        return {"sessions": [s.to_camel_dict() for s in _store.list(limit=limit)]}

    @router.get("/sessions/{session_id}")
    def get_session(session_id: str, _: SessionClaims):
        session = _store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Live session not found")
        return session.to_camel_dict()

    @router.post("/sessions/{session_id}/start-worker")
    def start_worker(session_id: str, _: SessionClaims):
        session = _store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Live session not found")
        cfg = _live_settings()
        with _workers_lock:
            if session_id in _workers:
                raise HTTPException(status_code=409, detail="Worker already running for this session")
            worker = LiveRtspWorker(
                session_id=session_id,
                rtsp_url=cfg["rtsp_url"],
                sample_fps=session.sample_fps,
                cfg=load_config(),
                hub=_hub,
                store=_store,
            )
            _workers[session_id] = worker
        worker.start()
        return {"status": "starting"}

    @router.post("/sessions/{session_id}/stop")
    def stop_session(session_id: str, _: SessionClaims):
        with _workers_lock:
            worker = _workers.pop(session_id, None)
        if worker is not None:
            worker.stop()
        _store.update_status(session_id, LiveSessionStatus.ENDED, ended=True)
        return {"status": "ended"}

    @router.post("/sessions/{session_id}/save-area")
    async def save_area_body(session_id: str, request: Request, _: SessionClaims):
        session = _store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Live session not found")
        body = await request.json()
        try:
            zone = zone_from_camel_dict(body["zone"])
            frame_ts_ms = int(body["liveFrameTimestampMs"])
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail="Body must include zone and liveFrameTimestampMs") from exc
        association = nearest_telemetry(_store, session_id, frame_ts_ms)
        saved = _store.save_area(session_id, zone, association, note=body.get("note"))
        return saved.to_camel_dict()

    @router.get("/sessions/{session_id}/saved-areas")
    def list_saved_areas(session_id: str, _: SessionClaims):
        return {"areas": [a.to_camel_dict() for a in _store.list_saved_areas(session_id)]}

    @router.get("/sessions/{session_id}/export")
    def export_session(session_id: str, request: Request, _: SessionClaims):
        session = _store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Live session not found")
        from api.main import _encode_artifact_token, _public_base, settings

        result = build_live_exports(session, _store, settings().output_dir)
        base = _public_base(request)
        artifact_urls = {
            name: f"{base}/vision/artifacts/{session_id}/{name}?token={_encode_artifact_token(session_id, name)}"
            for name in result["artifacts"]
        }
        return {"runId": session_id, "artifactUrls": artifact_urls}

    @router.post("/simulate/start")
    def simulate_start(_: SessionClaims):
        global _simulate_process
        with _simulate_lock:
            if _simulate_process is not None and _simulate_process.poll() is None:
                return {"status": "already_running"}
            sample = _find_sample_video()
            if sample is None:
                raise HTTPException(status_code=404, detail="No bundled sample video found to simulate a live feed")
            cfg = _live_settings()
            script = Path(__file__).resolve().parents[2] / "scripts" / "publish_simulated_live.py"
            _simulate_process = subprocess.Popen(
                ["python3", str(script), "--sample", str(sample), "--rtmp-url", cfg["rtmp_url"]],
            )
        return {"status": "starting", "sample": sample.name}

    @router.post("/simulate/stop")
    def simulate_stop(_: SessionClaims):
        global _simulate_process
        with _simulate_lock:
            if _simulate_process is not None and _simulate_process.poll() is None:
                _simulate_process.terminate()
                try:
                    _simulate_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _simulate_process.kill()
            _simulate_process = None
        return {"status": "stopped"}

    @router.websocket("/sessions/{session_id}/events")
    async def session_events(websocket: WebSocket, session_id: str):
        token = websocket.query_params.get("token", "")
        try:
            decode_session_token(token)
        except HTTPException:
            await websocket.close(code=4401)
            return
        _hub.bind_loop(asyncio.get_running_loop())
        await websocket.accept()
        queue = _hub.subscribe(session_id)
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            _hub.unsubscribe(session_id, queue)

    return router
