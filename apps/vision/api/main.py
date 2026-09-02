from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import jwt
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.access_log import AccessLogMiddleware
from api.auth import SessionClaims, _require_secret
from cropmerge import __version__
from cropmerge.config import load_config
from cropmerge.db import list_runs
from cropmerge.features.dinov3 import dinov3_available
from cropmerge.logging_utils import setup_access_logging, setup_logging
from cropmerge.enrich.llm_explain import health_status as llm_health_status
from cropmerge.enrich.llm_explain import llm_enrichment_enabled, summarize_zone
from cropmerge.enrich.llm_explain import unload as unload_llm
from cropmerge.live.router import create_live_router
from cropmerge.pipeline.processor import FieldTriageProcessor
from cropmerge.pipeline.schemas import ANALYSIS_VERSION, FieldTriageReport
from cropmerge.storage.analysis_cache import AnalysisCache, compute_cache_key, config_content_hash
from cropmerge.segmentation.sam3 import sam3_available
from cropmerge.pipeline.collection_analysis import CollectionImageInput, analyze_collection_images
from cropmerge.storage.blob_store import BlobStore
from cropmerge.storage.collection_manifest import CollectionIngestResult, ingest_zip_collection
from cropmerge.storage.manifest import Manifest
from cropmerge.storage.materialize import materialize_blob, materialize_source
from cropmerge.storage.policies import StorageClass
from cropmerge.storage.zip_collection import ZipLimits

try:
    from cropmerge.features.dinov2 import dinov2_available
except Exception:  # pragma: no cover
    def dinov2_available() -> bool:  # type: ignore
        return False

try:
    from cropmerge.segmentation.sam2 import sam2_available
except Exception:  # pragma: no cover
    def sam2_available() -> bool:  # type: ignore
        return False


log = logging.getLogger("cropmerge.api.main")

setup_logging(os.environ.get("CROP_MERGE_LOG_LEVEL", "INFO"))
setup_access_logging(
    log_path=os.environ.get("VISION_API_LOG_PATH", "").strip() or None,
    level=os.environ.get("VISION_ACCESS_LOG_LEVEL", "INFO"),
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}
UPLOAD_ID_RE = re.compile(r"^[a-f0-9]{32}$")
JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")
ACTIVE_JOB_STATUSES = frozenset({"preparing", "processing", "rendering"})


class VisionSettings(BaseModel):
    shared_secret: str
    cors_origins: list[str]
    data_dir: Path
    upload_dir: Path
    output_dir: Path
    job_database_path: Path
    public_base_url: str
    chunk_bytes: int
    small_upload_threshold_bytes: int
    max_upload_bytes: int
    artifact_token_ttl_seconds: int


def _path_env(name: str, fallback: Path | str) -> Path:
    return Path(os.environ.get(name, str(fallback))).expanduser().resolve()


def settings() -> VisionSettings:
    data_dir = _path_env("VISION_DATA_DIR", REPO_ROOT / "data")
    upload_dir = _path_env("VISION_UPLOAD_DIR", os.environ.get("UPLOADS_DIR", data_dir / "uploads"))
    output_dir = _path_env(
        "VISION_OUTPUT_DIR",
        os.environ.get("CROP_MERGE_OUTPUTS", os.environ.get("OUTPUTS_DIR", REPO_ROOT / "outputs")),
    )
    job_db = _path_env("VISION_JOB_DATABASE_PATH", data_dir / "db" / "vision-jobs.sqlite")
    cors_origins = [
        origin.strip().rstrip("/")
        for origin in os.environ.get(
            "VISION_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if origin.strip()
    ]
    chunk_bytes = int(os.environ.get("VISION_UPLOAD_CHUNK_BYTES", str(8 * 1024 * 1024)))
    small_threshold = int(os.environ.get("VISION_SMALL_UPLOAD_THRESHOLD_BYTES", str(24 * 1024 * 1024)))
    max_upload_bytes = int(os.environ.get("VISION_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024 * 1024)))
    if chunk_bytes <= 0 or small_threshold <= 0 or max_upload_bytes <= 0:
        raise RuntimeError("Vision upload limits must be positive")
    if chunk_bytes > max_upload_bytes or small_threshold > max_upload_bytes:
        raise RuntimeError("Vision upload limits are inconsistent")
    return VisionSettings(
        shared_secret=os.environ.get("VISION_SHARED_SECRET", ""),
        cors_origins=cors_origins,
        data_dir=data_dir,
        upload_dir=upload_dir,
        output_dir=output_dir,
        job_database_path=job_db,
        public_base_url=os.environ.get("VISION_PUBLIC_BASE_URL", "").rstrip("/"),
        chunk_bytes=chunk_bytes,
        small_upload_threshold_bytes=small_threshold,
        max_upload_bytes=max_upload_bytes,
        artifact_token_ttl_seconds=int(os.environ.get("VISION_ARTIFACT_TOKEN_TTL_SECONDS", "3600")),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_artifact_token(run_id: str, name: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "iss": "cropmerge-vision",
            "purpose": "artifact",
            "artifact": f"{run_id}/{name}",
            "iat": now,
            "exp": now + timedelta(seconds=settings().artifact_token_ttl_seconds),
        },
        _require_secret(),
        algorithm="HS256",
    )


def _verify_artifact_token(token: str, run_id: str, name: str) -> None:
    try:
        claims = jwt.decode(token, _require_secret(), algorithms=["HS256"], issuer="cropmerge-vision")
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Artifact link expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid artifact link") from exc
    if claims.get("purpose") != "artifact" or claims.get("artifact") != f"{run_id}/{name}":
        raise HTTPException(status_code=403, detail="Artifact link is not valid for this file")


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


def _safe_extension(filename: str) -> str:
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="A filename is required")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported type: {suffix or '(none)'}. Allowed: {allowed}")
    return suffix


def _valid_upload_id(upload_id: str) -> str:
    if not UPLOAD_ID_RE.fullmatch(upload_id):
        raise HTTPException(status_code=400, detail="Invalid upload ID")
    return upload_id


def _valid_job_id(job_id: str) -> str:
    if not JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=400, detail="Invalid analysis ID")
    return job_id


COLLECTION_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def _valid_collection_id(collection_id: str) -> str:
    if not COLLECTION_ID_RE.fullmatch(collection_id):
        raise HTTPException(status_code=400, detail="Invalid collection ID")
    return collection_id


def _safe_artifact_path(run_id: str, name: str) -> Path:
    _valid_job_id(run_id)
    pure_name = PurePosixPath(name)
    if not name or pure_name.is_absolute() or any(part in {"", ".", ".."} for part in pure_name.parts):
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    root = (settings().output_dir / run_id).resolve()
    path = (root / Path(*pure_name.parts)).resolve()
    if path != root and root not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    return path


class UploadRecord(BaseModel):
    id: str
    filename: str
    suffix: str
    size: int
    chunk_size: int
    total_chunks: int
    status: str
    created_at: str
    completed_path: str | None = None
    blob_sha256: str | None = None


class UploadInitRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)
    content_type: str | None = Field(default=None, max_length=255)


DEFAULT_SAMPLE_FPS = 1.0
# Safety ceiling, not a target: covers a full ~30 min drone clip at 1 FPS
# (matches the product's own "analyze every scheduled second" requirement)
# without letting a pathologically long upload run the single-worker GPU
# executor forever.
MAX_FRAMES_CEILING = 1800


class AnalysisRequest(BaseModel):
    upload_id: str = Field(alias="uploadId")
    sample_fps: float = Field(default=DEFAULT_SAMPLE_FPS, alias="sampleFps", gt=0, le=30)
    # None (the default) means: cover the whole video at sample_fps, up to
    # MAX_FRAMES_CEILING — see JobManager._run(). An explicit value still
    # acts as a hard cap for callers that want one.
    max_frames: int | None = Field(default=None, alias="maxFrames", gt=0, le=MAX_FRAMES_CEILING)
    skip_dino: bool = Field(default=False, alias="skipDino")
    segmentation_backend: str = Field(default="heuristic", alias="segmentationBackend", min_length=1, max_length=64)
    dino_backend: str = Field(default="heuristic", alias="dinoBackend", min_length=1, max_length=64)

    model_config = {"populate_by_name": True}


class CollectionAnalysisRequest(BaseModel):
    segmentation_backend: str = Field(default="heuristic", alias="segmentationBackend", min_length=1, max_length=64)
    dino_backend: str = Field(default="heuristic", alias="dinoBackend", min_length=1, max_length=64)

    model_config = {"populate_by_name": True}


class UploadStore:
    _lock = threading.Lock()

    def __init__(self, cfg: VisionSettings):
        self.cfg = cfg
        self.tmp_root = cfg.upload_dir / ".incomplete"
        self.completed_root = cfg.upload_dir / "completed"
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.completed_root.mkdir(parents=True, exist_ok=True)

    def _dir(self, upload_id: str) -> Path:
        return self.tmp_root / _valid_upload_id(upload_id)

    def _record_path(self, upload_id: str) -> Path:
        return self._dir(upload_id) / "upload.json"

    def _completed_record_path(self, upload_id: str) -> Path:
        """Completed-upload metadata lives beside the completed file itself
        (completed_root/<id>.json), NOT under .incomplete/<id>/ — keeping it
        there was a real, systematic bug (see docs/); complete()/save_small()
        delete the .incomplete/<id>/ staging directory once an upload
        finishes, and any record written back under that path would just
        recreate the very directory that was just deleted.
        """
        return self.completed_root / f"{_valid_upload_id(upload_id)}.json"

    def _write_record(self, record: UploadRecord) -> None:
        """Persist an IN-PROGRESS (status='uploading') record under its
        .incomplete/<id>/ staging directory. Never called for a completed
        record — see _write_completed_record()."""
        directory = self._dir(record.id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "upload.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
            handle.write(record.model_dump_json())
            temp = Path(handle.name)
        temp.replace(target)

    def _write_completed_record(self, record: UploadRecord) -> None:
        """Persist a COMPLETED record under completed_root, and remove its
        now-superseded .incomplete/<id>/ staging directory (chunk parts for
        the chunked-upload path, or just the initial 'uploading' record for
        the small-upload fast path) -- the fix for the directory-leak bug."""
        target = self._completed_record_path(record.id)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.completed_root, delete=False) as handle:
            handle.write(record.model_dump_json())
            temp = Path(handle.name)
        temp.replace(target)
        shutil.rmtree(self._dir(record.id), ignore_errors=True)

    def get(self, upload_id: str) -> UploadRecord:
        path = self._completed_record_path(upload_id)
        if not path.is_file():
            path = self._record_path(upload_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Upload not found")
        try:
            return UploadRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Upload metadata is unreadable") from exc

    def create(self, filename: str, size: int, *, chunk_size: int) -> UploadRecord:
        if size > self.cfg.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Upload exceeds VISION_MAX_UPLOAD_BYTES")
        suffix = _safe_extension(filename)
        upload_id = uuid.uuid4().hex
        total_chunks = max(1, (size + chunk_size - 1) // chunk_size)
        record = UploadRecord(
            id=upload_id,
            filename=Path(filename).name,
            suffix=suffix,
            size=size,
            chunk_size=chunk_size,
            total_chunks=total_chunks,
            status="uploading",
            created_at=_now(),
        )
        with self._lock:
            self._write_record(record)
        return record

    def write_chunk(self, upload_id: str, index: int, data: bytes) -> UploadRecord:
        record = self.get(upload_id)
        if record.status != "uploading":
            raise HTTPException(status_code=409, detail="Upload is not accepting chunks")
        if index < 0 or index >= record.total_chunks:
            raise HTTPException(status_code=400, detail="Chunk index is outside the upload range")
        expected = min(record.chunk_size, record.size - index * record.chunk_size)
        if len(data) != expected:
            raise HTTPException(status_code=400, detail=f"Chunk size must be exactly {expected} bytes")
        directory = self._dir(upload_id)
        target = directory / f"chunk_{index:08d}.part"
        with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
            handle.write(data)
            temp = Path(handle.name)
        temp.replace(target)
        return record

    def complete(self, upload_id: str) -> UploadRecord:
        with self._lock:
            record = self.get(upload_id)
            if record.status == "completed":
                return record
            if record.status != "uploading":
                raise HTTPException(status_code=409, detail="Upload cannot be completed")
            directory = self._dir(upload_id)
            parts = [directory / f"chunk_{index:08d}.part" for index in range(record.total_chunks)]
            if any(not part.is_file() for part in parts):
                raise HTTPException(status_code=409, detail="Upload has missing chunks")
            if sum(part.stat().st_size for part in parts) != record.size:
                raise HTTPException(status_code=409, detail="Upload size does not match declared size")
            target = self.completed_root / f"{record.id}{record.suffix}"
            with tempfile.NamedTemporaryFile("wb", dir=self.completed_root, delete=False) as handle:
                for part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, handle, length=1024 * 1024)
                temp = Path(handle.name)
            temp.replace(target)
            record.status = "completed"
            record.completed_path = str(target)
            record = _register_upload_blob(record)
            self._write_completed_record(record)
            return record

    def save_small(self, file: UploadFile) -> UploadRecord:
        suffix = _safe_extension(file.filename or "")
        record = self.create(file.filename or "upload", 1, chunk_size=self.cfg.small_upload_threshold_bytes)
        target = self.completed_root / f"{record.id}{suffix}"
        temp: Path | None = None
        written = 0
        try:
            with tempfile.NamedTemporaryFile("wb", dir=self.completed_root, delete=False) as handle:
                temp = Path(handle.name)
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > self.cfg.small_upload_threshold_bytes or written > self.cfg.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Upload is too large for the multipart fast path")
                    handle.write(chunk)
            if written == 0:
                raise HTTPException(status_code=400, detail="Empty upload")
            temp.replace(target)
        except Exception:
            if temp is not None:
                temp.unlink(missing_ok=True)
            raise
        record.size = written
        record.total_chunks = 1
        record.status = "completed"
        record.completed_path = str(target)
        record = _register_upload_blob(record)
        with self._lock:
            self._write_completed_record(record)
        return record

    def cancel(self, upload_id: str) -> None:
        with self._lock:
            record = self.get(upload_id)
            if record.status == "completed" and record.completed_path:
                Path(record.completed_path).unlink(missing_ok=True)
            shutil.rmtree(self._dir(upload_id), ignore_errors=True)


class JobStore:
    _lock = threading.Lock()

    def __init__(self, cfg: VisionSettings):
        self.path = cfg.job_database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _init(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS vision_jobs (
                    id TEXT PRIMARY KEY,
                    upload_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    stage TEXT,
                    message TEXT,
                    error TEXT,
                    options_json TEXT NOT NULL,
                    report_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_vision_jobs_created ON vision_jobs (created_at DESC);
                """
            )
            connection.execute(
                """
                UPDATE vision_jobs
                SET status='failed', error='Vision service restarted before this analysis completed',
                    message='Analysis interrupted by service restart', updated_at=?
                WHERE status IN ('queued', 'preparing', 'processing', 'rendering')
                """,
                (_now(),),
            )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["options"] = json.loads(out.pop("options_json"))
        report = out.pop("report_json")
        out["report"] = json.loads(report) if report else None
        return out

    def create(self, upload: UploadRecord, request: AnalysisRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        now = _now()
        values = {
            "id": job_id,
            "upload_id": upload.id,
            "filename": upload.filename,
            "created_at": now,
            "updated_at": now,
            "status": "queued",
            "progress": 0.0,
            "stage": "queued",
            "message": "Queued for vision processing",
            "error": None,
            "options_json": json.dumps(request.model_dump(by_alias=False)),
            "report_json": None,
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO vision_jobs (
                    id, upload_id, filename, created_at, updated_at, status, progress,
                    stage, message, error, options_json, report_json
                ) VALUES (:id, :upload_id, :filename, :created_at, :updated_at, :status, :progress,
                          :stage, :message, :error, :options_json, :report_json)
                """,
                values,
            )
        result = dict(values)
        result["options"] = json.loads(result.pop("options_json"))
        result["report"] = None
        result.pop("report_json")
        return result

    def get(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM vision_jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return self._row(row)

    def list(self, limit: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM vision_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(row) for row in rows]

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        if not changes:
            return self.get(job_id)
        allowed = {"status", "progress", "stage", "message", "error", "report_json"}
        if not set(changes).issubset(allowed):
            raise ValueError("Unexpected job update field")
        changes["updated_at"] = _now()
        assignments = ", ".join(f"{name} = :{name}" for name in changes)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE vision_jobs SET {assignments} WHERE id = :id", changes | {"id": job_id}
            )
            if cursor.rowcount != 1:
                raise HTTPException(status_code=404, detail="Analysis not found")
        return self.get(job_id)


def _public_report(report: dict[str, Any]) -> dict[str, Any]:
    """Remove server-local paths before a report leaves the vision host."""
    result = json.loads(json.dumps(report))
    source = result.get("source")
    if isinstance(source, dict):
        source.pop("path", None)
    artifacts = result.get("artifacts")
    safe_names = {
        "resultsJson": "results.json",
        "annotatedVideo": "annotated_video.mp4",
        "heatmapPng": "heatmap.png",
        "metricsJson": "metrics.json",
        "framesDir": "frames",
        "overlaysDir": "overlays",
    }
    if isinstance(artifacts, dict):
        for key, safe_name in safe_names.items():
            artifacts[key] = safe_name if artifacts.get(key) else None
    return result


def _enrich_zones_with_llm(report: dict[str, Any]) -> None:
    """Best-effort, in-place: adds `llmSummary` to each inspection zone dict
    when CROP_MERGE_LLM_ENRICHMENT=true and the model loads successfully.
    Never raises into the pipeline and never fabricates a summary — a zone
    simply has no `llmSummary` key when enrichment is off/unavailable, and
    the UI falls back to the existing heuristic `reasons` text.
    """
    if not llm_enrichment_enabled():
        return
    for zone in report.get("inspectionZones", []):
        try:
            summary = summarize_zone(
                primary_signal_label=str(zone.get("primarySignalLabel", "")),
                reasons=list(zone.get("reasons", [])),
                location=str(zone.get("relativeLocation", "")),
                review_priority=str(zone.get("reviewPriority", "")),
            )
        except Exception:
            summary = None
        if summary:
            zone["llmSummary"] = summary


def _existing_artifact_names(report: dict[str, Any]) -> list[str]:
    run_id = str(report.get("runId", ""))
    names = ["results.json", "metrics.json", "heatmap.png", "annotated_video.mp4", "segmentation_montage.jpg"]
    for frame in report.get("frameQuality", []):
        index = int(frame.get("frameIndex", 0))
        names.extend((f"frames/frame_{index:04d}.jpg", f"overlays/overlay_{index:04d}.jpg"))
    return [name for name in names if _safe_artifact_path(run_id, name).is_file()]


def _public_base(request: Request) -> str:
    configured = settings().public_base_url
    return configured or str(request.base_url).rstrip("/")


def _artifact_urls(report: dict[str, Any], request: Request) -> dict[str, str]:
    run_id = str(report["runId"])
    base = _public_base(request)
    return {
        name: f"{base}/vision/artifacts/{run_id}/{name}?token={_encode_artifact_token(run_id, name)}"
        for name in _existing_artifact_names(report)
    }


def _public_job(job: dict[str, Any], request: Request) -> dict[str, Any]:
    out = {
        "id": job["id"],
        "status": job["status"],
        "createdAt": job["created_at"],
        "updatedAt": job["updated_at"],
        "filename": job["filename"],
        "progress": float(job["progress"]),
        "stage": job["stage"],
        "message": job["message"],
        "error": job["error"],
        "report": None,
        "artifactUrls": {},
    }
    if job["report"]:
        out["report"] = _public_report(job["report"])
        out["artifactUrls"] = _artifact_urls(job["report"], request)
    return out


class JobManager:
    def __init__(self, cfg: VisionSettings):
        self.cfg = cfg
        self.uploads = UploadStore(cfg)
        self.jobs = JobStore(cfg)
        self.analysis_cache = AnalysisCache(cfg.data_dir / "db" / "analysis-cache.sqlite")
        # The CV pipeline is GPU-heavy; this POC intentionally runs one job at a time.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cropmerge-vision")

    def queue_status(self) -> dict[str, Any]:
        jobs = self.jobs.list(limit=100)
        active = [job for job in jobs if job["status"] in ACTIVE_JOB_STATUSES]
        queued = [job for job in jobs if job["status"] == "queued"]
        gpu_ok, gpu_message = _gpu_memory_ok()
        gpu_busy = len(active) > 0 or not gpu_ok
        return {
            "gpuBusy": gpu_busy,
            "gpuAvailable": gpu_ok,
            "gpuMessage": gpu_message,
            "activeJobId": active[0]["id"] if active else None,
            "queueDepth": len(queued),
            "maxQueueDepth": _max_queue_depth(),
        }

    def submit(self, upload: UploadRecord, request: AnalysisRequest) -> dict[str, Any]:
        status = self.queue_status()
        if not status["gpuAvailable"]:
            raise HTTPException(
                status_code=503,
                detail=status["gpuMessage"] or "GPU is unavailable on the vision server.",
            )
        if status["queueDepth"] >= status["maxQueueDepth"]:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"GPU queue is full ({status['queueDepth']} waiting). "
                    "Try again in a few minutes."
                ),
            )
        job = self.jobs.create(upload, request)
        refreshed = self.queue_status()
        if refreshed["gpuBusy"] or refreshed["queueDepth"] > 1:
            ahead = max(0, refreshed["queueDepth"] - 1)
            job = self.jobs.update(
                job["id"],
                message=f"Waiting for GPU — {ahead} job(s) ahead",
            )
        self.executor.submit(self._run, job["id"])
        return job

    def _run(self, job_id: str) -> None:
        try:
            job = self.jobs.get(job_id)
            if job["status"] == "cancelled":
                return
            self.jobs.update(job_id, status="preparing", progress=0.05, stage="preparing", message="Preparing input")
            upload = self.uploads.get(str(job["upload_id"]))
            if upload.status != "completed" or not upload.completed_path:
                raise RuntimeError("Completed upload is unavailable")
            self.jobs.update(job_id, status="processing", progress=0.1, stage="processing", message="Running vision pipeline")
            options = job["options"]
            cfg = load_config()

            cache_hit = self._try_analysis_cache_hit(upload, options, cfg)
            if cache_hit is not None:
                report = cache_hit
            else:
                input_path = _materialize_upload_source(upload)
                report_model = FieldTriageProcessor(cfg).process(
                    input_path,
                    self.cfg.output_dir,
                    sample_fps=float(options["sample_fps"]),
                    # None means "cover the whole video" — sample_frames() already
                    # stops at end-of-stream naturally; the ceiling only guards
                    # against pathologically long uploads.
                    max_frames=int(options["max_frames"]) if options["max_frames"] is not None else MAX_FRAMES_CEILING,
                    skip_dino=bool(options["skip_dino"]),
                    segmentation_backend=str(options["segmentation_backend"]),
                    dino_backend=str(options["dino_backend"]),
                    run_id=job_id,
                )
                # Provenance: the upload's CAS hash (see _register_upload_blob())
                # travels onto the result so a result stays forensically traceable
                # to its exact source bytes even as storage/compression policy
                # evolves later.
                report_model.source_sha256 = upload.blob_sha256
                report_model.config_version = config_content_hash(cfg)
                report = report_model.to_camel_dict()
                _release_gpu_cache()
                self._maybe_store_analysis_cache_entry(upload, options, cfg, job_id)

            self.jobs.update(job_id, status="rendering", progress=0.95, stage="rendering", message="Finalizing artifacts")
            # A cache hit's report keeps its ORIGINAL runId on purpose (see
            # _try_analysis_cache_hit) so artifact URLs resolve to the
            # already-existing files rather than duplicating them — this
            # job never owns its own output_dir/<job_id>/ directory in that
            # case, so it must not write into it or re-run LLM enrichment
            # against it (which would try to write there too).
            owns_own_artifacts = report.get("runId") == job_id
            if owns_own_artifacts:
                safe_report = _public_report(report)
                _safe_artifact_path(job_id, "results.json").write_text(json.dumps(safe_report, indent=2), encoding="utf-8")
                _register_run_artifacts_in_cas(job_id, report_model)
            self.jobs.update(
                job_id,
                status="completed",
                progress=1.0,
                stage="completed",
                message="Analysis complete",
                report_json=json.dumps(report),
            )
            # Enrichment is a non-blocking add-on: the job is already
            # "completed" with the full deterministic report above. This
            # patches richer descriptions in afterward (frontend already
            # polls job status, so no extra wiring needed) rather than
            # delaying "completed" behind an LLM load/generate.
            if owns_own_artifacts and llm_enrichment_enabled():
                threading.Thread(
                    target=self._enrich_after_completion, args=(job_id, report), daemon=True,
                    name=f"cropmerge-enrich-{job_id}",
                ).start()
        except Exception as exc:
            self._fail_job(job_id, exc)

    def _analysis_cache_key(self, upload: "UploadRecord", options: dict[str, Any], cfg: dict) -> str | None:
        """None means "not cacheable" (e.g. the upload was never CAS-registered)
        -- never a cache miss with a fabricated key."""
        if not upload.blob_sha256:
            return None
        return compute_cache_key(
            source_sha256=upload.blob_sha256,
            cfg=cfg,
            sample_fps=float(options["sample_fps"]),
            max_frames=int(options["max_frames"]) if options["max_frames"] is not None else MAX_FRAMES_CEILING,
            skip_dino=bool(options["skip_dino"]),
            segmentation_backend=str(options["segmentation_backend"]),
            dino_backend=str(options["dino_backend"]),
            analysis_version=ANALYSIS_VERSION,
        )

    def _try_analysis_cache_hit(
        self, upload: "UploadRecord", options: dict[str, Any], cfg: dict
    ) -> dict[str, Any] | None:
        """Phase 17: skip re-running the heavy pipeline when an identical
        source has already been analyzed under an identical configuration.
        Gated off by default (CROPMERGE_ANALYSIS_CACHE_ENABLED) -- a wrong
        hit would silently serve a stale result, unlike the other additive
        storage features in this campaign. Returns the reused report dict
        (with its ORIGINAL runId intentionally preserved, not job_id — see
        _run()'s owns_own_artifacts guard) or None on any miss/failure.
        Never raises: caching must never be why a real analysis fails.
        """
        if str(os.environ.get("CROPMERGE_ANALYSIS_CACHE_ENABLED", "")).lower() not in ("1", "true", "yes"):
            return None
        try:
            cache_key = self._analysis_cache_key(upload, options, cfg)
            if cache_key is None:
                return None
            entry = self.analysis_cache.get(cache_key)
            if entry is None:
                return None
            cached_job = self.jobs.get(entry.run_id)
            if not cached_job or cached_job["status"] != "completed" or not cached_job["report"]:
                return None
            if not _safe_artifact_path(entry.run_id, "results.json").is_file():
                return None  # cached job's artifacts were removed/never wrote — do not trust the DB row alone
            log.info("Analysis cache HIT: reusing run=%s for a new request", entry.run_id)
            return json.loads(json.dumps(cached_job["report"]))  # deep copy — never share a mutable dict
        except Exception:
            log.exception("Analysis cache lookup failed (non-fatal, running the pipeline normally)")
            return None

    def _maybe_store_analysis_cache_entry(
        self, upload: "UploadRecord", options: dict[str, Any], cfg: dict, job_id: str
    ) -> None:
        if str(os.environ.get("CROPMERGE_ANALYSIS_CACHE_ENABLED", "")).lower() not in ("1", "true", "yes"):
            return
        try:
            cache_key = self._analysis_cache_key(upload, options, cfg)
            if cache_key is not None:
                self.analysis_cache.put(cache_key, job_id)
        except Exception:
            log.exception("Analysis cache write failed for job=%s (non-fatal, job itself succeeded)", job_id)

    def _enrich_after_completion(self, job_id: str, report: dict[str, Any]) -> None:
        """Runs on a background thread (GPU1) after the job is already
        'completed'. Patches richer per-zone descriptions into the stored
        report + results.json — never blocks the job's own completion, and
        never touches job status (a slow/failed LLM must not turn a
        successful analysis into a failure)."""
        try:
            _enrich_zones_with_llm(report)
            self.jobs.update(job_id, report_json=json.dumps(report))
            safe_report = _public_report(report)
            _safe_artifact_path(job_id, "results.json").write_text(json.dumps(safe_report, indent=2), encoding="utf-8")
        except Exception:
            log.exception("Background LLM enrichment failed for job=%s", job_id)
        finally:
            # Unload even on failure — a shared-GPU host must not keep the
            # model resident just because this batch errored partway through.
            unload_llm()

    def _fail_job(self, job_id: str, exc: Exception) -> None:
        self.jobs.update(
            job_id,
            status="failed",
            progress=0.0,
            stage="failed",
            message="Analysis failed",
            error=str(exc),
        )


_job_manager: JobManager | None = None
_job_manager_key: tuple[Path, Path, Path] | None = None


def _manager() -> JobManager:
    global _job_manager, _job_manager_key
    cfg = settings()
    key = (cfg.upload_dir, cfg.output_dir, cfg.job_database_path)
    if _job_manager is None or _job_manager_key != key:
        _job_manager = JobManager(cfg)
        _job_manager_key = key
    return _job_manager


_storage_manifest: Manifest | None = None
_storage_manifest_key: Path | None = None


def _storage() -> Manifest:
    """Lazily-initialized, process-wide content-addressed storage manifest
    (see cropmerge/storage/). Rooted under settings().data_dir so it lives
    alongside the existing uploads/db directories. Registration failures
    here must never fail an upload — see _register_upload_blob()."""
    global _storage_manifest, _storage_manifest_key
    cfg = settings()
    key = cfg.data_dir
    if _storage_manifest is None or _storage_manifest_key != key:
        blob_store = BlobStore(cfg.data_dir / "blobs")
        _storage_manifest = Manifest(cfg.data_dir / "db" / "storage-manifest.sqlite", blob_store=blob_store)
        _storage_manifest_key = key
    return _storage_manifest


def _collection_record_path(collection_id: str) -> Path:
    return settings().data_dir / "collections" / f"{collection_id}.json"


def _collection_images_payload(result: CollectionIngestResult) -> dict[str, Any]:
    return {
        "collectionId": result.collection_id,
        "images": [
            {
                "id": img.id,
                "archivePath": img.archive_path,
                "relativeGroup": img.relative_group,
                "sourceSha256": img.source_sha256,
                "sizeBytes": img.size_bytes,
                "status": img.status,
                "duplicateOf": img.duplicate_of,
                "error": img.error,
            }
            for img in result.images
        ],
        "rejectedMembers": [{"name": r.name, "reason": r.reason} for r in result.rejected],
        "createdAt": _now(),
    }


def _write_collection_record(result: CollectionIngestResult) -> None:
    path = _collection_record_path(result.collection_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _collection_images_payload(result)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(payload, indent=2))
        temp = Path(handle.name)
    temp.replace(path)


def _read_collection_record(collection_id: str) -> dict[str, Any]:
    path = _collection_record_path(collection_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Collection not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Collection metadata is unreadable") from exc


def _public_collection(source: "CollectionIngestResult | dict[str, Any]") -> dict[str, Any]:
    """Accepts either a freshly created CollectionIngestResult or the
    persisted dict re-read from disk and emits the same camelCase API shape
    (a summary computed on top of whichever payload shape was given)."""
    payload = _collection_images_payload(source) if isinstance(source, CollectionIngestResult) else source
    images = payload["images"]
    rejected = payload["rejectedMembers"]
    groups = sorted({img["relativeGroup"] for img in images if img["relativeGroup"]})
    return {
        "collectionId": payload["collectionId"],
        "images": images,
        "rejectedMembers": rejected,
        "createdAt": payload.get("createdAt"),
        "summary": {
            "totalImages": len(images),
            "validImages": sum(1 for img in images if img["status"] == "valid"),
            "duplicateImages": sum(1 for img in images if img["status"] == "duplicate"),
            "failedImages": sum(1 for img in images if img["status"] == "failed"),
            "rejectedMemberCount": len(rejected),
            "groups": groups,
        },
    }


async def _ingest_collection_upload(file: UploadFile) -> CollectionIngestResult:
    """Stream the uploaded ZIP to a bounded temp file (never fully into
    memory) and hand it to ingest_zip_collection() -- see
    cropmerge/storage/zip_collection.py for why streaming-to-disk matters
    for a potentially large, untrusted archive. The temp file is removed
    once ingestion completes; the archive's actual content now lives in the
    content-addressed store, keyed by the collection's own manifest
    entries, not by this temp file.
    """
    limits = ZipLimits()
    collection_id = uuid.uuid4().hex[:12]
    tmp_dir = settings().data_dir / "tmp" / "collection-uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{collection_id}.zip"
    try:
        written = 0
        with tmp_path.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limits.max_zip_bytes:
                    raise HTTPException(status_code=413, detail=f"Archive exceeds {limits.max_zip_bytes} bytes")
                handle.write(chunk)
        return ingest_zip_collection(tmp_path, _storage(), collection_id=collection_id, limits=limits)
    finally:
        tmp_path.unlink(missing_ok=True)


def _collection_analysis_record_path(collection_id: str) -> Path:
    return settings().data_dir / "collections" / f"{collection_id}-analysis.json"


def _materialize_collection_image(collection_id: str, image: dict[str, Any]) -> Path:
    """Resolve one collection image's CAS blob to a real, decoder-safe file
    on disk. Raises (never silently substitutes another file) if the blob
    is missing -- a collection image's only-ever representation is its CAS
    blob, there is no separate "original path" fallback the way an upload's
    completed_path is (see materialize_blob()'s docstring)."""
    entry = _storage().get(f"collection:{collection_id}:image:{image['id']}")
    if entry is None:
        raise FileNotFoundError(f"no CAS manifest entry for collection image {image['id']}")
    suffix = Path(image["archivePath"]).suffix.lower()
    work_dir = settings().data_dir / "tmp" / "collection-materialized" / collection_id
    return materialize_blob(_storage().blob_store, entry.sha256, work_dir, suffix)


def _run_collection_analysis(collection_id: str, body: "CollectionAnalysisRequest") -> dict[str, Any]:
    record = _read_collection_record(collection_id)
    valid_images = [img for img in record["images"] if img["status"] == "valid"]

    inputs: list[CollectionImageInput] = []
    materialize_failures: list[dict[str, Any]] = []
    for img in valid_images:
        try:
            path = _materialize_collection_image(collection_id, img)
            inputs.append(CollectionImageInput(image_id=img["id"], relative_group=img["relativeGroup"], path=path))
        except Exception as exc:
            log.exception("Failed to materialize collection image %s/%s", collection_id, img["id"])
            materialize_failures.append({"imageId": img["id"], "error": str(exc)})

    output_root = settings().output_dir / "collections" / collection_id
    result = analyze_collection_images(
        collection_id,
        inputs,
        output_root,
        load_config(),
        segmentation_backend=body.segmentation_backend,
        dino_backend=body.dino_backend,
    )

    payload = {
        "collectionId": collection_id,
        "analyzedAt": _now(),
        "imageResults": [
            {
                "imageId": r.image_id,
                "relativeGroup": r.relative_group,
                "status": r.status,
                "error": r.error,
                "fieldDetected": r.field_detected,
                "cropCoverage": r.crop_coverage,
                "bareSoilFraction": r.bare_soil_fraction,
            }
            for r in result.image_results
        ]
        + [
            {
                "imageId": f["imageId"], "relativeGroup": None, "status": "failed",
                "error": f["error"], "fieldDetected": None, "cropCoverage": None, "bareSoilFraction": None,
            }
            for f in materialize_failures
        ],
        "groupStatistics": [
            {
                "group": g.group,
                "imageCount": g.image_count,
                "cropCoverageMedian": g.crop_coverage_median,
                "cropCoverageIqr": list(g.crop_coverage_iqr) if g.crop_coverage_iqr else None,
                "bareSoilMedian": g.bare_soil_median,
                "bareSoilIqr": list(g.bare_soil_iqr) if g.bare_soil_iqr else None,
            }
            for g in result.group_statistics
        ],
        "summary": {
            "totalValidImages": len(valid_images),
            "analyzedCount": result.analyzed_count,
            "failedCount": result.failed_count + len(materialize_failures),
        },
    }

    path = _collection_analysis_record_path(collection_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(payload, indent=2))
        temp = Path(handle.name)
    temp.replace(path)
    return payload


def _read_collection_analysis_record(collection_id: str) -> dict[str, Any]:
    path = _collection_analysis_record_path(collection_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Collection has not been analyzed yet")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Collection analysis metadata is unreadable") from exc


_JPEG_SUFFIXES = {".jpg", ".jpeg"}


def _jxl_min_savings_ratio() -> float:
    return float(os.environ.get("CROPMERGE_JXL_MIN_SAVINGS_RATIO", "0.05"))


def _register_upload_blob(record: "UploadRecord") -> "UploadRecord":
    """Hash the just-completed upload into the content-addressed store and
    record a logical reference for it. Purely additive bookkeeping for this
    integration phase: record.completed_path keeps pointing at the original
    plain file (unchanged read path for analysis/materialization) — this
    only gives real dedup-bytes visibility (see docs/STORAGE_BASELINE.md)
    and a provenance hash. A failure here must never fail the upload
    itself, so any exception is logged and swallowed.

    For JPEG uploads, also attempts a verified reversible JPEG XL archive
    (see cropmerge/storage/jxl_archive.py) as an ADDITIONAL, separately
    content-addressed blob — never a replacement for the original file on
    disk. This stays additive-only in this integration round: source
    replacement (deleting the JPEG once a verified reconstructible JXL
    exists) is deliberately deferred, per the storage campaign's own
    "lossy/destructive source replacement stays off until benchmarked"
    default — reversible archival is lossless and verified, but the actual
    deletion step still deserves its own dedicated review of every code
    path that reads completed_path directly.
    """
    if not record.completed_path:
        return record
    try:
        blob_ref = _storage().blob_store.put(Path(record.completed_path))
        _storage().add(f"upload:{record.id}", blob_ref, storage_class=StorageClass.SCIENTIFIC_SOURCE.value)
        record.blob_sha256 = blob_ref.sha256
    except Exception:
        log.exception("Storage blob registration failed for upload=%s (upload itself still succeeded)", record.id)
        return record

    if record.suffix.lower() in _JPEG_SUFFIXES:
        _try_archive_jpeg_reversible(record)
    return record


def _try_archive_jpeg_reversible(record: "UploadRecord") -> None:
    from cropmerge.storage.jxl_archive import archive_jpeg_reversible, jxl_available

    if not jxl_available():
        return
    jxl_tmp_path: Path | None = None
    try:
        jxl_tmp_dir = settings().data_dir / "tmp" / "jxl-archive"
        jxl_tmp_dir.mkdir(parents=True, exist_ok=True)
        jxl_tmp_path = jxl_tmp_dir / f"{record.id}.jxl"
        result = archive_jpeg_reversible(Path(record.completed_path), jxl_tmp_path)
        if not result.verified:
            log.info("JXL archival skipped for upload=%s: %s", record.id, result.reason)
            return
        if result.savings_ratio < _jxl_min_savings_ratio():
            log.info(
                "JXL archival rejected for upload=%s: savings %.1f%% below threshold",
                record.id, result.savings_ratio * 100,
            )
            return
        jxl_blob_ref = _storage().blob_store.put(jxl_tmp_path)
        _storage().add(
            f"upload:{record.id}:jxl_archive", jxl_blob_ref, storage_class=StorageClass.REVERSIBLE_SOURCE.value,
        )
        log.info(
            "JXL reversible archive verified+stored for upload=%s: %.1f%% savings",
            record.id, result.savings_ratio * 100,
        )
    except Exception:
        log.exception("JXL archival attempt failed for upload=%s (non-fatal, original JPEG untouched)", record.id)
    finally:
        if jxl_tmp_path is not None:
            jxl_tmp_path.unlink(missing_ok=True)


def _materialize_upload_source(upload: "UploadRecord") -> Path:
    """Phase 9 of the storage integration campaign: analysis reads through
    the storage abstraction instead of reaching for completed_path
    directly. In this integration round the only registered source class is
    SCIENTIFIC_SOURCE (a byte-identical CAS copy of the exact same upload —
    see _register_upload_blob), so this changes WHERE the bytes are read
    from, never WHAT bytes are read; a materialization miss/failure falls
    back to upload.completed_path unconditionally (materialize_source()
    itself never raises). This is the one seam a later storage
    representation change (e.g. serving a verified reversible JXL archive
    instead) would need to touch, not every analysis call site.
    """
    fallback = Path(upload.completed_path)
    try:
        work_dir = settings().data_dir / "tmp" / "materialized-sources"
        return materialize_source(_storage(), f"upload:{upload.id}", fallback, work_dir, suffix=upload.suffix)
    except Exception:
        log.exception("Source materialization failed for upload=%s, using completed_path directly", upload.id)
        return fallback


def _register_run_artifacts_in_cas(job_id: str, report_model: FieldTriageReport) -> None:
    """Phase 10 of the storage integration campaign: register only the
    STABLE final artifacts (results.json, metrics.json, the annotated
    review video, the heatmap preview) into the same content-addressed
    store uploads use — never per-frame/debug intermediates (frames_dir,
    overlays_dir stay ordinary temp/output files, unregistered, per the
    campaign's "start with source uploads, then only stable final
    artifacts" instruction). Non-fatal: a completed analysis must never be
    put at risk by a storage bookkeeping failure.
    """
    candidates: list[tuple[str, str | None, StorageClass]] = [
        ("results.json", str(_safe_artifact_path(job_id, "results.json")), StorageClass.DERIVED_ANALYSIS),
        ("metrics.json", report_model.artifacts.metrics_json, StorageClass.DERIVED_ANALYSIS),
        ("annotated_video", report_model.artifacts.annotated_video, StorageClass.DERIVED_PREVIEW),
        ("heatmap.png", report_model.artifacts.heatmap_png, StorageClass.DERIVED_PREVIEW),
    ]
    for name, path_str, storage_class in candidates:
        if not path_str:
            continue
        path = Path(path_str)
        if not path.is_file():
            continue
        try:
            blob_ref = _storage().blob_store.put(path)
            _storage().add(f"run:{job_id}:{name}", blob_ref, storage_class=storage_class.value)
        except Exception:
            log.exception("Artifact CAS registration failed for run=%s artifact=%s (artifact itself is unaffected)", job_id, name)


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _ffmpeg_ok() -> bool:
    return shutil.which("ffmpeg") is not None or shutil.which("ffprobe") is not None


def _opencv_ok() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _gpu_memory_ok() -> tuple[bool, str | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return True, None
        free, total = torch.cuda.mem_get_info()
        min_free = int(os.environ.get("VISION_GPU_MIN_FREE_BYTES", str(512 * 1024 * 1024)))
        if free < min_free:
            used_pct = round(100 * (1 - free / total)) if total else 100
            return False, f"GPU memory is nearly full ({used_pct}% used). Try again shortly."
        return True, None
    except Exception:
        return True, None


def _release_gpu_cache() -> None:
    """Called after every job (recorded-mode SAM2/DINOv2 inference is not
    kept resident between jobs — each job creates fresh segmenter/embedder
    instances). PyTorch's caching allocator keeps freed tensor memory in a
    pool for reuse rather than returning it to the OS, so nvidia-smi shows
    it as still "used" until this runs — this is what nvidia-smi-measured
    idle VRAM is actually seeing, not a real leak.
    """
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        log.exception("GPU cache release failed (non-fatal)")


def _max_queue_depth() -> int:
    return max(1, int(os.environ.get("VISION_MAX_QUEUE_DEPTH", "3")))


def _tcp_reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _mediamtx_reachable() -> bool:
    return _tcp_reachable(os.environ.get("MEDIAMTX_HOST", "127.0.0.1"), int(os.environ.get("MEDIAMTX_RTSP_PORT", "8554")))


def _mqtt_reachable() -> bool:
    return _tcp_reachable(os.environ.get("MQTT_BROKER_HOST", "127.0.0.1"), int(os.environ.get("MQTT_BROKER_PORT", "1883")))


def create_app() -> FastAPI:
    cfg = settings()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(
        title="CropMerge Vision Engine",
        version=__version__,
        description="Direct browser-facing CV inference service for CropMerge Field Triage.",
    )
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Range"],
        expose_headers=["Accept-Ranges", "Content-Range", "Content-Length"],
    )
    app.include_router(create_live_router())

    @app.get("/vision/health")
    def health(response: Response):
        _no_store(response)
        seg = os.environ.get("CROP_MERGE_SEGMENTATION_BACKEND", "heuristic")
        dino = os.environ.get("CROP_MERGE_DINO_BACKEND", "heuristic")
        queue = _manager().queue_status()
        gpu_ok = bool(queue["gpuAvailable"])
        return {
            "status": "ok" if _opencv_ok() and gpu_ok else "degraded",
            "service": "cropmerge-vision",
            "version": __version__,
            "device": _device(),
            "segmentationBackend": seg,
            "dinoBackend": dino,
            "samAvailable": sam3_available(),
            "sam2Available": sam2_available(),
            "dinoAvailable": dinov3_available(),
            "dinov2Available": dinov2_available(),
            "torchAvailable": _torch_ok(),
            "opencvAvailable": _opencv_ok(),
            "ffmpegAvailable": _ffmpeg_ok(),
            "gpuBusy": queue["gpuBusy"],
            "gpuAvailable": queue["gpuAvailable"],
            "gpuMessage": queue["gpuMessage"],
            "activeJobId": queue["activeJobId"],
            "queueDepth": queue["queueDepth"],
            "maxQueueDepth": queue["maxQueueDepth"],
            "mediamtxReachable": _mediamtx_reachable(),
            "mqttConnected": _mqtt_reachable(),
            "explanation": llm_health_status(),
        }

    @app.get("/vision/runs")
    def runs(response: Response, _: SessionClaims, limit: int = 20):
        _no_store(response)
        return {"runs": list_runs(limit=min(max(limit, 1), 100))}

    @app.post("/vision/uploads/init")
    def init_upload(body: UploadInitRequest, response: Response, _: SessionClaims):
        _no_store(response)
        manager = _manager()
        record = manager.uploads.create(body.filename, body.size, chunk_size=manager.cfg.chunk_bytes)
        return {
            "uploadId": record.id,
            "chunkSize": record.chunk_size,
            "totalChunks": record.total_chunks,
            "smallUploadThresholdBytes": manager.cfg.small_upload_threshold_bytes,
        }

    @app.put("/vision/uploads/{upload_id}/chunks/{index}")
    async def upload_chunk(upload_id: str, index: int, request: Request, response: Response, _: SessionClaims):
        _no_store(response)
        _valid_upload_id(upload_id)
        manager = _manager()
        record = manager.uploads.get(upload_id)
        if index < 0 or index >= record.total_chunks:
            raise HTTPException(status_code=400, detail="Chunk index is outside the upload range")
        expected = min(record.chunk_size, record.size - index * record.chunk_size)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) != expected:
            raise HTTPException(status_code=400, detail=f"Chunk Content-Length must be {expected} bytes")
        chunks: list[bytes] = []
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > expected:
                raise HTTPException(status_code=413, detail="Chunk exceeds its configured size")
            chunks.append(chunk)
        manager.uploads.write_chunk(upload_id, index, b"".join(chunks))
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @app.post("/vision/uploads/{upload_id}/complete")
    def complete_upload(upload_id: str, response: Response, _: SessionClaims):
        _no_store(response)
        record = _manager().uploads.complete(upload_id)
        return {"uploadId": record.id, "status": record.status, "size": record.size}

    @app.post("/vision/uploads")
    def upload_small(response: Response, _: SessionClaims, file: UploadFile = File(...)):
        _no_store(response)
        record = _manager().uploads.save_small(file)
        return {"uploadId": record.id, "status": record.status, "size": record.size}

    @app.delete("/vision/uploads/{upload_id}")
    def cancel_upload(upload_id: str, response: Response, _: SessionClaims):
        _no_store(response)
        _manager().uploads.cancel(upload_id)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @app.post("/vision/collections", status_code=201)
    async def create_collection(response: Response, _: SessionClaims, file: UploadFile = File(...)):
        _no_store(response)
        result = await _ingest_collection_upload(file)
        if result.archive_rejected is not None:
            raise HTTPException(status_code=400, detail=result.archive_rejected)
        _write_collection_record(result)
        return _public_collection(result)

    @app.get("/vision/collections/{collection_id}")
    def get_collection(collection_id: str, _: SessionClaims):
        return _public_collection(_read_collection_record(_valid_collection_id(collection_id)))

    @app.post("/vision/collections/{collection_id}/analyze")
    def analyze_collection(collection_id: str, body: CollectionAnalysisRequest, _: SessionClaims):
        return _run_collection_analysis(_valid_collection_id(collection_id), body)

    @app.get("/vision/collections/{collection_id}/analysis")
    def get_collection_analysis(collection_id: str, _: SessionClaims):
        return _read_collection_analysis_record(_valid_collection_id(collection_id))

    @app.post("/vision/analyses", status_code=202)
    def create_analysis(body: AnalysisRequest, request: Request, response: Response, _: SessionClaims):
        _no_store(response)
        manager = _manager()
        upload = manager.uploads.get(body.upload_id)
        if upload.status != "completed":
            raise HTTPException(status_code=409, detail="Upload must be completed before analysis starts")
        return _public_job(manager.submit(upload, body), request)

    @app.get("/vision/analyses")
    def list_analyses(request: Request, response: Response, _: SessionClaims, limit: int = 20):
        _no_store(response)
        jobs = _manager().jobs.list(limit=min(max(limit, 1), 100))
        return {"analyses": [_public_job(job, request) for job in jobs]}

    @app.get("/vision/analyses/{job_id}")
    def get_analysis(job_id: str, request: Request, response: Response, _: SessionClaims):
        _no_store(response)
        return _public_job(_manager().jobs.get(_valid_job_id(job_id)), request)

    @app.post("/vision/analyses/{job_id}/cancel")
    def cancel_analysis(job_id: str, request: Request, response: Response, _: SessionClaims):
        _no_store(response)
        manager = _manager()
        job = manager.jobs.get(_valid_job_id(job_id))
        if job["status"] != "queued":
            raise HTTPException(status_code=409, detail="Only queued analyses can be cancelled")
        return _public_job(
            manager.jobs.update(job_id, status="cancelled", stage="cancelled", message="Analysis cancelled"),
            request,
        )

    @app.get("/vision/artifacts/{run_id}/{name:path}")
    def get_artifact(run_id: str, name: str, token: str):
        _verify_artifact_token(token, run_id, name)
        path = _safe_artifact_path(run_id, name)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        # Starlette's FileResponse handles HTTP Range requests for video seeking.
        return FileResponse(path, headers={"Cache-Control": "private, no-store", "Accept-Ranges": "bytes"})

    @app.post("/vision/analyze")
    async def legacy_analyze(
        response: Response,
        _: SessionClaims,
        file: UploadFile = File(...),
        sample_fps: float = Form(2.0),
        max_frames: int = Form(120),
        skip_dino: bool = Form(False),
        segmentation_backend: str = Form("heuristic"),
        dino_backend: str = Form("heuristic"),
    ):
        """Explicit opt-in only; deployed browsers must use uploads + async jobs."""
        _no_store(response)
        if os.environ.get("VISION_ENABLE_LEGACY_SYNC_ANALYZE", "false").lower() != "true":
            raise HTTPException(status_code=410, detail="Use /vision/uploads and /vision/analyses for asynchronous analysis")
        suffix = _safe_extension(file.filename or "upload.mp4")
        temporary = Path(tempfile.mkdtemp(prefix="cropmerge_"))
        try:
            input_path = temporary / f"input{suffix}"
            input_path.write_bytes(await file.read())
            report = FieldTriageProcessor(load_config()).process(
                input_path,
                settings().output_dir,
                sample_fps=sample_fps,
                max_frames=max_frames if max_frames > 0 else None,
                skip_dino=skip_dino,
                segmentation_backend=segmentation_backend,
                dino_backend=dino_backend,
            )
            return _public_report(report.to_camel_dict())
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    @app.post("/vision/segment")
    async def segment_only(response: Response, _: SessionClaims, file: UploadFile = File(...)):
        """Lightweight single-image segmentation for authenticated debugging."""
        _no_store(response)
        import cv2
        import numpy as np

        from cropmerge.segmentation import create_segmenter

        data = await file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        seg = create_segmenter(os.environ.get("CROP_MERGE_SEGMENTATION_BACKEND", "heuristic"), load_config())
        result = seg.segment_image(bgr, 0, 0.0)
        fractions: dict[str, float] = {}
        if result.label_map is not None:
            from cropmerge.segmentation.postprocess import class_fractions

            fractions = {
                key: value for key, value in class_fractions(result.label_map).items() if not key.startswith("_")
            }
        return {
            "backend": result.backend,
            "isFallback": result.is_fallback,
            "fieldDetected": bool(result.field_mask is not None and float(np.mean(result.field_mask)) > 0.05),
            "classCoverage": fractions,
        }

    return app


app = create_app()
