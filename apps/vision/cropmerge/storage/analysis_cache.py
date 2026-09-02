"""Phase 17 of the storage integration campaign: skip re-running the heavy
CV pipeline (SAM2/DINOv2/structural/FarmTech) when an identical source has
already been analyzed under an identical configuration.

Cache key = hash(source_sha256, sample_fps, max_frames, skip_dino,
segmentation_backend, dino_backend, ANALYSIS_VERSION, config content hash).
Combining an explicit ANALYSIS_VERSION constant (bumped manually when
processor.py's algorithm changes in a way not captured by config) with a
hash of the *actual loaded config dict* (so any configs/*.yaml edit
automatically invalidates old cache entries, with no reliance on someone
remembering to bump a version string for that case) is the "configVersion /
analysisVersion" pairing the campaign asks for.

Deliberately does NOT cache raw tensors, DINO embeddings, or any
intermediate model output -- only maps a cache key to the run_id of a
previously-completed job whose OWN on-disk artifacts (results.json,
metrics.json, annotated video, heatmap) are the actual cached payload. A
cache hit reuses those files (by hardlink, same pattern as
cropmerge/storage/materialize.py) rather than duplicating them.

Gated off by default (CROPMERGE_ANALYSIS_CACHE_ENABLED) -- unlike upload
dedup/JXL archival, a wrong cache hit would silently serve a stale result
for what looks like a fresh analysis, so this stays opt-in until it has
seen real production traffic.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def compute_cache_key(
    *,
    source_sha256: str,
    cfg: dict,
    sample_fps: float,
    max_frames: int | None,
    skip_dino: bool,
    segmentation_backend: str,
    dino_backend: str,
    analysis_version: str,
) -> str:
    """Deterministic cache key. `cfg` is the actual loaded config dict
    (cropmerge.config.load_config()'s result), serialized canonically
    (sort_keys) so key order never causes a spurious miss."""
    payload = {
        "sourceSha256": source_sha256,
        "sampleFps": sample_fps,
        "maxFrames": max_frames,
        "skipDino": skip_dino,
        "segmentationBackend": segmentation_backend,
        "dinoBackend": dino_backend,
        "analysisVersion": analysis_version,
        "configHash": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def config_content_hash(cfg: dict) -> str:
    """Short, stable hash of the loaded config content -- exposed
    separately so callers can attach it to a report as `configVersion`
    without needing the full cache-key payload (which also folds in
    per-request options like sample_fps)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    run_id: str
    created_at: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_cache (
    cache_key   TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


class AnalysisCache:
    """Crash-safe SQLite-backed cache_key -> run_id mapping."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), isolation_level=None, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def get(self, cache_key: str) -> CacheEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM analysis_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        return CacheEntry(cache_key=row["cache_key"], run_id=row["run_id"], created_at=row["created_at"])

    def put(self, cache_key: str, run_id: str) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO analysis_cache (cache_key, run_id, created_at) VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET run_id=excluded.run_id, created_at=excluded.created_at
                """,
                (cache_key, run_id, created_at),
            )

    def close(self) -> None:
        self._conn.close()
