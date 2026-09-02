"""Vision run store — SQLite default, Postgres if VISION_DATABASE_URL is postgresql."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("cropmerge.db")

# cropmerge/db.py → vision → apps → monorepo root
_ROOT = Path(__file__).resolve().parents[3]


def _default_sqlite_path() -> Path:
    env = os.environ.get("VISION_DATABASE_URL", "")
    if env.startswith("sqlite:///"):
        return Path(env.replace("sqlite:///", ""))
    db_dir = Path(os.environ.get("DATABASE_DIR", _ROOT / "data" / "db"))
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "vision.sqlite"


def _init_sqlite(conn: sqlite3.Connection) -> None:
    sql_path = _ROOT / "tooling" / "db" / "sqlite-init-vision.sql"
    if sql_path.is_file():
        conn.executescript(sql_path.read_text(encoding="utf-8"))
    else:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vision_runs (
              run_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              source_filename TEXT,
              status TEXT NOT NULL DEFAULT 'completed',
              segmentation_backend TEXT,
              dino_backend TEXT,
              used_fallback INTEGER,
              frames_sampled INTEGER,
              frames_usable INTEGER,
              zone_count INTEGER,
              field_detected INTEGER,
              total_runtime_sec REAL,
              stage_latency_json TEXT,
              results_path TEXT,
              metrics_json TEXT,
              report_json TEXT
            );
            """
        )
    conn.commit()


def connect_sqlite(path: Path | None = None) -> sqlite3.Connection:
    path = path or _default_sqlite_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_sqlite(conn)
    return conn


def record_run(report: Any, status: str = "completed") -> None:
    """Persist a FieldTriageReport summary. Never raises into the pipeline."""
    try:
        camel = report.to_camel_dict() if hasattr(report, "to_camel_dict") else report
        run_id = camel.get("runId") or getattr(report, "run_id", None)
        if not run_id:
            return
        analysis = camel.get("analysis") or {}
        field = camel.get("field") or {}
        source = camel.get("source") or {}
        artifacts = camel.get("artifacts") or {}
        zones = camel.get("inspectionZones") or []

        url = os.environ.get("VISION_DATABASE_URL", "")
        if url.startswith("postgresql://") or url.startswith("postgres://"):
            _record_pg(url, camel, status)
            return

        conn = connect_sqlite()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO vision_runs (
                  run_id, created_at, source_filename, status,
                  segmentation_backend, dino_backend, used_fallback,
                  frames_sampled, frames_usable, zone_count, field_detected,
                  total_runtime_sec, stage_latency_json, results_path,
                  metrics_json, report_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    camel.get("createdAt") or datetime.now(timezone.utc).isoformat(),
                    source.get("filename"),
                    status,
                    analysis.get("segmentationBackend"),
                    analysis.get("dinoBackend"),
                    1 if analysis.get("usedFallback") else 0,
                    analysis.get("framesSampled"),
                    analysis.get("framesUsable"),
                    len(zones),
                    1 if field.get("detected") else 0,
                    analysis.get("totalRuntimeSec"),
                    json.dumps(analysis.get("stageLatencySec") or {}),
                    artifacts.get("resultsJson"),
                    json.dumps(
                        {
                            "zoneCount": len(zones),
                            "stageLatencySec": analysis.get("stageLatencySec"),
                        }
                    ),
                    json.dumps(camel),
                ),
            )
            conn.commit()
            log.info("DB recorded run_id=%s → %s", run_id, _default_sqlite_path())
        finally:
            conn.close()
    except Exception as e:
        log.warning("DB record_run failed: %s", e)


def _record_pg(url: str, camel: dict, status: str) -> None:
    try:
        import psycopg

        analysis = camel.get("analysis") or {}
        field = camel.get("field") or {}
        source = camel.get("source") or {}
        artifacts = camel.get("artifacts") or {}
        zones = camel.get("inspectionZones") or []
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO vision_runs (
                      run_id, created_at, source_filename, status,
                      segmentation_backend, dino_backend, used_fallback,
                      frames_sampled, frames_usable, zone_count, field_detected,
                      total_runtime_sec, stage_latency_json, results_path,
                      metrics_json, report_json
                    ) VALUES (
                      %(run_id)s, %(created_at)s, %(source_filename)s, %(status)s,
                      %(seg)s, %(dino)s, %(fb)s,
                      %(fs)s, %(fu)s, %(zc)s, %(fd)s,
                      %(rt)s, %(lat)s::jsonb, %(rp)s,
                      %(mj)s::jsonb, %(rj)s::jsonb
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                      status = EXCLUDED.status,
                      report_json = EXCLUDED.report_json
                    """,
                    {
                        "run_id": camel.get("runId"),
                        "created_at": camel.get("createdAt"),
                        "source_filename": source.get("filename"),
                        "status": status,
                        "seg": analysis.get("segmentationBackend"),
                        "dino": analysis.get("dinoBackend"),
                        "fb": bool(analysis.get("usedFallback")),
                        "fs": analysis.get("framesSampled"),
                        "fu": analysis.get("framesUsable"),
                        "zc": len(zones),
                        "fd": bool(field.get("detected")),
                        "rt": analysis.get("totalRuntimeSec"),
                        "lat": json.dumps(analysis.get("stageLatencySec") or {}),
                        "rp": artifacts.get("resultsJson"),
                        "mj": json.dumps({"zoneCount": len(zones)}),
                        "rj": json.dumps(camel),
                    },
                )
            conn.commit()
    except ImportError:
        log.warning("psycopg not installed; skip postgres vision write")


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    conn = connect_sqlite()
    try:
        rows = conn.execute(
            "SELECT run_id, created_at, source_filename, status, zone_count, field_detected, total_runtime_sec "
            "FROM vision_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
