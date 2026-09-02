from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

from cropmerge.db import connect_sqlite
from cropmerge.live.schemas import LiveSession, LiveSessionStatus, SavedInspectionArea
from cropmerge.pipeline.schemas import InspectionZone
from cropmerge.telemetry.schemas import DroneTelemetry, TelemetryAssociation, TelemetryAvailability


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiveSessionStore:
    """SQLite-backed store for live sessions/telemetry/saved areas, mirroring
    the shape of api/main.py's JobStore (same lock-then-connect pattern).
    Uses the same vision.sqlite database as recorded-run history
    (cropmerge.db.connect_sqlite), whose init script already creates the
    live_* tables (tooling/db/sqlite-init-vision.sql).
    """

    _lock = threading.Lock()

    def _connect(self):
        return connect_sqlite()

    def create(self, *, stream_path: str, sample_fps: float, simulated: bool, whep_url: str) -> LiveSession:
        session_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_sessions (id, created_at, updated_at, status, stream_path, sample_fps, simulated, whep_url)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (session_id, now, now, LiveSessionStatus.DISCONNECTED.value, stream_path, sample_fps, int(simulated), whep_url),
            )
        return LiveSession(
            id=session_id, created_at=now, ended_at=None, status=LiveSessionStatus.DISCONNECTED,
            stream_path=stream_path, sample_fps=sample_fps, simulated=simulated, whep_url=whep_url,
        )

    @staticmethod
    def _row_to_session(row) -> LiveSession:
        return LiveSession(
            id=row["id"], created_at=row["created_at"], ended_at=row["ended_at"],
            status=LiveSessionStatus(row["status"]), stream_path=row["stream_path"],
            sample_fps=row["sample_fps"], simulated=bool(row["simulated"]), whep_url=row["whep_url"],
        )

    def get(self, session_id: str) -> LiveSession | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM live_sessions WHERE id = ?", (session_id,)).fetchone()
        return self._row_to_session(row) if row else None

    def list(self, limit: int = 20) -> list[LiveSession]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM live_sessions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def update_status(self, session_id: str, status: LiveSessionStatus, *, ended: bool = False) -> None:
        with self._lock, self._connect() as conn:
            if ended:
                conn.execute(
                    "UPDATE live_sessions SET status=?, updated_at=?, ended_at=? WHERE id=?",
                    (status.value, _now(), _now(), session_id),
                )
            else:
                conn.execute(
                    "UPDATE live_sessions SET status=?, updated_at=? WHERE id=?",
                    (status.value, _now(), session_id),
                )

    def record_telemetry(self, session_id: str, telemetry: DroneTelemetry) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_telemetry (
                    session_id, received_at, timestamp_ms, latitude, longitude,
                    altitude_m, heading_deg, gimbal_pitch_deg, raw_json
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    session_id, _now(), telemetry.timestamp_ms, telemetry.latitude, telemetry.longitude,
                    telemetry.altitude_m, telemetry.heading_deg, telemetry.gimbal_pitch_deg,
                    telemetry.model_dump_json(),
                ),
            )

    def nearest_telemetry_row(self, session_id: str, frame_ts_ms: int):
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT * FROM live_telemetry WHERE session_id = ?
                ORDER BY ABS(timestamp_ms - ?) ASC LIMIT 1
                """,
                (session_id, frame_ts_ms),
            ).fetchone()

    def list_telemetry(self, session_id: str) -> list[DroneTelemetry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM live_telemetry WHERE session_id = ? ORDER BY timestamp_ms ASC", (session_id,)
            ).fetchall()
        return [
            DroneTelemetry(
                timestamp_ms=r["timestamp_ms"], latitude=r["latitude"], longitude=r["longitude"],
                altitude_m=r["altitude_m"], heading_deg=r["heading_deg"], gimbal_pitch_deg=r["gimbal_pitch_deg"],
                source="dji-cloud-api",
            )
            for r in rows
        ]

    def save_area(
        self,
        session_id: str,
        zone: InspectionZone,
        association: TelemetryAssociation,
        *,
        snapshot_path: str | None = None,
        note: str | None = None,
    ) -> SavedInspectionArea:
        area_id = f"{session_id}-{uuid.uuid4().hex[:8]}"
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_saved_areas (
                    id, session_id, zone_json, telemetry_availability, telemetry_json,
                    snapshot_path, saved_at, note
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    area_id, session_id, zone.model_dump_json(), association.availability.value,
                    association.telemetry.model_dump_json() if association.telemetry else None,
                    snapshot_path, now, note,
                ),
            )
        return SavedInspectionArea(
            id=area_id, session_id=session_id, zone=zone, telemetry_association=association,
            snapshot_url=None, saved_at=now, note=note,
        )

    def list_saved_areas(self, session_id: str) -> list[SavedInspectionArea]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM live_saved_areas WHERE session_id = ? ORDER BY saved_at ASC", (session_id,)
            ).fetchall()
        out = []
        for r in rows:
            telemetry = DroneTelemetry.model_validate_json(r["telemetry_json"]) if r["telemetry_json"] else None
            association = TelemetryAssociation(
                availability=TelemetryAvailability(r["telemetry_availability"]),
                telemetry=telemetry,
                tolerance_ms=0,
            )
            out.append(
                SavedInspectionArea(
                    id=r["id"], session_id=r["session_id"],
                    zone=InspectionZone.model_validate_json(r["zone_json"]),
                    telemetry_association=association, snapshot_url=None,
                    saved_at=r["saved_at"], note=r["note"],
                )
            )
        return out
