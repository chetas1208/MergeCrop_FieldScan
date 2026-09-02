from pathlib import Path

from cropmerge.live.schemas import LiveSessionStatus
from cropmerge.live.session_store import LiveSessionStore
from cropmerge.pipeline.schemas import (
    InspectionZone,
    RelativeLocation,
    ReviewPriority,
    ZoneEvidence,
)
from cropmerge.telemetry.schemas import DroneTelemetry, TelemetryAssociation, TelemetryAvailability


def _store(tmp_path: Path, monkeypatch) -> LiveSessionStore:
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path / "db"))
    monkeypatch.delenv("VISION_DATABASE_URL", raising=False)
    return LiveSessionStore()


def _zone(zone_id="z1") -> InspectionZone:
    return InspectionZone(
        id=zone_id,
        review_priority=ReviewPriority.HIGH,
        anomaly_score=0.8,
        persistence_score=0.9,
        first_seen_ms=0,
        last_seen_ms=1000,
        first_seen_sec=0.0,
        last_seen_sec=1.0,
        frames_seen=5,
        relative_location=RelativeLocation.C,
        centroid_norm={"x": 0.5, "y": 0.5},
        bbox_norm={"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2},
        evidence=ZoneEvidence(),
        reasons=["test"],
        recommendation="Review",
    )


def test_create_and_get_session(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    session = store.create(stream_path="cropmerge/live", sample_fps=2.0, simulated=True, whep_url="http://x")
    fetched = store.get(session.id)
    assert fetched is not None
    assert fetched.status == LiveSessionStatus.DISCONNECTED
    assert fetched.simulated is True


def test_update_status_transitions(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    session = store.create(stream_path="cropmerge/live", sample_fps=2.0, simulated=False, whep_url="http://x")
    store.update_status(session.id, LiveSessionStatus.LIVE)
    assert store.get(session.id).status == LiveSessionStatus.LIVE
    store.update_status(session.id, LiveSessionStatus.ENDED, ended=True)
    ended = store.get(session.id)
    assert ended.status == LiveSessionStatus.ENDED
    assert ended.ended_at is not None


def test_record_and_query_nearest_telemetry(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    session = store.create(stream_path="cropmerge/live", sample_fps=2.0, simulated=False, whep_url="http://x")
    store.record_telemetry(session.id, DroneTelemetry(timestamp_ms=1000, latitude=1.0, longitude=2.0))
    store.record_telemetry(session.id, DroneTelemetry(timestamp_ms=5000, latitude=3.0, longitude=4.0))
    row = store.nearest_telemetry_row(session.id, 1100)
    assert row["timestamp_ms"] == 1000
    assert row["latitude"] == 1.0

    telemetry_list = store.list_telemetry(session.id)
    assert len(telemetry_list) == 2
    assert telemetry_list[0].timestamp_ms == 1000


def test_save_and_list_areas_never_fabricates_unavailable_telemetry(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    session = store.create(stream_path="cropmerge/live", sample_fps=2.0, simulated=True, whep_url="http://x")
    association = TelemetryAssociation(availability=TelemetryAvailability.UNAVAILABLE, telemetry=None, tolerance_ms=2000)
    saved = store.save_area(session.id, _zone(), association, note="check this")

    areas = store.list_saved_areas(session.id)
    assert len(areas) == 1
    assert areas[0].id == saved.id
    assert areas[0].telemetry_association.availability == TelemetryAvailability.UNAVAILABLE
    assert areas[0].telemetry_association.telemetry is None
    assert areas[0].note == "check this"
