import json

from cropmerge.live.schemas import (
    ErrorEvent,
    FrameAnalyzedEvent,
    LiveInspectionAreaEvent,
    LiveSessionStatus,
    StateChangeEvent,
    TelemetryEvent,
    zone_from_camel_dict,
)
from cropmerge.pipeline.schemas import (
    InspectionZone,
    RelativeLocation,
    ReviewPriority,
    ZoneEvidence,
    zone_to_camel_dict,
)
from cropmerge.telemetry.schemas import DroneTelemetry, TelemetryAssociation, TelemetryAvailability


def _zone(zone_id="z1") -> InspectionZone:
    return InspectionZone(
        id=zone_id,
        review_priority=ReviewPriority.HIGH,
        anomaly_score=0.8,
        appearance_anomaly_score=0.5,
        structural_anomaly_score=0.8,
        persistence_score=0.9,
        first_seen_ms=0,
        last_seen_ms=1000,
        first_seen_sec=0.0,
        last_seen_sec=1.0,
        frames_seen=5,
        relative_location=RelativeLocation.C,
        centroid_norm={"x": 0.5, "y": 0.5},
        bbox_norm={"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2},
        evidence=ZoneEvidence(crop_coverage_delta=0.1),
        reasons=["test reason"],
        recommendation="Review",
    )


def test_state_change_event_camel_dict():
    event = StateChangeEvent(session_id="s1", status=LiveSessionStatus.LIVE)
    out = event.to_camel_dict()
    assert out == {"type": "state_change", "sessionId": "s1", "status": "live"}
    json.dumps(out)  # must be JSON-serializable for the WS transport


def test_telemetry_event_camel_dict():
    event = TelemetryEvent(session_id="s1", telemetry=DroneTelemetry(timestamp_ms=100, latitude=1.0))
    out = event.to_camel_dict()
    assert out["type"] == "telemetry"
    assert out["telemetry"]["latitude"] == 1.0
    json.dumps(out)


def test_error_event_camel_dict():
    event = ErrorEvent(session_id="s1", message="boom")
    assert event.to_camel_dict() == {"type": "error", "sessionId": "s1", "message": "boom"}


def test_frame_analyzed_event_wraps_zones():
    association = TelemetryAssociation(availability=TelemetryAvailability.UNAVAILABLE, telemetry=None, tolerance_ms=2000)
    live_zone = LiveInspectionAreaEvent(
        **_zone().model_dump(), live_frame_timestamp_ms=5000, telemetry_association=association
    )
    event = FrameAnalyzedEvent(session_id="s1", frame_timestamp_ms=5000, zones=[live_zone])
    out = event.to_camel_dict()
    assert out["type"] == "frame_analyzed"
    assert out["zones"][0]["liveFrameTimestampMs"] == 5000
    assert out["zones"][0]["telemetryAssociation"]["availability"] == "unavailable"
    assert out["zones"][0]["id"] == "z1"
    json.dumps(out)


def test_zone_from_camel_dict_is_inverse_of_zone_to_camel_dict():
    original = _zone()
    camel = zone_to_camel_dict(original)
    restored = zone_from_camel_dict(camel)
    assert restored.id == original.id
    assert restored.review_priority == original.review_priority
    assert restored.anomaly_score == original.anomaly_score
    assert restored.bbox_norm == original.bbox_norm
    assert restored.evidence.crop_coverage_delta == original.evidence.crop_coverage_delta
    assert restored.reasons == original.reasons
