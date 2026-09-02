from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel

from cropmerge.pipeline.schemas import InspectionZone, ZoneEvidence, zone_to_camel_dict
from cropmerge.telemetry.schemas import DroneTelemetry, TelemetryAssociation


class LiveSessionStatus(str, Enum):
    DISCONNECTED = "disconnected"
    WAITING = "waiting"
    CONNECTING = "connecting"
    LIVE = "live"
    STREAM_LOST = "stream_lost"
    RECONNECTING = "reconnecting"
    ENDED = "ended"


class LiveSession(BaseModel):
    id: str
    created_at: str
    ended_at: str | None = None
    status: LiveSessionStatus
    stream_path: str
    sample_fps: float
    simulated: bool
    whep_url: str

    def to_camel_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "createdAt": self.created_at,
            "endedAt": self.ended_at,
            "status": self.status.value,
            "streamPath": self.stream_path,
            "sampleFps": self.sample_fps,
            "simulated": self.simulated,
            "whepUrl": self.whep_url,
        }


def zone_from_camel_dict(data: dict[str, Any]) -> InspectionZone:
    """Inverse of pipeline.schemas.zone_to_camel_dict — the browser only
    ever holds the camelCase zone shape it received over the live events
    WebSocket, so Save-for-Follow-Up (POST .../save-area) must parse that
    shape back, not the Python-native snake_case InspectionZone fields.
    """
    evidence = data.get("evidence") or {}
    return InspectionZone(
        id=data["id"],
        review_priority=data["reviewPriority"],
        anomaly_score=data["anomalyScore"],
        appearance_anomaly_score=data.get("appearanceAnomalyScore", 0.0),
        structural_anomaly_score=data.get("structuralAnomalyScore", 0.0),
        review_score=data.get("reviewScore", 0.0),
        persistence_score=data["persistenceScore"],
        primary_type=data.get("primaryType", "general_visual_variation"),
        primary_signal_label=data.get("primarySignalLabel", "General visual variation"),
        persistent_observations=data.get("persistentObservations", 0),
        total_observations=data.get("totalObservations", 0),
        first_seen_ms=data["firstSeenMs"],
        last_seen_ms=data["lastSeenMs"],
        first_seen_sec=data["firstSeenSec"],
        last_seen_sec=data["lastSeenSec"],
        frames_seen=data["framesSeen"],
        relative_location=data["relativeLocation"],
        centroid_norm=data["centroidNorm"],
        bbox_norm=data["bboxNorm"],
        evidence=ZoneEvidence(
            crop_coverage_delta=evidence.get("cropCoverageDelta"),
            color_difference=evidence.get("colorDifference"),
            vegetation_difference=evidence.get("vegetationDifference"),
            texture_difference=evidence.get("textureDifference"),
            embedding_difference=evidence.get("embeddingDifference"),
            persistence=evidence.get("persistence"),
            appearance_anomaly_score=evidence.get("appearanceAnomalyScore"),
            structural_anomaly_score=evidence.get("structuralAnomalyScore"),
            row_continuity_before=evidence.get("rowContinuityBefore"),
            row_continuity_after=evidence.get("rowContinuityAfter"),
            gap_extent_normalized=evidence.get("gapExtentNormalized"),
            soil_exposure_delta=evidence.get("soilExposureDelta"),
            fragmentation_score=evidence.get("fragmentationScore"),
            registration_confidence=evidence.get("registrationConfidence"),
        ),
        reasons=data.get("reasons", []),
        recommendation=data.get("recommendation", ""),
    )


class LiveInspectionAreaEvent(InspectionZone):
    """Extends the existing InspectionZone wholesale — live mode reuses the
    exact recorded-mode taxonomy and evidence shape, only adding the two
    live-specific fields below."""

    live_frame_timestamp_ms: int
    telemetry_association: TelemetryAssociation

    def to_camel_dict(self) -> dict[str, Any]:
        out = zone_to_camel_dict(self)
        out["liveFrameTimestampMs"] = self.live_frame_timestamp_ms
        out["telemetryAssociation"] = self.telemetry_association.to_camel_dict()
        return out


class SavedInspectionArea(BaseModel):
    id: str
    session_id: str
    zone: InspectionZone
    telemetry_association: TelemetryAssociation
    snapshot_url: str | None = None
    saved_at: str
    note: str | None = None

    def to_camel_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sessionId": self.session_id,
            "zone": zone_to_camel_dict(self.zone),
            "telemetryAssociation": self.telemetry_association.to_camel_dict(),
            "snapshotUrl": self.snapshot_url,
            "savedAt": self.saved_at,
            "note": self.note,
        }


class FrameAnalyzedEvent(BaseModel):
    type: Literal["frame_analyzed"] = "frame_analyzed"
    session_id: str
    frame_timestamp_ms: int
    zones: list[LiveInspectionAreaEvent]

    def to_camel_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "sessionId": self.session_id,
            "frameTimestampMs": self.frame_timestamp_ms,
            "zones": [z.to_camel_dict() for z in self.zones],
        }


class StateChangeEvent(BaseModel):
    type: Literal["state_change"] = "state_change"
    session_id: str
    status: LiveSessionStatus

    def to_camel_dict(self) -> dict[str, Any]:
        return {"type": self.type, "sessionId": self.session_id, "status": self.status.value}


class TelemetryEvent(BaseModel):
    type: Literal["telemetry"] = "telemetry"
    session_id: str
    telemetry: DroneTelemetry

    def to_camel_dict(self) -> dict[str, Any]:
        return {"type": self.type, "sessionId": self.session_id, "telemetry": self.telemetry.to_camel_dict()}


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    session_id: str
    message: str

    def to_camel_dict(self) -> dict[str, Any]:
        return {"type": self.type, "sessionId": self.session_id, "message": self.message}


LiveStreamEvent = Union[FrameAnalyzedEvent, StateChangeEvent, TelemetryEvent, ErrorEvent]
