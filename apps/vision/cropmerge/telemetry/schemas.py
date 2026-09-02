from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class TelemetryAvailability(str, Enum):
    EXACT = "exact"
    NEARBY = "nearby"
    UNAVAILABLE = "unavailable"


class DroneTelemetry(BaseModel):
    """Fields are None, never guessed, when the source message didn't carry
    them. `source='unavailable'` means no MQTT telemetry has been received
    for the session at all (e.g. simulated live input).
    """

    timestamp_ms: int
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    heading_deg: float | None = None
    gimbal_pitch_deg: float | None = None
    source: str = "unavailable"  # 'dji-cloud-api' | 'unavailable'

    def to_camel_dict(self) -> dict:
        return {
            "timestampMs": self.timestamp_ms,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitudeM": self.altitude_m,
            "headingDeg": self.heading_deg,
            "gimbalPitchDeg": self.gimbal_pitch_deg,
            "source": self.source,
        }


class TelemetryAssociation(BaseModel):
    availability: TelemetryAvailability
    telemetry: DroneTelemetry | None = None
    tolerance_ms: int

    def to_camel_dict(self) -> dict:
        return {
            "availability": self.availability.value,
            "telemetry": self.telemetry.to_camel_dict() if self.telemetry else None,
            "toleranceMs": self.tolerance_ms,
        }
