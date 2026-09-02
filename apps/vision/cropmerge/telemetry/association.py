from __future__ import annotations

from cropmerge.telemetry.schemas import DroneTelemetry, TelemetryAssociation, TelemetryAvailability

DEFAULT_EXACT_MS = 250
DEFAULT_TOLERANCE_MS = 2000


def nearest_telemetry(
    store,
    session_id: str,
    frame_ts_ms: int,
    *,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
    exact_ms: int = DEFAULT_EXACT_MS,
) -> TelemetryAssociation:
    """Find the closest telemetry sample to a frame timestamp, tagged
    exact/nearby/unavailable. Never interpolates or extrapolates a
    coordinate — if nothing is within `tolerance_ms`, or no telemetry
    exists at all for this session, the association is 'unavailable'.
    """
    row = store.nearest_telemetry_row(session_id, frame_ts_ms)
    if row is None:
        return TelemetryAssociation(availability=TelemetryAvailability.UNAVAILABLE, telemetry=None, tolerance_ms=tolerance_ms)

    delta_ms = abs(int(row["timestamp_ms"]) - frame_ts_ms)
    if delta_ms > tolerance_ms:
        return TelemetryAssociation(availability=TelemetryAvailability.UNAVAILABLE, telemetry=None, tolerance_ms=tolerance_ms)

    telemetry = DroneTelemetry(
        timestamp_ms=row["timestamp_ms"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        altitude_m=row["altitude_m"],
        heading_deg=row["heading_deg"],
        gimbal_pitch_deg=row["gimbal_pitch_deg"],
        source="dji-cloud-api",
    )
    availability = TelemetryAvailability.EXACT if delta_ms <= exact_ms else TelemetryAvailability.NEARBY
    return TelemetryAssociation(availability=availability, telemetry=telemetry, tolerance_ms=tolerance_ms)
