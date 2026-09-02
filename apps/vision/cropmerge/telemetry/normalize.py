from __future__ import annotations

from cropmerge.geo.hooks import FramePose
from cropmerge.telemetry.schemas import DroneTelemetry


def to_frame_pose(telemetry: DroneTelemetry) -> FramePose:
    """Convert normalized live telemetry into the existing (future-SfM-hook)
    FramePose shape, rather than duplicating its field list. FramePose stays
    the single source of truth for the eventual GPS→SfM→orthomosaic pipeline
    (cropmerge/geo/hooks.py); this is the only place live telemetry touches it.
    """
    return FramePose(
        timestamp_sec=telemetry.timestamp_ms / 1000.0,
        latitude=telemetry.latitude,
        longitude=telemetry.longitude,
        altitude=telemetry.altitude_m,
        heading=telemetry.heading_deg,
        gimbal_pitch=telemetry.gimbal_pitch_deg,
    )
