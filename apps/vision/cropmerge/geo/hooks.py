"""Future geospatial hooks — V1 is image-relative only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FramePose:
    """Optional drone pose for future SfM / orthomosaic pipeline."""

    timestamp_sec: float
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    heading: float | None = None
    gimbal_pitch: float | None = None
    camera_intrinsics: dict[str, float] | None = None


def pose_from_metadata(meta_gps: dict[str, Any] | None, timestamp_sec: float = 0.0) -> FramePose:
    if not meta_gps:
        return FramePose(timestamp_sec=timestamp_sec)
    return FramePose(
        timestamp_sec=timestamp_sec,
        latitude=meta_gps.get("latitude"),
        longitude=meta_gps.get("longitude"),
        altitude=meta_gps.get("altitude"),
    )


FUTURE_PIPELINE = """
Drone images/video
  → GPS + orientation + camera calibration
  → multi-view correspondence
  → SfM / photogrammetry
  → orthomosaic
  → georeferenced field polygons
  → CropMerge map

Later integrations: USDA CDL, basemaps, terrain, weather, farm boundaries.
"""
