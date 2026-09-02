from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

import cv2

from cropmerge.pipeline.schemas import VideoSourceMeta

log = logging.getLogger("cropmerge.video.metadata")

# QuickTime/MP4 "location" tag (DJI and iPhone both write this), e.g.
# "+40.1234-088.5678+041.200/" — signed lat, signed lon, optional signed alt.
_ISO6709_RE = re.compile(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?/?$")


def _parse_iso6709(value: str | None) -> dict | None:
    if not isinstance(value, str):
        return None
    match = _ISO6709_RE.match(value.strip())
    if not match:
        return None
    lat_str, lon_str, alt_str = match.groups()
    try:
        return {
            "latitude": float(lat_str),
            "longitude": float(lon_str),
            "altitude": float(alt_str) if alt_str is not None else None,
        }
    except ValueError:
        return None


def _dms_to_decimal(dms, ref: str) -> float | None:
    try:
        degrees, minutes, seconds = (float(part) for part in dms)
    except (TypeError, ValueError):
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _image_gps(path: Path) -> dict | None:
    """EXIF GPS for JPEG/TIFF photos — the primary artifact type for
    consumer drones (DJI Mini 2) with no video telemetry export."""
    try:
        from PIL import ExifTags, Image

        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
            if not gps_ifd:
                return None
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
    except Exception:
        return None

    lat = _dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef", "N")) if gps.get("GPSLatitude") else None
    lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E")) if gps.get("GPSLongitude") else None
    if lat is None and lon is None:
        return None
    altitude = None
    if gps.get("GPSAltitude") is not None:
        try:
            altitude = float(gps["GPSAltitude"])
            if gps.get("GPSAltitudeRef") == 1:
                altitude = -altitude
        except (TypeError, ValueError):
            altitude = None
    return {"latitude": lat, "longitude": lon, "altitude": altitude}


def _ffprobe(path: Path) -> dict:
    if not shutil.which("ffprobe"):
        return {}
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return json.loads(out)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as e:
        log.warning("ffprobe failed for %s: %s", path, e)
        return {}


def _opencv_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    if meta["fps"] <= 1e-3:
        meta["fps"] = 30.0
    if meta["frame_count"] <= 0 and meta["fps"] > 0:
        meta["frame_count"] = 0
    meta["duration_sec"] = (
        meta["frame_count"] / meta["fps"] if meta["fps"] > 0 else 0.0
    )
    return meta


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v"}


def _image_meta(path: Path) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    h, w = bgr.shape[:2]
    return {
        "width": int(w),
        "height": int(h),
        "fps": 1.0,
        "frame_count": 1,
        "duration_sec": 0.0,
    }


def extract_metadata(path: str | Path) -> VideoSourceMeta:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix not in VIDEO_EXTS | IMAGE_EXTS:
        raise ValueError(f"Unsupported media extension: {path.suffix}")

    if suffix in IMAGE_EXTS:
        cv = _image_meta(path)
        return VideoSourceMeta(
            filename=path.name,
            path=str(path),
            duration_sec=float(cv["duration_sec"]),
            width=int(cv["width"]),
            height=int(cv["height"]),
            fps=float(cv["fps"]),
            frame_count=int(cv["frame_count"]),
            codec="image",
            orientation=None,
            created_at=None,
            gps=_image_gps(path),
        )

    cv = _opencv_meta(path)
    probe = _ffprobe(path)
    codec = None
    created = None
    orientation = None
    gps = None

    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            codec = stream.get("codec_name")
            tags = stream.get("tags") or {}
            rot = tags.get("rotate") or stream.get("rotation")
            if rot is not None:
                try:
                    orientation = int(float(rot))
                except (TypeError, ValueError):
                    pass
            break

    fmt = probe.get("format") or {}
    tags = fmt.get("tags") or {}
    created = tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate")
    # GPS hooks for DJI/QuickTime telemetry — never assumed present
    location_tag = tags.get("location") or tags.get("com.apple.quicktime.location.ISO6709")
    gps = _parse_iso6709(location_tag)

    width = int(cv["width"] or 0)
    height = int(cv["height"] or 0)
    fps = float(cv["fps"] or 0.0)
    frame_count = int(cv["frame_count"] or 0)
    duration = float(cv["duration_sec"] or 0.0)
    if duration <= 0 and fmt.get("duration"):
        try:
            duration = float(fmt["duration"])
            if frame_count <= 0 and fps > 0:
                frame_count = int(round(duration * fps))
        except (TypeError, ValueError):
            pass

    return VideoSourceMeta(
        filename=path.name,
        path=str(path),
        duration_sec=duration,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        codec=codec,
        orientation=orientation,
        created_at=created,
        gps=gps,
    )
