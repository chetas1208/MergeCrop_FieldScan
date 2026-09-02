from __future__ import annotations

import json
import zipfile
from pathlib import Path

from cropmerge.live.schemas import LiveSession
from cropmerge.live.session_store import LiveSessionStore
from cropmerge.telemetry.schemas import TelemetryAvailability

LIVE_ARTIFACT_NAMES = ("session.json", "inspection_areas.geojson", "fieldscan-results.zip")


def _saved_area_geojson_feature(area) -> dict:
    telemetry = area.telemetry_association.telemetry
    availability = area.telemetry_association.availability
    geometry = None
    if availability != TelemetryAvailability.UNAVAILABLE and telemetry and telemetry.latitude is not None and telemetry.longitude is not None:
        geometry = {"type": "Point", "coordinates": [telemetry.longitude, telemetry.latitude]}
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "id": area.id,
            "primaryType": area.zone.primary_type.value,
            "reviewPriority": area.zone.review_priority.value,
            "anomalyScore": area.zone.anomaly_score,
            "telemetryAvailability": availability.value,
            "savedAt": area.saved_at,
            "note": area.note,
        },
    }


def build_live_exports(session: LiveSession, store: LiveSessionStore, output_root: Path) -> dict:
    """Write session.json / inspection_areas.geojson / fieldscan-results.zip
    under outputs/<session_id>/, reusing the same directory convention
    recorded runs use so the existing signed-artifact route can serve them
    unmodified. Never fabricates geometry: a saved area with unavailable
    telemetry gets `geometry: null` in the GeoJSON, not a guessed point.
    """
    run_dir = output_root / session.id
    run_dir.mkdir(parents=True, exist_ok=True)
    saved_areas = store.list_saved_areas(session.id)

    session_payload = {
        **session.to_camel_dict(),
        "savedAreaCount": len(saved_areas),
    }
    session_path = run_dir / "session.json"
    session_path.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")

    geojson = {
        "type": "FeatureCollection",
        "features": [_saved_area_geojson_feature(a) for a in saved_areas],
    }
    geojson_path = run_dir / "inspection_areas.geojson"
    geojson_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")

    zip_path = run_dir / "fieldscan-results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(session_path, arcname="session.json")
        zf.write(geojson_path, arcname="inspection_areas.geojson")
        saved_dir = run_dir / "saved"
        if saved_dir.is_dir():
            for snapshot in saved_dir.glob("*.jpg"):
                zf.write(snapshot, arcname=f"saved/{snapshot.name}")

    return {"runId": session.id, "artifacts": [name for name in LIVE_ARTIFACT_NAMES if (run_dir / name).is_file()]}
