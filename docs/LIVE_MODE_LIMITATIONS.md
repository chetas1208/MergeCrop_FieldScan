# Live Drone Mode — Limitations

Sibling to `docs/LIMITATIONS.md` (recorded-mode limitations), which still
applies to the CV output itself (RGB-only, not calibrated probabilities,
not a health diagnosis). This file covers what's specific to Live Drone
mode's simulated-vs-real distinction.

## Simulated vs. real, feature by feature

| Aspect | Simulated Live Input | Real DJI hardware |
|---|---|---|
| Video source | Bundled sample MP4 looped via FFmpeg into MediaMTX | Pilot 2 → RTMP/RTSP → MediaMTX |
| UI label | Always shows "SIMULATED LIVE INPUT" | No badge |
| Telemetry | None — `DroneTelemetry.source = 'unavailable'` on every association | DJI Cloud API MQTT (`UNVERIFIED_REQUIRES_HARDWARE`, see `docs/DJI_M3M_FIELD_TEST.md`) |
| Save-for-Follow-Up coordinates | Always `telemetryAssociation.availability = 'unavailable'`, no coordinates in export | Real lat/lon when telemetry is within tolerance |
| CV backend | Same heuristic/CPU pipeline as recorded mode's default | Same — this campaign did not add or require GPU-only inference for live mode |
| Persistence model | Rolling-window temporal consensus (`cropmerge/live/rolling_zones.py`), same aggregation code as recorded mode | Same — no separate live-only anomaly logic |

## Explicitly not implemented in this campaign

- Custom DJI Android controller app / MSDK integration — Pilot 2 remains the flight system by design (DJI documents Pilot 2 / MSDK coexistence issues).
- Autonomous flight control, virtual-stick control, WPML mission generation.
- Real geospatial tracking across frames — persistence is video/image-relative (centroid/grid-based), not a full multi-object tracker with ground-truth coordinates, matching the same caveat recorded mode already documents.
- NodeODM / orthomosaic photogrammetry for live sessions.
- Automated Pilot 2 pairing — Pilot 2 Open Platforms connection is operator-driven in Pilot 2's own UI; CropMerge only shows connection info and detects telemetry/stream evidence.

## Deployment scope

Live Drone mode is Docker-Compose/local-network only. It is never available
on the hosted Vercel deployment of `apps/web` — that deployment target is
serverless and cannot hold the persistent WHEP/WebSocket/MQTT connections
Live Drone mode needs. `NUXT_PUBLIC_MEDIAMTX_WHEP_URL` unset (the default)
means the feature is off; the UI shows an explanatory message instead of
attempting to connect.
