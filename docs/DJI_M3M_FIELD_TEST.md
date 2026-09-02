# DJI Mavic 3M Field Test — Hardware Verification Checklist

This is the hand-off from "built and tested in simulation" to "tested against
real DJI hardware." **Do not check a box unless it was physically observed.**
Every DJI-touching feature must be tagged with exactly one of:

- `VERIFIED_IN_CODE` — implemented, reads/writes the right shape, no hardware needed to confirm.
- `VERIFIED_WITH_SIMULATOR` — exercised end-to-end via `publish_simulated_live.py` / a mock MQTT publisher.
- `VERIFIED_WITH_HARDWARE` — confirmed against a real Pilot 2 / RC Pro Enterprise / Mavic 3M session.
- `UNVERIFIED_REQUIRES_HARDWARE` — code exists, cannot be confirmed without hardware.
- `NOT_IMPLEMENTED` — out of scope for this campaign.

## Feature verification status (as of Campaign 01)

| Feature | Status | Notes |
|---|---|---|
| MediaMTX RTMP/RTSP/WHEP pipeline | `VERIFIED_WITH_SIMULATOR` | Config validated (`docker compose config`); build not tested (no Docker daemon in the dev sandbox this was built in) — validate `docker compose up --build` on a real Docker host before hardware day. |
| Simulated live publisher → MediaMTX → browser WHEP + vision RTSP worker | `VERIFIED_IN_CODE` | Router endpoints and event pipeline tested via FastAPI TestClient; full video path needs a running MediaMTX (untested in this sandbox — no Docker daemon available). |
| Live session create/start/stop/export | `VERIFIED_WITH_SIMULATOR` | Full round-trip tested via TestClient: session create → save-area → export → signed artifact URLs. |
| Save-for-Follow-Up (telemetry association) | `VERIFIED_IN_CODE` | exact/nearby/unavailable logic unit-tested; never fabricates coordinates. |
| Live Inspection Area WebSocket events + browser overlay | `VERIFIED_IN_CODE` | Coordinate-mapping math unit-tested (letterbox correctness); full live-stream visual check not run (no Docker/MediaMTX in this sandbox). |
| DJI Cloud API MQTT client (`telemetry/dji_cloud_client.py`) | `UNVERIFIED_REQUIRES_HARDWARE` | Topic names (`thing/product/{sn}/osd`, `.../state`) and payload field names are **placeholders tagged `NEEDS_VERIFICATION_FROM_DJI_DOCS`** — confirm against DJI's published Cloud API Thing Model reference before trusting them. |
| Mosquitto as the MQTT broker | `VERIFIED_IN_CODE` | Anonymous/plaintext, port 1883. Re-verify against real Pilot 2 requirements (see below). |
| Pilot 2 Open Platforms connection flow | `NOT_IMPLEMENTED` (UI copy only) | `/live-drone/setup` shows LAN-address candidates and the general DJI-documented flow; no code drives Pilot 2's pairing itself — that's operator-driven in Pilot 2's own UI. |
| Custom DJI controller APK / MSDK | `NOT_IMPLEMENTED` | Deliberately out of scope — DJI documents Pilot 2 / MSDK coexistence issues; Pilot 2 stays the flight system. |

## Pre-flight checklist (fill in during the actual test)

- [ ] RC Pro Enterprise and this laptop are on networks that can reach each other
- [ ] Pilot 2 Open Platforms configured with this laptop's LAN address (see `/live-drone/setup` for candidates — confirm manually, container-reported addresses are not authoritative)
- [ ] MQTT gateway/device connects to Mosquitto (`docker compose logs mosquitto`)
- [ ] Telemetry messages actually arrive (`GET /vision/health` → `mqttConnected: true`, and `TelemetryPanel` shows non-null values)
- [ ] Confirm real DJI OSD/state topic names and payload field names against DJI's Cloud API docs; update `telemetry/dji_cloud_client.py`'s `NEEDS_VERIFICATION_FROM_DJI_DOCS` constants if they differ
- [ ] Live stream begins (Pilot 2 configured RTMP/RTSP target = this MediaMTX instance)
- [ ] MediaMTX shows an active publisher on `cropmerge/live`
- [ ] Browser receives WebRTC video (not the "SIMULATED LIVE INPUT" banner)
- [ ] `apps/vision`'s RTSP worker independently receives frames (check `/vision/health` and live event stream)
- [ ] Frame timestamps vs. telemetry timestamps: confirm `exact`/`nearby`/`unavailable` tagging behaves sanely against real data (not just simulated `unavailable`)
- [ ] Save an Inspection Area for follow-up with real telemetry attached
- [ ] Export a session and confirm `inspection_areas.geojson` has real (not simulated-null) coordinates for the saved area
- [ ] Decide: does Mosquitto's plaintext/anonymous config satisfy what DJI's bridge actually sends, or is TLS (8883) / auth required?

## Known gaps going into hardware testing

- This build ran in a sandbox with no Docker daemon and no NVIDIA driver matching the installed CUDA build — `docker compose up --build` itself was config-validated but not build-tested; run it once on the real deployment host before the field day.
- `deploy/mosquitto/mosquitto.conf` is anonymous/plaintext by design for local dev — this is the first thing to reconsider once real DJI traffic is involved.
