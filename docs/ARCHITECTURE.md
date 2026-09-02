# Architecture

## Separation of concerns

| Layer | Owns | Must not own |
|-------|------|--------------|
| **Nuxt (`apps/web`)** | UX, uploads, job state, public REST, artifact proxy, typed contracts | PyTorch inference, mask math |
| **Vision (`apps/vision`)** | Decode, quality, segment, features, anomaly, temporal, render | Product copy beyond disclaimer, user accounts |
| **`packages/types`** | Shared TS domain model | Runtime logic |

Neither layer duplicates business rules: priority thresholds and fusion weights live in vision config; Nuxt only displays the contract.

## Pipeline

```text
Video → metadata/decode → sample → quality
     → segment/track (+ registration)
     → field mask
     → RGB features + embeddings
     → spatial grid anomaly
     → temporal consensus → zones
     → annotated video + heatmap + JSON
```

## Interfaces

- `Segmenter`: `segment_image` / `start_video` / `track_video`
- `EmbeddingExtractor`: `embed_tiles`
- Config YAML for prompts, grid, anomaly weights, temporal gates

## Geospatial (future hook only)

V1 is **image-relative**. Optional GPS fields preserved when present.

```text
Drone media → GPS/orientation/intrinsics → SfM → orthomosaic
  → georeferenced polygons → CropMerge map
  (+ USDA CDL, basemaps, terrain later)
```

See `cropmerge/geo/hooks.py`.

## Live Drone pipeline (Docker Compose only)

```text
Mavic 3M → RC Pro Enterprise → DJI Pilot 2 → RTMP/RTSP → MediaMTX
                                                              │
                                          ┌───────────────────┴───────────────────┐
                                          ▼                                       ▼
                                 WebRTC/WHEP → apps/web browser        RTSP → apps/vision live worker
```

The simulated path (`apps/vision/scripts/publish_simulated_live.py`, an
FFmpeg loop of a bundled sample) publishes into the same MediaMTX ingest
point a real Pilot 2 stream would use — everything downstream is identical
whether the source is real or simulated.

- `apps/vision/cropmerge/live/rtsp_worker.py` samples the RTSP feed at a
  configurable rate (default 2 FPS) and calls
  `cropmerge/pipeline/frame_analysis.py:analyze_single_frame()` — the exact
  same per-frame segmentation/anomaly logic recorded-mode's batch pipeline
  uses (extracted from `pipeline/processor.py` specifically so the two
  modes cannot drift into separate analysis engines).
- `cropmerge/live/rolling_zones.py` reuses `anomaly/temporal.py`'s
  `aggregate_zones()` unmodified over a rolling window, instead of a
  separate live-only persistence model.
- `cropmerge/telemetry/` is DJI Cloud API MQTT ingest, normalization, and
  frame/telemetry association (exact/nearby/unavailable, no interpolation)
  — kept import-independent from `pipeline/`/`segmentation/` so it could be
  lifted into a standalone service later without touching the CV code.
- Live sessions and saved Inspection Areas reuse the same `outputs/<id>/`
  artifact directory and signed-URL serving (`/vision/artifacts/{id}/{name}`)
  that recorded runs already use — no parallel artifact-serving path.

**Why this never runs on Vercel:** `apps/web`'s production deploy is
serverless (see `.vercelignore` excluding `apps/vision/`), which cannot
hold the persistent WHEP/WebSocket/MQTT connections Live Drone mode needs.
It is Docker-Compose/local-network only — see `docs/LIVE_MODE_LIMITATIONS.md`.
