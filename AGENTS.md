# CropMerge Field Triage — agent rules

## Product

RGB drone **video or image** → field segmentation → visual anomaly mapping → farmer review.  
**Not** disease / nutrient / irrigation / yield diagnosis.

## Stack

- `apps/web` — Nuxt 3 product UI + public API (jobs, artifacts, frame review)
- `apps/vision` — Python FastAPI CV engine (`field_cv` / heuristic offline default)
- `packages/*` — shared types + validation
- Default DB: SQLite under `data/db/` (paths with spaces must be quoted)

## Runtime

- Vision: `127.0.0.1:8001` (`uvicorn api.main:app`)
- Web: `localhost:3000` (`pnpm dev`)
- Env: `VISION_SERVICE_URL`, `OUTPUTS_DIR`, `UPLOADS_DIR`, `CROP_MERGE_OUTPUTS`

## UX invariants

- Upload drop zone must **not** cover Analyze/Browse with a full-area file input (buttons stay clickable).
- Results page must show **every sampled frame** (`FrameReview`: filmstrip, raw/overlay/split, quality, zone→frame jump).
- Real samples served from `apps/web/public/samples/` with one-click loaders on the home page.
- Language: visual anomaly / inspection zone / review recommended — never disease claims.

## Samples

| Path | Notes |
|------|--------|
| `apps/web/public/samples/real_field_drone.mp4` | Real drone clip (CC BY-SA 4.0) |
| `apps/web/public/samples/real_soybean_field.jpg` | USDA soybean (public domain) |
| `apps/vision/samples/synthetic_field.mp4` | Offline synthetic |
| Attribution | `.../samples/SOURCES.md` |

## GPU / model backends (this host)

- `apps/vision/.venv` is Python 3.12 with `torch==2.5.1+cu121` / `torchvision==0.20.1+cu121` (matches this host's driver 535.288.01 / CUDA 12.2, 2x RTX 3090 free). Real SAM2 + DINOv2 verified running (`is_fallback=false`) — see `deploy/local/vision.env`. The old Python 3.13 venv (cu130, couldn't see the GPU at all) is preserved at `apps/vision/.venv-py313-broken-cuda` for reference, unused.
- Optional local-LLM enrichment (`CROP_MERGE_LLM_ENRICHMENT=true`, `cropmerge/enrich/llm_explain.py`) rewrites Inspection Area `reasons` into a short farmer-facing note via Qwen2.5-7B-Instruct on `cuda:1`, downloaded to `/usr/data/923873155/llm-models` (not under the repo). Off by default; never fabricates — absent `llmSummary` means "show `reasons` instead."
- Docker Compose's `vision` image stays CPU/heuristic-only by design (portability) — this GPU setup is the native/nohup path only.

## Live Drone (Docker Compose only)

- Two input modes share one analysis engine: **Recorded Scan** (existing, above) and **Live Drone** (Mavic 3M → RC Pro Enterprise → DJI Pilot 2 → RTMP/RTSP → MediaMTX → WebRTC/WHEP to browser + RTSP to `apps/vision`'s live worker).
- Start: `docker compose up --build` → `http://localhost:3000` → "Live Drone" nav link. New services: `web` (3000), `vision` (8001), `mediamtx` (1935 RTMP / 8554 RTSP / 8889 WHEP), `mosquitto` (1883 MQTT).
- **Live Drone mode never runs on Vercel.** It needs persistent WHEP + MQTT connections that a serverless Nitro deploy cannot hold — `NUXT_PUBLIC_MEDIAMTX_WHEP_URL` must stay unset there (empty = disabled, matches the existing opt-in-env pattern). Never wire MediaMTX/MQTT client code into `apps/web/server/`.
- No hardware needed to test the full loop: "Start Simulated Live Input" runs `apps/vision/scripts/publish_simulated_live.py` (FFmpeg loop of a bundled sample) into MediaMTX; the UI always labels this **SIMULATED LIVE INPUT**.
- Backend code: `apps/vision/cropmerge/live/` (RTSP worker, session store, WS event hub, export) and `apps/vision/cropmerge/telemetry/` (DJI Cloud API MQTT client, association) — see `docs/DJI_M3M_FIELD_TEST.md` for what's verified vs. hardware-pending.
- DJI MQTT topic names/payload fields in `telemetry/dji_cloud_client.py` are placeholders tagged `NEEDS_VERIFICATION_FROM_DJI_DOCS` — confirm against DJI's Cloud API Thing Model reference before trusting them against real hardware.

## GitHub

- Remote: `git@github.com:chetas1208/CropMerge-Field-Triage.git` (**private**)
- Follow global ship playbook: `~/.grok/rules/github-private-ship.md`
- Do not commit: `.venv`, `node_modules`, weights (`*.pt`/`*.pth`), sqlite, `outputs/`, raw `source_*` media, secrets

## Verify before calling UI work done

- Happy-path analyze (video + image) via `/api/analyses`
- Artifact URLs 200 for video, heatmap, montage, `frames/frame_XXXX.jpg`, `overlays/overlay_XXXX.jpg`
- If browser tools missing: API + static checks; say what was not browser-verified
