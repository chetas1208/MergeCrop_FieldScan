# CropMerge Field Triage

**RGB drone video → field segmentation → visual anomaly mapping → farmer review**

Product-shaped prototype for Midwest (Illinois) corn/soy context. Ordinary RGB (DJI Mini 2–class) footage in; structured inspection zones out. **Not** disease/nutrient/yield diagnosis.

```text
Nuxt/Vercel product UI  ──small session token──►  Python vision engine
Browser  ──direct uploads, jobs, artifacts──►  Cloudflare Tunnel ──► vision engine
```

## Architecture

```mermaid
flowchart TD
  V[RGB drone video] --> D[Decode + metadata]
  D --> S[Frame sample 1-5 FPS]
  S --> Q[Quality analysis]
  Q --> SEG[Segmentation SAM3 / heuristic]
  Q --> REG[Frame registration]
  SEG --> FM[Field + class masks]
  REG --> FM
  FM --> RGB[RGB ExG / color / texture]
  FM --> DINO[DINOv3 / heuristic embeddings]
  RGB --> AN[Spatial anomaly grid]
  DINO --> AN
  AN --> TC[Temporal consensus]
  TC --> Z[Inspection zones]
  Z --> OUT[Annotated MP4 + heatmap + JSON]
  OUT --> UI[CropMerge Nuxt UI]
```

## Monorepo layout

```text
apps/web          Nuxt + TypeScript product UI + public API
apps/vision       Python FastAPI CV engine
packages/types    Shared domain types
packages/validation  Zod schemas
docs/             Architecture, limitations, data sources
outputs/          Analysis artifacts
```

## Local databases

Default = **SQLite** (no Docker). Optional = **Postgres 16**.

```bash
bash scripts/init-local-dbs.sh
set -a; source data/db/local.env; set +a
```

| DB | File / URL | Owner |
|----|------------|--------|
| Product | `data/db/cropmerge.sqlite` | Nuxt jobs / analyses |
| Vision | `data/db/vision.sqlite` | Pipeline run log |

```bash
# optional Postgres
docker compose up -d postgres
export DATABASE_URL=postgresql://cropmerge:cropmerge@127.0.0.1:5432/cropmerge
export VISION_DATABASE_URL=postgresql://cropmerge:cropmerge@127.0.0.1:5432/cropmerge_vision
# Adminer UI: docker compose --profile tools up -d adminer → :8088
```

Health: `GET /api/health` (includes `db`), `GET /api/db/status`, `GET /vision/runs`.

## Quick start

### Docker Compose (recommended — also enables Live Drone mode)

```bash
cp .env.example .env   # set VISION_SHARED_SECRET to a random 32-byte value
docker compose up --build
# → http://localhost:3000
```

Starts `web` (3000), `vision` (8001), `mediamtx` (RTMP 1935 / RTSP 8554 / WHEP 8889), and
`mosquitto` (MQTT 1883). Postgres/Adminer stay opt-in (`--profile tools`). CPU-only in
Docker (no `torch` extra) — same heuristic-fallback behavior as the native path below.

### Manual (no Docker)

#### 1. Vision engine

```bash
cd apps/vision
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# optional offline demo video
python scripts/generate_demo_video.py
pytest -v
uvicorn api.main:app --host 127.0.0.1 --port 8001
```

Default backends are OpenCV `field_cv` + heuristic embeddings (no GPU weights required).
Optional SAM2 / DINOv2 weights: see `apps/vision/scripts/download_weights.py`.

#### 2. Web app

```bash
# from repo root
corepack enable && pnpm install
export VISION_SERVICE_URL=http://127.0.0.1:8001
export NUXT_PUBLIC_VISION_API_URL=http://127.0.0.1:8001
export NUXT_VISION_SHARED_SECRET='replace-with-a-random-32-byte-secret'
export VISION_SHARED_SECRET="$NUXT_VISION_SHARED_SECRET"
export VISION_CORS_ORIGINS=http://localhost:3000
export OUTPUTS_DIR=$PWD/outputs
export UPLOADS_DIR=$PWD/data/uploads
pnpm dev
# → http://localhost:3000
```

#### 3. Demo samples (in-repo)

| File | Kind | Notes |
|------|------|--------|
| `apps/web/public/samples/real_field_drone.mp4` | Real drone video | 10s crop field flyover (CC BY-SA 4.0) |
| `apps/web/public/samples/real_soybean_field.jpg` | Real field image | USDA soybean field (public domain) |
| `apps/vision/samples/synthetic_field.mp4` | Synthetic | Offline CI demo |

In the UI: **Try a real sample** → **Analyze field**. Attribution: `/samples/SOURCES.md`.

#### 4. One-shot CLI demo

```bash
pnpm demo
```

## Live Drone mode

Second input mode alongside recorded-video analysis: Mavic 3M → RC Pro Enterprise →
DJI Pilot 2 → RTMP/RTSP → MediaMTX → WebRTC/WHEP (browser) + RTSP (`apps/vision`'s
live CV worker). **Docker Compose only** — never available on the hosted Vercel
deployment (no persistent WHEP/MQTT there).

No DJI hardware needed to try it: open `/live-drone` → **Start Simulated Live Input**
(loops a bundled sample through the same MediaMTX pipeline a real drone would use,
always labeled "SIMULATED LIVE INPUT"). To connect a real Mavic 3M, see
`/live-drone/setup` and `docs/DJI_M3M_FIELD_TEST.md`.

Details: `docs/ARCHITECTURE.md` (pipeline diagram), `docs/LIVE_MODE_LIMITATIONS.md`
(what's simulated vs. real).

## Browser-facing API (Vision)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/vision/uploads` | Small direct multipart upload |
| POST | `/vision/uploads/init` | Start a resumable chunked upload |
| PUT | `/vision/uploads/:id/chunks/:index` | Upload one chunk |
| POST | `/vision/uploads/:id/complete` | Atomically assemble upload |
| POST | `/vision/analyses` | Queue analysis (`202 Accepted`) |
| GET | `/vision/analyses/:id` | Job status, report, signed artifact URLs |
| GET | `/vision/artifacts/:runId/:name` | Signed direct artifact download/stream |

`POST /api/vision/session` is the only Nuxt control-plane endpoint used by the deployed browser. The old `/api/analyses` and `/api/artifacts` routes are retained only behind `NUXT_ENABLE_LEGACY_LOCAL_PROXY=true`; they are disabled by default so Vercel never proxies media.

## Vision diagnostics

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/vision/health` | Lightweight public health check |
| POST | `/vision/analyze` | Legacy, explicitly opt-in synchronous endpoint |
| POST | `/vision/segment` | Authenticated debug segmentation |

## Model backends

| Component | Production path | Offline default |
|-----------|-----------------|-----------------|
| Segmentation | Meta SAM 3.1 (`SAM3_CHECKPOINT`) | `heuristic` (explicit fallback) |
| Embeddings | DINOv3 (`DINOV3_CHECKPOINT`) | histogram/texture descriptors |

Fallback outputs set `usedFallback: true` and are **never** labeled as SAM/DINO results.

## Language rules (hard)

Use: visual anomaly, inspection zone, review recommended, exploratory RGB analysis.

Avoid: disease detected, unhealthy crop, N deficiency, water stress confirmed, yield loss predicted.

## Outputs per run

```text
outputs/<run_id>/
  results.json
  metrics.json
  annotated_video.mp4
  heatmap.png
  segmentation_montage.jpg
  frames/
  overlays/
```

## Tests

```bash
cd apps/vision && pytest -v
pnpm -r test   # TypeScript: Vitest unit tests for live-mode logic + shared schemas
```

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/TECHNICAL_SUMMARY.md](docs/TECHNICAL_SUMMARY.md)
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md)
- [docs/LIVE_MODE_LIMITATIONS.md](docs/LIVE_MODE_LIMITATIONS.md)
- [docs/DJI_M3M_FIELD_TEST.md](docs/DJI_M3M_FIELD_TEST.md)

## License

Prototype for evaluation. Third-party model weights subject to their licenses (Meta SAM/DINO).
