# CropMerge Vision Engine

Python CV service. **Not** the product UI.

Owns: decode, quality, segmentation, DINO/RGB features, anomaly, temporal consensus, artifacts.

Does **not** own: UX, job store, public API contracts (Nuxt does).

## Quick start

```bash
cd apps/vision
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_demo_video.py
python scripts/analyze_video.py \
  --input samples/synthetic_field.mp4 \
  --output ../../outputs/demo \
  --segmentation-backend heuristic
pytest -v
export VISION_SHARED_SECRET='replace-with-a-random-32-byte-secret'
export VISION_CORS_ORIGINS=http://localhost:3000
uvicorn api.main:app --host 127.0.0.1 --port 8001
```

## Backends

| Backend | Role |
|---------|------|
| `heuristic` | Explicit RGB fallback (default offline) |
| `sam3` | Meta SAM 3.1 when `SAM3_CHECKPOINT` + package available |
| `dinov3` | Dense features when `DINOV3_CHECKPOINT` set |

Fallback outputs always set `usedFallback` / `is_fallback` — never claimed as SAM/DINO.

## API

- `GET /vision/health`
- `POST /vision/uploads` (small direct multipart upload)
- `POST /vision/uploads/init`, `PUT /vision/uploads/{id}/chunks/{index}`, `POST /vision/uploads/{id}/complete` (resumable upload)
- `POST /vision/analyses` (returns `202 Accepted`)
- `GET /vision/analyses/{id}` (poll job and receive signed artifact URLs)
- `GET /vision/artifacts/{run_id}/{name}?token=...` (direct, range-capable artifact streaming)
- `POST /vision/segment` (authenticated debug endpoint)

`GET /vision/health` remains public and performs no media decode or inference. All other endpoints require a 15-minute HS256 bearer token minted by Nuxt's private `POST /api/vision/session` endpoint. See the repository [deployment guide](../../docs/DEPLOYMENT.md).
