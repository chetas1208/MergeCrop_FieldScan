# Vision API — GPU host service

The Python FastAPI application in `apps/vision` runs on the local GPU server, bound to **127.0.0.1:8001**. It is not deployed to Vercel.

## Startup command

```bash
/home/923873155/CrropMerge\ Assignment/apps/vision/.venv/bin/python \
  -m uvicorn api.main:app --host 127.0.0.1 --port 8001 --workers 1
```

Adjust paths if the repository lives elsewhere. Use `--workers 1` because the job queue intentionally runs one GPU analysis at a time.

## Environment file

Create `/etc/cropmerge/vision.env` with mode `0600`, owned by the service user:

```dotenv
VISION_SHARED_SECRET=<same value as NUXT_VISION_SHARED_SECRET on Vercel>
VISION_CORS_ORIGINS=https://<vercel-production-domain>,http://localhost:3000,http://127.0.0.1:3000
VISION_PUBLIC_BASE_URL=https://vision.<domain>
VISION_HOST=127.0.0.1
VISION_PORT=8001
VISION_DATA_DIR="/home/923873155/CrropMerge Assignment/data"
VISION_UPLOAD_DIR="/home/923873155/CrropMerge Assignment/data/uploads"
VISION_OUTPUT_DIR="/home/923873155/CrropMerge Assignment/outputs"
VISION_JOB_DATABASE_PATH="/home/923873155/CrropMerge Assignment/data/db/vision-jobs.sqlite"
VISION_UPLOAD_CHUNK_BYTES=8388608
VISION_SMALL_UPLOAD_THRESHOLD_BYTES=25165824
VISION_MAX_UPLOAD_BYTES=10737418240
VISION_ARTIFACT_TOKEN_TTL_SECONDS=3600
CROP_MERGE_SEGMENTATION_BACKEND=heuristic
CROP_MERGE_DINO_BACKEND=heuristic
```

Generate the shared secret:

```bash
openssl rand -hex 32
```

After SAM/DINO weights are downloaded, source `apps/vision/models/weights.env` or set backend env vars in this file.

## systemd installation

```bash
sudo mkdir -p /etc/cropmerge
sudo cp deploy/server/vision.env.example /etc/cropmerge/vision.env
sudo chmod 600 /etc/cropmerge/vision.env
sudo chown $(whoami):$(whoami) /etc/cropmerge/vision.env
# Edit /etc/cropmerge/vision.env with real values

sudo cp deploy/systemd/cropmerge-vision.service.example /etc/systemd/system/cropmerge-vision.service
# Edit User= and paths if needed

sudo systemctl daemon-reload
sudo systemctl enable --now cropmerge-vision
sudo systemctl status cropmerge-vision
```

If an older uvicorn process is already listening on port 8001, stop it before enabling the service:

```bash
sudo systemctl stop cropmerge-vision 2>/dev/null || true
pkill -f "uvicorn api.main:app.*8001" || true
sudo systemctl start cropmerge-vision
```

## Health checks

| Endpoint | Auth | Purpose |
| --- | --- | --- |
| `GET /vision/health` | Public | Lightweight liveness (no inference) |
| `GET /vision/analyses/{id}` | Bearer JWT | Job status + progress |

Expected health response:

```json
{
  "status": "ok",
  "service": "cropmerge-vision",
  "version": "0.1.0",
  "device": "cuda",
  "opencvAvailable": true
}
```

## Logs and restart

```bash
journalctl -u cropmerge-vision -f
sudo systemctl restart cropmerge-vision
```

## API surface (browser-facing)

All authenticated routes require `Authorization: Bearer <session-token>` from `POST /api/vision/session` on the Nuxt host.

| Route | Description |
| --- | --- |
| `POST /vision/uploads` | Small file fast path (multipart) |
| `POST /vision/uploads/init` | Start chunked upload |
| `PUT /vision/uploads/{id}/chunks/{index}` | Upload one chunk |
| `POST /vision/uploads/{id}/complete` | Assemble file |
| `POST /vision/analyses` | Start async job (202) |
| `GET /vision/analyses/{id}` | Poll status |
| `GET /vision/artifacts/{runId}/{name}?token=…` | Signed artifact download (supports Range) |

Legacy synchronous `POST /vision/analyze` is disabled unless `VISION_ENABLE_LEGACY_SYNC_ANALYZE=true`.
