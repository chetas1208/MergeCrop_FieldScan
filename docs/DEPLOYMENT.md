# Deployment: Vercel + Cloudflare Tunnel + local Vision API

## Architecture

Vercel serves only `apps/web`. Browser media uploads, analysis status polling, and generated artifacts travel directly to `https://vision.<domain>` and never through a Nuxt/Vercel media route. Nuxt retains one small same-origin endpoint, `POST /api/vision/session`, which mints a 15-minute HS256 session token.

The vision API keeps a one-worker local queue, SQLite job state, chunked uploads, and artifacts on the GPU host. The CV pipeline, model selection, segmentation, and anomaly calculation are unchanged.

## Deployment inventory

| Item | Value |
| --- | --- |
| Web app | `apps/web` (`@cropmerge/web`) |
| Package manager | pnpm 9.15.0; Node 22.22.2 locally |
| Web build | `pnpm --filter @cropmerge/web build` |
| Vision app | `apps/vision` |
| Vision command | `.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8001 --workers 1` |
| Vision health | `GET /vision/health` |
| Production transport | named Cloudflare Tunnel hostname -> `http://127.0.0.1:8001` |

## Vercel project

Create one Vercel project named `cropmerge-field-triage` at the **repository root**. Set its framework to Nuxt and its build command to:

```bash
pnpm --filter @cropmerge/web build
```

The native Nuxt Vercel preset is selected automatically when Vercel builds the project; no custom Nitro preset or output-directory override is needed. `.vercelignore` prevents the local Python application, model files, database, uploads, and generated artifacts from being uploaded to Vercel.

Set these Vercel environment variables for Preview and Production before deploying:

| Variable | Value |
| --- | --- |
| `NUXT_PUBLIC_APP_NAME` | `CropMerge Field Triage` |
| `NUXT_PUBLIC_VISION_API_URL` | `https://vision.<domain>` |
| `NUXT_PUBLIC_VISION_SMALL_UPLOAD_THRESHOLD_BYTES` | `25165824` |
| `NUXT_VISION_SHARED_SECRET` | the generated 32-byte-or-longer shared secret |
| `NUXT_VISION_SESSION_ORIGINS` | Vercel production origin, plus any custom app origin |

Do **not** add `VISION_*`, storage paths, database URLs, model paths, or raw-source settings to Vercel. They describe the physical GPU server and would either be unused or expose deployment detail. `NUXT_VISION_SHARED_SECRET` is intentionally private; it is not a `NUXT_PUBLIC_*` variable.

## GPU host

Create `/etc/cropmerge/vision.env` with mode `0600`, owned by the service account. It must contain the same secret used by `NUXT_VISION_SHARED_SECRET` and production paths such as:

```dotenv
VISION_SHARED_SECRET=<same secret as Vercel>
VISION_CORS_ORIGINS=https://<your-vercel-production-domain>
VISION_PUBLIC_BASE_URL=https://vision.<domain>
VISION_DATA_DIR="/home/923873155/CrropMerge Assignment/data"
VISION_UPLOAD_DIR="/home/923873155/CrropMerge Assignment/data/uploads"
VISION_OUTPUT_DIR="/home/923873155/CrropMerge Assignment/outputs"
VISION_JOB_DATABASE_PATH="/home/923873155/CrropMerge Assignment/data/db/vision-jobs.sqlite"
VISION_UPLOAD_CHUNK_BYTES=8388608
VISION_SMALL_UPLOAD_THRESHOLD_BYTES=25165824
VISION_MAX_UPLOAD_BYTES=10737418240
CROP_MERGE_SEGMENTATION_BACKEND=field_cv
CROP_MERGE_DINO_BACKEND=heuristic
```

Install `deploy/systemd/cropmerge-vision.service.example` as `/etc/systemd/system/cropmerge-vision.service`, then enable and start it. It intentionally binds Uvicorn to `127.0.0.1`; do not open port 8001 in a firewall or router.

## Cloudflare Tunnel

Use a remotely managed named tunnel named `cropmerge-field-triage`. In Cloudflare Zero Trust, add a published application with:

| Setting | Value |
| --- | --- |
| Public hostname | `vision.<domain>` |
| Service | `http://127.0.0.1:8001` |

Install the tunnel with the one-time token supplied by Cloudflare:

```bash
sudo cloudflared service install <tunnel-token>
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```

The tunnel token must stay out of Git, shell history, logs, and this document. A named tunnel has an outbound connection only; no temporary `trycloudflare.com` URL and no inbound 8001 rule are used.

## Post-deployment checks

1. `GET https://vision.<domain>/vision/health` returns a lightweight `status: ok` response.
2. From the Vercel origin, upload the real image and video samples. The browser should send chunk/direct upload requests to `vision.<domain>`, receive a `202` job, and poll that host until completion.
3. Confirm `annotated_video.mp4` returns `206 Partial Content` for a `Range` request, then play and seek it in the deployed UI.
4. Confirm `results.json`, heatmap, montage, every `frames/frame_XXXX.jpg`, and every `overlays/overlay_XXXX.jpg` load from `vision.<domain>`, not `/api/artifacts` on Vercel.
