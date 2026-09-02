# CropMerge Field Triage — Deployment

Deploy the **existing** local system without rewriting the CV pipeline. Vercel serves the Nuxt product UI only; the browser talks directly to the Python vision API through a stable Cloudflare Tunnel hostname for uploads, jobs, and artifacts.

## Architecture

```text
                     ┌─────────────────┐
                     │     GitHub      │
                     │    Monorepo     │
                     └────────┬────────┘
                              │ push
                              ▼
                     ┌─────────────────┐
                     │     Vercel      │
                     │  Nuxt + TS Web  │
                     └────────┬────────┘
                              │
                              ▼
                         User Browser
                              │
                    direct HTTPS (uploads, jobs, artifacts)
                              │
                              ▼
                 https://vision.<domain>
                              │
                              ▼
                   ┌──────────────────┐
                   │    Cloudflare    │
                   │  Named Tunnel    │
                   └────────┬─────────┘
                            │ outbound only
                            ▼
              ┌────────────────────────────┐
              │     LOCAL GPU SERVER       │
              │  FastAPI @ 127.0.0.1:8001  │
              │  SAM / DINO / OpenCV       │
              │  uploads / jobs / outputs  │
              └────────────────────────────┘
```

**Critical rule:** Nuxt/Vercel must never proxy MP4 uploads, annotated videos, or large artifacts. Vercel function bodies are limited to 4.5 MB; Cloudflare proxied uploads cap at 100–200 MB — use chunked uploads for larger drone footage.

## Components

| Layer | Location | Docs |
| --- | --- | --- |
| Nuxt web app | Vercel | [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md) |
| Vision API | Local GPU host | [server/README.md](./server/README.md) |
| Cloudflare Tunnel | systemd on GPU host | [cloudflare/README.md](./cloudflare/README.md) |
| systemd units | GPU host | [systemd/](./systemd/) |

## Persistent nohup (survives reboot)

```bash
# One-time: register @reboot + 5-minute watchdog
./deploy/scripts/install-reboot.sh

# Manual start (vision + tunnel, nohup)
./deploy/scripts/start-all.sh
```

| Script | Purpose |
| --- | --- |
| `start-vision.sh` | Vision API on `127.0.0.1:8001` (GPU, sam2/dinov2) |
| `start-tunnel.sh` | Cloudflare quick tunnel → vision |
| `start-all.sh` | Both |
| `watchdog.sh` | Restarts anything unhealthy (cron every 1 min) |
| `tunnel-lock.sh` | Freeze quick-tunnel URL for N days (local ops lock) |
| `tail-api-logs.sh` | Follow API + tunnel logs |

Logs: `deploy/local/reboot.log`, `deploy/local/watchdog.log`

### Quick tunnel lifespan (`trycloudflare.com`)

Cloudflare **does not** offer a 1-month setting on quick tunnels. There is no SLA; the URL changes whenever `cloudflared` restarts, and Cloudflare may delete tunnels disconnected for several minutes.

| Tunnel type | URL stability | Max practical span |
| --- | --- | --- |
| **Quick** (`trycloudflare.com`) | Random; changes on restart | Hours–days while process stays up; **not guaranteed** |
| **Named** (`vision.yourdomain.com`) | Fixed DNS | **Months+** (persists across restarts) |

This repo uses a **local 30-day lock** so our scripts do not restart or rotate the URL unless it actually dies:

```bash
./deploy/scripts/tunnel-lock.sh status
# re-lock after expiry:
./deploy/scripts/tunnel-lock.sh init 30 "$(cat deploy/local/tunnel.url)"
```

For a true fixed hostname with no monthly churn, use a [named tunnel](./cloudflare/README.md).

## GPU

Set in `deploy/local/vision.env`:

```dotenv
CUDA_VISIBLE_DEVICES=1          # free GPU on multi-3090 hosts
CROP_MERGE_DEVICE=cuda
CROP_MERGE_SEGMENTATION_BACKEND=sam2
CROP_MERGE_DINO_BACKEND=dinov2
```

Health should report `"device": "cuda"`, `"sam2Available": true`.

1. **Vision service** — create `/etc/cropmerge/vision.env`, install `cropmerge-vision.service`, verify `curl http://127.0.0.1:8001/vision/health`.
2. **Cloudflare Tunnel** — create named tunnel `cropmerge-field-triage`, hostname `vision.<domain>` → `http://127.0.0.1:8001`, install `cloudflared` systemd service.
3. **Vercel** — set env vars (see below), deploy from repo root with `pnpm --filter @cropmerge/web build`.
4. **CORS** — add the Vercel production origin to `VISION_CORS_ORIGINS` on the GPU host.
5. **Verify** — run the checklist in [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md#post-deployment-checks).

## Environment variables

Copy [.env.example](../.env.example) for local development. Production splits across Vercel (Nuxt) and `/etc/cropmerge/vision.env` (Python).

### Vercel (Preview + Production)

| Variable | Purpose |
| --- | --- |
| `NUXT_PUBLIC_VISION_API_URL` | `https://vision.<domain>` |
| `NUXT_PUBLIC_VISION_SMALL_UPLOAD_THRESHOLD_BYTES` | `25165824` (24 MiB fast path) |
| `NUXT_VISION_SHARED_SECRET` | Same 32+ byte secret as `VISION_SHARED_SECRET` |
| `NUXT_VISION_SESSION_ORIGINS` | Optional extra session origins (production Vercel URL is always allowed) |

Do **not** put `VISION_*`, model paths, or storage paths on Vercel.

### GPU host (`/etc/cropmerge/vision.env`)

See [server/README.md](./server/README.md). Must include matching `VISION_SHARED_SECRET`, production `VISION_CORS_ORIGINS`, and `VISION_PUBLIC_BASE_URL=https://vision.<domain>`.

## Verification

```bash
# A. Local Python
curl -s http://127.0.0.1:8001/vision/health | jq .

# B. Public tunnel
curl -s https://vision.<domain>/vision/health | jq .

# C. CORS (replace origins)
curl -sI -X OPTIONS https://vision.<domain>/vision/uploads/init \
  -H "Origin: https://<vercel-production-domain>" \
  -H "Access-Control-Request-Method: POST"

# D. Logs
journalctl -u cropmerge-vision -f
journalctl -u cloudflared -f
```

## Preview deployments

Arbitrary `*.vercel.app` preview URLs are **not** automatically allowed on the vision API. Either:

- add specific preview origins to `VISION_CORS_ORIGINS`, or
- accept that preview builds cannot reach the GPU server until configured.

## Local development (unchanged)

```bash
# Terminal 1 — vision
cd apps/vision && source .venv/bin/activate
uvicorn api.main:app --host 127.0.0.1 --port 8001

# Terminal 2 — web
pnpm dev
```

Set `NUXT_PUBLIC_VISION_API_URL=http://127.0.0.1:8001` in a local `.env` (not committed).
