# Cloudflare Tunnel — vision API exposure

Expose the local vision API at a stable hostname (e.g. `vision.example.com`) using a **remotely managed named tunnel**. Do not use temporary `trycloudflare.com` URLs in production. Do not open inbound port 8001 on the router or firewall.

## Prerequisites

- A domain on Cloudflare (DNS managed by Cloudflare)
- Cloudflare Zero Trust access (free tier is sufficient)
- `cloudflared` installed on the GPU host

## Named tunnel setup (Cloudflare Zero Trust dashboard)

1. Go to **Networks → Tunnels → Create a tunnel**.
2. Choose **Cloudflared** connector, name it **`cropmerge-field-triage`**.
3. Copy the **install command** (contains a one-time tunnel token). **Do not commit or log the token.**
4. Add a **Public Hostname**:
   - Subdomain: `vision`
   - Domain: `<your-domain>`
   - Service type: HTTP
   - URL: `http://127.0.0.1:8001`
5. Save. Cloudflare creates the DNS record automatically when proxied.

## Install cloudflared as a systemd service

On the GPU host (Linux with systemd):

```bash
# One-time install using the token from the dashboard (replace <token>):
sudo cloudflared service install <tunnel-token>

sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```

The token file is stored under `/etc/cloudflared/` with restricted permissions. Never add it to Git.

## Logs

Two log files cover the Cloudflare → vision path:

| File | Contents |
| --- | --- |
| `deploy/local/cloudflared.log` | Tunnel connector status, connection errors, public URL |
| `deploy/local/vision-api.log` | Structured JSON access log for every `/vision/*` request |

Tail both:

```bash
./deploy/scripts/tail-api-logs.sh
```

Each API line is JSON, for example:

```json
{"event":"api_access","request_id":"a1b2c3d4e5f6","method":"POST","path":"/vision/analyses","status":202,"duration_ms":12.4,"client_ip":"203.0.113.10","cf_ray":"abc123","origin":"https://cropmerge-field-triage.vercel.app","via_cloudflare":true,"resource_id":"d46506578751"}
```

Configure on the GPU host:

```dotenv
VISION_API_LOG_PATH=/path/to/deploy/local/vision-api.log
VISION_ACCESS_LOG_LEVEL=INFO
```

Notes:
- `GET /vision/health` is skipped by default to reduce noise (set header `X-Cropmerge-Access-Health: true` to log a health probe).
- Authorization headers, JWTs, and upload bytes are **never** logged.
- Responses include `X-Request-Id` for correlation.

Cloudflare connector log level:

```bash
cloudflared tunnel --url http://127.0.0.1:8001 --loglevel info
```

## Verify tunnel

```bash
# From any machine with network access:
curl -s https://vision.<domain>/vision/health | jq .

# On the GPU host:
journalctl -u cloudflared -f
```

Expected: HTTP 200 with `"service": "cropmerge-vision"`.

## CORS alignment

After Vercel deploys, add the production frontend origin to the GPU host env:

```dotenv
VISION_CORS_ORIGINS=https://<vercel-production-domain>,http://localhost:3000
VISION_PUBLIC_BASE_URL=https://vision.<domain>
```

Restart the vision service:

```bash
sudo systemctl restart cropmerge-vision
```

Test CORS:

```bash
curl -sI -X OPTIONS "https://vision.<domain>/vision/uploads/init" \
  -H "Origin: https://<vercel-production-domain>" \
  -H "Access-Control-Request-Method: POST"
```

Look for `access-control-allow-origin: https://<vercel-production-domain>`.

## Security notes

- Tunnel tokens grant connector access — treat as credentials.
- The Python API stays on `127.0.0.1`; only Cloudflare reaches it outbound.
- Authenticated analysis/upload routes require short-lived JWTs minted by Nuxt.
- Artifact downloads use separate signed query tokens (30–60 min TTL).
- Do not enable `allow_origins=["*"]` with credentials.

## Startup order after reboot

```text
network-online
    → cropmerge-vision.service (127.0.0.1:8001)
    → cloudflared.service (outbound tunnel)
    → https://vision.<domain>/vision/health returns 200
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Tunnel hostname 502 | `systemctl status cropmerge-vision`; confirm API on 8001 |
| Tunnel not connecting | `journalctl -u cloudflared`; egress to Cloudflare |
| CORS blocked in browser | `VISION_CORS_ORIGINS` includes exact Vercel origin (no trailing slash) |
| Upload fails at ~100 MB | Use chunked upload path (default above 24 MiB) |
| Video won't seek | Confirm `Range` returns 206 from `/vision/artifacts/...` |
