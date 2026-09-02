#!/usr/bin/env bash
# Start Cloudflare quick tunnel -> local vision API (port 8001).
# For production use a named tunnel; see deploy/cloudflare/README.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/scripts/tunnel-lock.sh"
LOG="$ROOT/deploy/local/cloudflared.log"
TUNNEL_URL_FILE="$ROOT/deploy/local/tunnel.url"
mkdir -p "$ROOT/deploy/local"

# Resolve PORT from deploy/local/vision.env (not just an already-exported
# VISION_PORT in the calling shell) so this matches whatever start-vision.sh
# actually bound to.
VISION_ENV_FILE_FOR_PORT="${VISION_ENV_FILE:-$ROOT/deploy/local/vision.env}"
if [[ -f "$VISION_ENV_FILE_FOR_PORT" ]]; then
  # shellcheck disable=SC1090
  VISION_PORT="$(grep '^VISION_PORT=' "$VISION_ENV_FILE_FOR_PORT" | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true)"
fi
PORT="${VISION_PORT:-8001}"

if [[ "${FORCE_TUNNEL_RESTART:-0}" != "1" ]] && tunnel_lock_active; then
  LOCKED_URL="$(grep '^LOCKED_URL=' "$LOCK_FILE" 2>/dev/null | cut -d= -f2- | tr -d '[:space:]' || true)"
  if [[ -n "$LOCKED_URL" ]] && curl -sf "${LOCKED_URL%/}/vision/health" >/dev/null 2>&1; then
    echo "$LOCKED_URL" >"$TUNNEL_URL_FILE"
    echo "Tunnel lock active — keeping $LOCKED_URL (until $(tunnel_lock_until))"
    exit 0
  fi
  echo "Tunnel lock active but locked URL unhealthy — restarting cloudflared" >&2
fi

pkill -f "cloudflared tunnel --url http://127.0.0.1:${PORT}" 2>/dev/null || true
sleep 1
nohup cloudflared tunnel --url "http://127.0.0.1:${PORT}" --loglevel info >>"$LOG" 2>&1 &
echo "cloudflared PID=$!"
for _ in $(seq 1 15); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | tail -1 || true)"
  if [[ -n "$URL" ]]; then
    echo "$URL" >"$ROOT/deploy/local/tunnel.url"
    echo "Tunnel URL: $URL"
    "$ROOT/deploy/scripts/sync-tunnel-url.sh" || true
    exit 0
  fi
  sleep 1
done
echo "Tunnel starting — check $LOG"
tail -5 "$LOG"
