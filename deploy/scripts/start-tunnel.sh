#!/usr/bin/env bash
# Start Cloudflare quick tunnel -> local vision API (port 8001).
# For production use a named tunnel; see deploy/cloudflare/README.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/scripts/tunnel-lock.sh"
LOG="$ROOT/deploy/local/cloudflared.log"
TUNNEL_URL_FILE="$ROOT/deploy/local/tunnel.url"
PORT="${VISION_PORT:-8001}"
mkdir -p "$ROOT/deploy/local"

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
