#!/usr/bin/env bash
# Keep vision + tunnel alive; sync public URL when cloudflared rotates.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/scripts/tunnel-lock.sh"
PORT="${VISION_PORT:-8001}"
TUNNEL_LOG="$ROOT/deploy/local/cloudflared.log"
TUNNEL_URL_FILE="$ROOT/deploy/local/tunnel.url"

current_tunnel_url() {
  if [[ -f "$TUNNEL_URL_FILE" ]]; then
    local from_file
    from_file="$(tr -d '[:space:]' <"$TUNNEL_URL_FILE" || true)"
    if [[ "$from_file" =~ ^https://[a-z0-9-]+\.trycloudflare\.com$ ]]; then
      echo "$from_file"
      return 0
    fi
  fi
  grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | tail -1 || true
}

TUNNEL_URL="$(current_tunnel_url)"

vision_ok=false
if curl -sf "http://127.0.0.1:${PORT}/vision/health" >/dev/null 2>&1; then
  vision_ok=true
fi

tunnel_ok=false
if [[ -n "$TUNNEL_URL" ]] && curl -sf "${TUNNEL_URL}/vision/health" >/dev/null 2>&1; then
  tunnel_ok=true
fi

if ! pgrep -f "uvicorn api.main:app --host 127.0.0.1 --port ${PORT}" >/dev/null || ! $vision_ok; then
  echo "$(date -Is) restarting vision"
  "$ROOT/deploy/scripts/start-vision.sh" || true
fi

if ! pgrep -f "cloudflared tunnel --url http://127.0.0.1:${PORT}" >/dev/null || ! $tunnel_ok; then
  if tunnel_lock_active && [[ -n "$TUNNEL_URL" ]] && curl -sf "${TUNNEL_URL}/vision/health" >/dev/null 2>&1; then
    echo "$(date -Is) tunnel lock active — not restarting healthy cloudflared"
  else
    echo "$(date -Is) restarting cloudflared"
    "$ROOT/deploy/scripts/start-tunnel.sh" || true
    TUNNEL_URL="$(current_tunnel_url)"
  fi
fi

if tunnel_lock_active; then
  exit 0
fi

"$ROOT/deploy/scripts/sync-tunnel-url.sh" || true
