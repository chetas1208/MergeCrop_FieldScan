#!/usr/bin/env bash
# Keep VISION_PUBLIC_BASE_URL + Vercel in sync with the live cloudflared quick-tunnel URL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/scripts/tunnel-lock.sh"
TUNNEL_LOG="$ROOT/deploy/local/cloudflared.log"
TUNNEL_URL_FILE="$ROOT/deploy/local/tunnel.url"
VISION_ENV="$ROOT/deploy/local/vision.env"
ROOT_ENV="$ROOT/.env"
PORT="${VISION_PORT:-8001}"
VERCEL_SYNC_LOG="$ROOT/deploy/local/vercel-sync.log"

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

tunnel_health_ok() {
  local url="$1"
  [[ -n "$url" ]] && curl -sf "${url%/}/vision/health" >/dev/null 2>&1
}

set_env_url() {
  local file="$1"
  local url="$2"
  [[ -f "$file" ]] || return 0
  if grep -q '^VISION_PUBLIC_BASE_URL=' "$file"; then
    sed -i "s|^VISION_PUBLIC_BASE_URL=.*|VISION_PUBLIC_BASE_URL=${url}|" "$file"
  else
    echo "VISION_PUBLIC_BASE_URL=${url}" >>"$file"
  fi
}

sync_vercel_production() {
  local url="$1"
  command -v vercel >/dev/null 2>&1 || return 0
  (
    cd "$ROOT/apps/web"
    printf '%s' "$url" | vercel env add NUXT_PUBLIC_VISION_API_URL production --force >>"$VERCEL_SYNC_LOG" 2>&1
    vercel deploy --prebuilt --prod --yes >>"$VERCEL_SYNC_LOG" 2>&1 || \
      vercel deploy --prod --yes >>"$VERCEL_SYNC_LOG" 2>&1 || true
  )
}

URL="$(current_tunnel_url)"
if [[ -z "$URL" ]]; then
  echo "$(date -Is) sync-tunnel-url: no tunnel URL in log yet"
  exit 0
fi

if tunnel_lock_active; then
  LOCKED_URL="$(grep '^LOCKED_URL=' "$LOCK_FILE" 2>/dev/null | cut -d= -f2- | tr -d '[:space:]' || true)"
  if [[ -n "$LOCKED_URL" ]]; then
    URL="$LOCKED_URL"
  fi
  echo "$URL" >"$TUNNEL_URL_FILE"
  echo "$(date -Is) sync-tunnel-url: lock active until $(tunnel_lock_until) — URL frozen at $URL"
  exit 0
fi

if ! tunnel_health_ok "$URL"; then
  echo "$(date -Is) sync-tunnel-url: tunnel URL not healthy yet ($URL)"
  exit 0
fi

echo "$URL" >"$TUNNEL_URL_FILE"

OLD_URL=""
if [[ -f "$VISION_ENV" ]]; then
  OLD_URL="$(grep '^VISION_PUBLIC_BASE_URL=' "$VISION_ENV" | cut -d= -f2- | tr -d '[:space:]' || true)"
fi

set_env_url "$VISION_ENV" "$URL"
set_env_url "$ROOT_ENV" "$URL"

if [[ "$OLD_URL" == "$URL" ]]; then
  exit 0
fi

echo "$(date -Is) sync-tunnel-url: ${OLD_URL:-<unset>} -> $URL"

# Vision reads VISION_PUBLIC_BASE_URL at process start for artifact URLs.
if pgrep -f "uvicorn api.main:app --host 127.0.0.1 --port ${PORT}" >/dev/null; then
  "$ROOT/deploy/scripts/start-vision.sh" >>"$ROOT/deploy/local/watchdog.log" 2>&1 || true
fi

sync_vercel_production "$URL"
