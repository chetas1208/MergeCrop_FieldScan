#!/usr/bin/env bash
# Tail Cloudflare tunnel + vision API access logs together.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
API_LOG="${VISION_API_LOG_PATH:-$ROOT/deploy/local/vision-api.log}"
TUNNEL_LOG="$ROOT/deploy/local/cloudflared.log"
mkdir -p "$(dirname "$API_LOG")"
touch "$API_LOG" "$TUNNEL_LOG"
echo "API log:    $API_LOG"
echo "Tunnel log: $TUNNEL_LOG"
echo "---"
tail -n 40 -f "$API_LOG" "$TUNNEL_LOG"
