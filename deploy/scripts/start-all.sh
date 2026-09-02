#!/usr/bin/env bash
# Start vision API + Cloudflare tunnel (nohup).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
"$ROOT/deploy/scripts/start-vision.sh"
"$ROOT/deploy/scripts/start-tunnel.sh"
