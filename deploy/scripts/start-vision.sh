#!/usr/bin/env bash
# Start CropMerge vision API (nohup). Survives SSH disconnect; pair with install-reboot.sh for boot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${VISION_ENV_FILE:-$ROOT/deploy/local/vision.env}"
LOG="$ROOT/deploy/local/vision.log"
PIDFILE="$ROOT/deploy/local/vision.pid"
PORT="${VISION_PORT:-8001}"

mkdir -p "$ROOT/deploy/local"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  exit 1
fi

# Stop prior instance on our port only
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  sleep 1
fi
pkill -f "uvicorn api.main:app --host 127.0.0.1 --port ${PORT}" 2>/dev/null || true
sleep 1

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
# Model weights (sam2 / dinov2) when present
if [[ -f "$ROOT/apps/vision/models/weights.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/apps/vision/models/weights.env"
fi
set +a

cd "$ROOT/apps/vision"
nohup .venv/bin/python -m uvicorn api.main:app \
  --host 127.0.0.1 --port "$PORT" --workers 1 >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Vision API PID $(cat "$PIDFILE") on 127.0.0.1:$PORT"
for _ in $(seq 1 20); do
  if curl -sf "http://127.0.0.1:${PORT}/vision/health" >/dev/null; then
    curl -s "http://127.0.0.1:${PORT}/vision/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print('device',d.get('device'),'seg',d.get('segmentationBackend'),'gpu',d.get('gpuAvailable'))"
    exit 0
  fi
  sleep 1
done
echo "Vision API did not become healthy — check $LOG" >&2
exit 1
