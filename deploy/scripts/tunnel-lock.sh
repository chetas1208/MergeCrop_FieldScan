#!/usr/bin/env bash
# Optional stability lock: keep the current public tunnel URL until LOCKED_UNTIL.
# Quick tunnels (trycloudflare.com) are not guaranteed by Cloudflare for any duration;
# this lock only prevents our scripts from rotating the URL unnecessarily.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCK_FILE="${TUNNEL_LOCK_FILE:-$ROOT/deploy/local/tunnel.lock}"

tunnel_lock_until() {
  if [[ ! -f "$LOCK_FILE" ]]; then
    return 0
  fi
  # shellcheck disable=SC1090
  source "$LOCK_FILE"
  echo "${LOCKED_UNTIL:-}"
}

tunnel_lock_active() {
  local until
  until="$(tunnel_lock_until)"
  [[ -n "$until" ]] || return 1
  local now epoch_until epoch_now
  epoch_until="$(date -d "$until" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S%z" "$until" +%s 2>/dev/null || echo 0)"
  epoch_now="$(date +%s)"
  (( epoch_until > epoch_now ))
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  case "${1:-status}" in
    status)
      if tunnel_lock_active; then
        echo "locked until $(tunnel_lock_until)"
        exit 0
      fi
      echo "not locked"
      exit 1
      ;;
    init)
      days="${2:-30}"
      mkdir -p "$(dirname "$LOCK_FILE")"
      until="$(date -u -d "+${days} days" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u -v+"${days}d" +"%Y-%m-%dT%H:%M:%SZ")"
      cat >"$LOCK_FILE" <<EOF
# CropMerge quick-tunnel stability lock (local ops only; not a Cloudflare SLA).
LOCKED_UNTIL=${until}
LOCKED_URL=${3:-}
EOF
      echo "Tunnel lock until ${until}"
      ;;
    *)
      echo "usage: $0 [status|init [days] [url]]" >&2
      exit 2
      ;;
  esac
fi
