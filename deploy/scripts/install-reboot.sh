#!/usr/bin/env bash
# Register @reboot + watchdog cron jobs (no sudo). Re-run to refresh entries.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MARKER="# cropmerge-field-triage"
TMP="$(mktemp)"
( crontab -l 2>/dev/null | grep -v "$MARKER" || true ) >"$TMP"
cat >>"$TMP" <<EOF
@reboot sleep 45 && $ROOT/deploy/scripts/start-all.sh >> $ROOT/deploy/local/reboot.log 2>&1 $MARKER
*/1 * * * * $ROOT/deploy/scripts/watchdog.sh >> $ROOT/deploy/local/watchdog.log 2>&1 $MARKER
EOF
crontab "$TMP"
rm -f "$TMP"
echo "Installed crontab:"
crontab -l | grep "$MARKER"
