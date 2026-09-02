"""Retention config surface for TEMPORARY-class storage.

This module intentionally does NOT delete anything. It only:

  1. exposes the TTL configuration knobs, and
  2. provides a pure function that identifies which files under a temp
     directory are old enough to be considered orphaned.

Wiring an actual cleanup daemon/cron that calls `find_orphaned_temp_files`
and then deletes the results is a later phase — deleting files
automatically is exactly the kind of blast-radius expansion this
foundational task is scoped to avoid.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

#: Default time-to-live, in hours, for files under a designated temp
#: directory before they're considered orphaned. Overridable via the
#: CROPMERGE_TEMP_TTL_HOURS environment variable.
DEFAULT_TEMP_TTL_HOURS = 24.0

_TEMP_TTL_ENV_VAR = "CROPMERGE_TEMP_TTL_HOURS"


def get_temp_ttl_hours() -> float:
    """Read the configured temp-file TTL (hours) from the environment.

    Falls back to DEFAULT_TEMP_TTL_HOURS if unset or unparseable.
    """
    raw = os.environ.get(_TEMP_TTL_ENV_VAR, "")
    if not raw.strip():
        return DEFAULT_TEMP_TTL_HOURS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TEMP_TTL_HOURS
    if value <= 0:
        return DEFAULT_TEMP_TTL_HOURS
    return value


def find_orphaned_temp_files(
    temp_dir: str | Path,
    ttl_hours: float | None = None,
    now: float | None = None,
) -> list[Path]:
    """Return files under `temp_dir` whose mtime is older than `ttl_hours`.

    Pure and read-only: does not delete or modify anything, does not touch
    the filesystem beyond stat-ing files. Directories are not included in
    the result (only regular files). If `temp_dir` does not exist, returns
    an empty list rather than raising.

    Args:
        temp_dir: directory to scan (recursively).
        ttl_hours: override for the TTL; defaults to `get_temp_ttl_hours()`.
        now: override for "current time" (unix seconds), for deterministic
            testing; defaults to `time.time()`.
    """
    temp_dir = Path(temp_dir)
    if not temp_dir.is_dir():
        return []

    ttl = ttl_hours if ttl_hours is not None else get_temp_ttl_hours()
    cutoff = (now if now is not None else time.time()) - (ttl * 3600.0)

    orphaned: list[Path] = []
    for path in temp_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            # File vanished between listing and stat (e.g. concurrent
            # cleanup) — not our problem to report.
            continue
        if mtime < cutoff:
            orphaned.append(path)
    return orphaned
