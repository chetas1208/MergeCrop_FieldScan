from __future__ import annotations

import json
import os
import time
from pathlib import Path

from scripts.cleanup_temp_storage import _scan_incomplete_uploads, _scan_jxl_archive_tmp

NOW = time.time()
OLD = NOW - 48 * 3600
FRESH = NOW - 1 * 3600


def _touch(path: Path, mtime: float, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))


def test_abandoned_upload_flagged_after_ttl(tmp_path: Path):
    uploads_dir = tmp_path / "uploads"
    d = uploads_dir / ".incomplete" / "abc123"
    _touch(d / "upload.json", OLD, json.dumps({"status": "uploading"}).encode())
    _touch(d / "chunk_00000000.part", OLD)

    candidates = _scan_incomplete_uploads(uploads_dir, ttl_hours=24, now=NOW)

    assert len(candidates) == 1
    assert candidates[0].path == d
    assert not candidates[0].anomaly
    assert "abandoned" in candidates[0].reason


def test_fresh_uploading_upload_not_flagged(tmp_path: Path):
    uploads_dir = tmp_path / "uploads"
    d = uploads_dir / ".incomplete" / "fresh1"
    _touch(d / "upload.json", FRESH, json.dumps({"status": "uploading"}).encode())

    candidates = _scan_incomplete_uploads(uploads_dir, ttl_hours=24, now=NOW)
    assert candidates == []


def test_completed_but_not_cleaned_up_is_an_anomaly_not_deletable(tmp_path: Path):
    uploads_dir = tmp_path / "uploads"
    d = uploads_dir / ".incomplete" / "stuck1"
    _touch(d / "upload.json", OLD, json.dumps({"status": "completed"}).encode())

    candidates = _scan_incomplete_uploads(uploads_dir, ttl_hours=24, now=NOW)

    assert len(candidates) == 1
    assert candidates[0].anomaly is True
    assert "completed" in candidates[0].reason


def test_partial_upload_with_no_record_flagged_after_ttl(tmp_path: Path):
    uploads_dir = tmp_path / "uploads"
    d = uploads_dir / ".incomplete" / "crashed1"
    _touch(d / "chunk_00000000.part", OLD)  # no upload.json at all

    candidates = _scan_incomplete_uploads(uploads_dir, ttl_hours=24, now=NOW)

    assert len(candidates) == 1
    assert not candidates[0].anomaly
    assert "no upload.json" in candidates[0].reason


def test_missing_uploads_dir_returns_empty(tmp_path: Path):
    assert _scan_incomplete_uploads(tmp_path / "does-not-exist", ttl_hours=24, now=NOW) == []


def test_jxl_tmp_orphan_detected_after_ttl(tmp_path: Path):
    data_dir = tmp_path / "data"
    stale = data_dir / "tmp" / "jxl-archive" / "abc.jxl"
    _touch(stale, OLD)
    fresh = data_dir / "tmp" / "jxl-archive" / "def.jxl"
    _touch(fresh, FRESH)

    candidates = _scan_jxl_archive_tmp(data_dir, ttl_hours=24, now=NOW)

    assert len(candidates) == 1
    assert candidates[0].path == stale
