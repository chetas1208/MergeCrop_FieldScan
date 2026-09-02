from __future__ import annotations

import os
import time

from cropmerge.storage.policies import StorageClass
from cropmerge.storage.retention import (
    DEFAULT_TEMP_TTL_HOURS,
    find_orphaned_temp_files,
    get_temp_ttl_hours,
)


def test_storage_class_has_expected_members():
    names = {c.name for c in StorageClass}
    assert names == {
        "SCIENTIFIC_SOURCE",
        "REVERSIBLE_SOURCE",
        "VALIDATED_LOSSY_SOURCE",
        "DERIVED_ANALYSIS",
        "DERIVED_PREVIEW",
        "TEMPORARY",
    }
    for member in StorageClass:
        assert member.__doc__, f"{member.name} must document its compression policy"


def test_get_temp_ttl_hours_default(monkeypatch):
    monkeypatch.delenv("CROPMERGE_TEMP_TTL_HOURS", raising=False)
    assert get_temp_ttl_hours() == DEFAULT_TEMP_TTL_HOURS


def test_get_temp_ttl_hours_from_env(monkeypatch):
    monkeypatch.setenv("CROPMERGE_TEMP_TTL_HOURS", "6")
    assert get_temp_ttl_hours() == 6.0


def test_get_temp_ttl_hours_ignores_garbage(monkeypatch):
    monkeypatch.setenv("CROPMERGE_TEMP_TTL_HOURS", "not-a-number")
    assert get_temp_ttl_hours() == DEFAULT_TEMP_TTL_HOURS
    monkeypatch.setenv("CROPMERGE_TEMP_TTL_HOURS", "-5")
    assert get_temp_ttl_hours() == DEFAULT_TEMP_TTL_HOURS


def test_find_orphaned_temp_files_missing_dir_returns_empty(tmp_path):
    assert find_orphaned_temp_files(tmp_path / "nope", ttl_hours=1) == []


def test_find_orphaned_temp_files_identifies_old_files_only(tmp_path):
    now = time.time()
    old_file = tmp_path / "old.tmp"
    new_file = tmp_path / "new.tmp"
    old_file.write_bytes(b"old")
    new_file.write_bytes(b"new")

    ten_hours_ago = now - 10 * 3600
    one_hour_ago = now - 1 * 3600
    os.utime(old_file, (ten_hours_ago, ten_hours_ago))
    os.utime(new_file, (one_hour_ago, one_hour_ago))

    orphaned = find_orphaned_temp_files(tmp_path, ttl_hours=5, now=now)

    assert orphaned == [old_file]


def test_find_orphaned_temp_files_recurses_and_skips_directories(tmp_path):
    now = time.time()
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    old_nested = nested_dir / "old.tmp"
    old_nested.write_bytes(b"x")
    os.utime(old_nested, (now - 100 * 3600, now - 100 * 3600))

    orphaned = find_orphaned_temp_files(tmp_path, ttl_hours=1, now=now)

    assert orphaned == [old_nested]
    # Directories themselves are never returned.
    assert nested_dir not in orphaned


def test_find_orphaned_temp_files_does_not_delete_anything(tmp_path):
    now = time.time()
    f = tmp_path / "leave_me.tmp"
    f.write_bytes(b"data")
    os.utime(f, (now - 1000 * 3600, now - 1000 * 3600))

    find_orphaned_temp_files(tmp_path, ttl_hours=1, now=now)

    assert f.is_file()  # still there — the function is pure/read-only
