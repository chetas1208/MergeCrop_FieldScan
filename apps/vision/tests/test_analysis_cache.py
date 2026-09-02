from __future__ import annotations

from pathlib import Path

from cropmerge.storage.analysis_cache import AnalysisCache, compute_cache_key, config_content_hash

BASE_KWARGS = dict(
    source_sha256="a" * 64,
    cfg={"structural": {"x": 1}},
    sample_fps=1.0,
    max_frames=100,
    skip_dino=False,
    segmentation_backend="sam2",
    dino_backend="dinov2",
    analysis_version="analysis-v1",
)


def test_identical_inputs_produce_identical_key():
    a = compute_cache_key(**BASE_KWARGS)
    b = compute_cache_key(**BASE_KWARGS)
    assert a == b


def test_different_source_hash_changes_key():
    a = compute_cache_key(**BASE_KWARGS)
    b = compute_cache_key(**{**BASE_KWARGS, "source_sha256": "b" * 64})
    assert a != b


def test_different_config_content_changes_key():
    a = compute_cache_key(**BASE_KWARGS)
    b = compute_cache_key(**{**BASE_KWARGS, "cfg": {"structural": {"x": 2}}})
    assert a != b


def test_different_sample_fps_changes_key():
    a = compute_cache_key(**BASE_KWARGS)
    b = compute_cache_key(**{**BASE_KWARGS, "sample_fps": 2.0})
    assert a != b


def test_different_analysis_version_changes_key():
    a = compute_cache_key(**BASE_KWARGS)
    b = compute_cache_key(**{**BASE_KWARGS, "analysis_version": "analysis-v2"})
    assert a != b


def test_config_key_order_does_not_affect_hash():
    cfg_a = {"a": 1, "b": 2}
    cfg_b = {"b": 2, "a": 1}
    assert config_content_hash(cfg_a) == config_content_hash(cfg_b)


def test_cache_miss_returns_none(tmp_path: Path):
    cache = AnalysisCache(tmp_path / "cache.sqlite")
    assert cache.get("nonexistent") is None


def test_cache_put_then_get_round_trips(tmp_path: Path):
    cache = AnalysisCache(tmp_path / "cache.sqlite")
    cache.put("key1", "run123456789")

    entry = cache.get("key1")

    assert entry is not None
    assert entry.run_id == "run123456789"


def test_cache_put_overwrites_existing_key(tmp_path: Path):
    cache = AnalysisCache(tmp_path / "cache.sqlite")
    cache.put("key1", "old-run-id1")
    cache.put("key1", "new-run-id1")

    entry = cache.get("key1")

    assert entry.run_id == "new-run-id1"
