"""Real bug fix, 2026-09-02, traced from field feedback ("zero findings
apparently impossible"): cropmerge/anomaly/spatial.py::_to_unit()'s
relative-ranking blend (normalize_scores() is a pure min-max rescale) always
inflated the single "most different" cell toward a high score every frame,
even when the underlying z-scores were all noise-level -- there is always a
most-different cell in any grid, rank normalization can't tell "genuinely
anomalous" from "the least identical of several near-identical cells."
These tests prove a genuinely uniform field now scores near-zero everywhere
(so zero Inspection Areas is actually reachable), while a real outlier is
still detected.
"""
from __future__ import annotations

import numpy as np

from cropmerge.anomaly.spatial import score_frame
from cropmerge.config import load_config


def _uniform_field_with_noise(h: int = 160, w: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), (40, 150, 40), dtype=np.float32)  # uniform green (BGR)
    noise = rng.normal(0, 1.5, size=base.shape)  # tiny sensor-noise-level variation
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def test_uniform_field_scores_near_zero_everywhere():
    cfg = load_config()
    cfg["anomaly"]["use_isolation_forest"] = False  # isolate the z-score/ranking fix being tested
    bgr = _uniform_field_with_noise()
    field = np.ones((160, 160), dtype=bool)
    label = np.full((160, 160), "CROP", dtype=object)
    emb = np.zeros((8, 8, 16), dtype=np.float32)  # identical (zero) embeddings everywhere

    cells, heat = score_frame(bgr, field, label, emb, cfg)

    scores = [c.appearance_anomaly_score for c in cells if c.valid]
    assert scores, "expected valid cells"
    assert max(scores) < 0.15, f"a genuinely uniform field should not produce a high-scoring cell: {scores}"
    assert float(np.max(heat)) < 0.15


def test_uniform_field_with_real_outlier_still_detects_it():
    cfg = load_config()
    cfg["anomaly"]["use_isolation_forest"] = False
    bgr = _uniform_field_with_noise()
    # A genuinely different patch -- bare soil colored, not just noise.
    bgr[10:50, 10:50] = (30, 90, 140)
    field = np.ones((160, 160), dtype=bool)
    label = np.full((160, 160), "CROP", dtype=object)
    emb = np.zeros((8, 8, 16), dtype=np.float32)

    cells, heat = score_frame(bgr, field, label, emb, cfg)

    scores = [c.appearance_anomaly_score for c in cells if c.valid]
    assert max(scores) > 0.3, "a real, distinct outlier patch must still be detected"


def test_min_z_range_threshold_gates_the_relative_ranking_blend():
    """Direct proof at the _to_unit() level, not just the full score_frame()
    integration: setting the gate to 0 (always blend) vs a high value
    (never blend) changes the output for a noise-only input."""
    from cropmerge.anomaly.spatial import _to_unit

    rng = np.random.default_rng(1)
    z = np.abs(rng.normal(0, 0.05, size=36))  # tiny noise-level z-scores
    valid = np.ones(36, dtype=bool)

    always_blend = _to_unit(z, valid, min_z_range=0.0)
    never_blend = _to_unit(z, valid, min_z_range=1.0)

    assert float(np.max(always_blend)) > 0.3, "old behavior: rank blend inflates the max cell"
    assert float(np.max(never_blend)) < 0.05, "fixed behavior: noise-level z stays near zero"
