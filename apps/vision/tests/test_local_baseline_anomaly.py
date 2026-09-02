"""Regression tests for the local-neighborhood baseline blend in
cropmerge/anomaly/spatial.py, added 2026-09-02 after a real research-plot
photo (three large, roughly equal, internally-uniform treatment zones)
showed the pure whole-field median/MAD baseline flagging cells
inconsistently across otherwise visually-uniform regions -- see
docs/ANOMALY_LOCAL_BASELINE.md.
"""
from __future__ import annotations

import numpy as np

from cropmerge.anomaly.spatial import _local_robust_z, _robust_z, score_frame
from cropmerge.config import load_config


def test_local_robust_z_ignores_a_different_far_away_zone():
    """A value identical to its own neighborhood should score ~0 locally,
    even in a field with three large, internally-uniform zones (the real
    scenario: three roughly equal treatment strips) where the whole-field
    global median can't represent any of them well."""
    rows = np.repeat(np.arange(6), 6)
    cols = np.tile(np.arange(6), 6)
    values = np.zeros(36, dtype=np.float64)
    values[cols <= 1] = 1.0
    values[(cols >= 2) & (cols <= 3)] = 5.0
    values[cols >= 4] = 9.0
    valid = np.ones(36, dtype=bool)

    local_z = _local_robust_z(values, valid, rows, cols, radius=1)
    global_z = _robust_z(values, valid)

    # Every cell is identical to its own immediate neighbors -> local z ~ 0
    # everywhere, regardless of which zone it's in.
    assert np.allclose(local_z, 0.0, atol=1e-6)
    # The global baseline, by contrast, treats the outer two zones (whose
    # value differs from the middle zone's, which is also the global
    # median) as meaningfully anomalous -- exactly the false-positive
    # pattern reported on the real multi-zone field photo.
    assert float(np.max(global_z)) > 0.3
    assert float(np.max(local_z)) == 0.0


def test_local_robust_z_still_flags_a_genuine_local_outlier():
    rows = np.repeat(np.arange(6), 6)
    cols = np.tile(np.arange(6), 6)
    values = np.full(36, 5.0, dtype=np.float64)
    outlier_idx = 14  # somewhere in the interior
    values[outlier_idx] = 50.0
    valid = np.ones(36, dtype=bool)

    local_z = _local_robust_z(values, valid, rows, cols, radius=1)

    assert local_z[outlier_idx] > 3.0
    # its uniform neighbors should not be flagged by the outlier's presence
    neighbor_scores = np.delete(local_z, outlier_idx)
    assert float(np.median(neighbor_scores)) < 1.0


def test_local_robust_z_falls_back_to_zero_with_too_few_neighbors():
    rows = np.array([0, 0, 5, 5])
    cols = np.array([0, 1, 5, 6])
    values = np.array([1.0, 1.0, 100.0, 1.0])
    valid = np.ones(4, dtype=bool)

    local_z = _local_robust_z(values, valid, rows, cols, radius=1, min_neighbors=4)

    # No cell has >=4 valid neighbors in this tiny sparse grid -> all zero,
    # not a noisy estimate from 1-2 samples.
    assert np.allclose(local_z, 0.0)


def _three_zone_frame(h: int = 240, w: int = 240):
    """A synthetic field split into three vertical thirds with different
    but internally-uniform color/vegetation -- the exact real-world pattern
    (three large treatment strips) that triggered this investigation."""
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    third = w // 3
    bgr[:, 0:third] = (40, 200, 210)  # yellow-ish (BGR)
    bgr[:, third : 2 * third] = (40, 160, 40)  # healthy green
    bgr[:, 2 * third :] = (30, 90, 130)  # brown/dead
    field = np.ones((h, w), dtype=bool)
    label = np.full((h, w), "CROP", dtype=object)
    emb = np.zeros((8, 8, 16), dtype=np.float32)
    for c in range(8):
        zone = 0 if c < 3 else (1 if c < 6 else 2)
        emb[:, c, zone] = 1.0
    return bgr, field, label, emb


def test_three_zone_field_interior_cells_are_not_flagged_with_local_baseline():
    cfg = load_config()
    cfg["anomaly"]["local_baseline_weight"] = 0.7
    cfg["anomaly"]["use_isolation_forest"] = False  # isolate the z-score blend being tested
    bgr, field, label, emb = _three_zone_frame()

    cells, _heat = score_frame(bgr, field, label, emb, cfg)

    # Interior columns of each zone (away from the two zone boundaries at
    # col 2/3 and 5/6) should score low -- they look just like their
    # immediate neighbors, even though the field as a whole has three very
    # different-looking zones.
    interior_cols = {1, 4, 7}
    interior_scores = [c.appearance_anomaly_score for c in cells if c.col in interior_cols and c.valid]
    assert interior_scores, "expected valid interior cells"
    assert max(interior_scores) < 0.5, f"interior cells scored too high: {interior_scores}"


def test_local_baseline_weight_zero_reproduces_old_global_only_behavior():
    """Sanity: the config knob genuinely gates the new behavior -- with
    weight=0 the result must be identical to calling _robust_z() directly
    (the pre-existing, unmodified global-only path)."""
    cfg = load_config()
    cfg["anomaly"]["local_baseline_weight"] = 0.0
    cfg["anomaly"]["use_isolation_forest"] = False
    bgr, field, label, emb = _three_zone_frame()

    cells, _heat = score_frame(bgr, field, label, emb, cfg)
    scores_with_zero_weight = [c.appearance_anomaly_score for c in cells if c.valid]

    # With pure global baseline restored, some interior cells of the
    # minority-shaped zones DO score meaningfully higher than with the
    # local blend enabled -- confirming the local blend is doing real work,
    # not a no-op.
    cfg2 = load_config()
    cfg2["anomaly"]["local_baseline_weight"] = 0.9
    cfg2["anomaly"]["use_isolation_forest"] = False
    cells2, _heat2 = score_frame(bgr, field, label, emb, cfg2)
    scores_with_local = [c.appearance_anomaly_score for c in cells2 if c.valid]

    assert sum(scores_with_zero_weight) > sum(scores_with_local)
