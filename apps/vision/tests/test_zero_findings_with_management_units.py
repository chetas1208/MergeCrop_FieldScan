"""End-to-end proof (Phase 9 of the "Make FarmTech Load-Bearing" campaign:
"absolute, non-rank-based inspection thresholds preserving zero findings
must be possible") that today's full stack -- appearance scoring, the
Phase 3 unit-type zone-label gate, the Phase 5-7 local per-unit occupancy
baseline, and the Phase 8 per-unit row angle -- together still produce ZERO
Inspection Areas on a genuinely uniform, healthy field across several
frames. cropmerge.anomaly.explain.review_priority() itself is already
absolute (fixed cfg thresholds, no percentile/rank comparison against other
zones in the same frame or video -- see its source), so this test's real
job is confirming none of the NEW management-unit machinery accidentally
manufactures a finding where the appearance-side fix (see
test_zero_findings_possible.py) already proved there shouldn't be one.
"""
from __future__ import annotations

import numpy as np

from cropmerge.anomaly.temporal import aggregate_zones
from cropmerge.config import load_config
from cropmerge.pipeline.frame_analysis import analyze_single_frame


class _ZeroEmbedder:
    """Identical (zero) embeddings everywhere -- isolates the z-score/
    ranking and local-baseline machinery from any real embedding-backend
    variance, matching test_zero_findings_possible.py's own approach."""

    def embed_tiles(self, bgr, mask, rows, cols):
        return np.zeros((rows, cols, 16), dtype=np.float32)


def _uniform_healthy_frame(seed: int, h: int = 160, w: int = 160) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), (40, 150, 40), dtype=np.float32)  # uniform green (BGR)
    noise = rng.normal(0, 1.5, size=base.shape)  # tiny sensor-noise-level variation only
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def test_zero_findings_still_reachable_with_management_units_and_local_baseline():
    cfg = load_config()
    cfg["anomaly"]["use_isolation_forest"] = False
    h, w = 160, 160
    field = np.ones((h, w), dtype=bool)
    label = np.full((h, w), "CROP", dtype=object)
    crop_mask = field.copy()
    embedder = _ZeroEmbedder()

    frame_cells = []
    timestamps = []
    quality_weights = []
    for i in range(5):
        bgr = _uniform_healthy_frame(seed=i, h=h, w=w)
        cells, _heat, _sres = analyze_single_frame(bgr, field, label, crop_mask, embedder, cfg, 8, 8)
        frame_cells.append(cells)
        timestamps.append(float(i))
        quality_weights.append(1.0)

    zones = aggregate_zones(frame_cells, timestamps, quality_weights, (h, w), cfg)

    assert zones == [], f"a genuinely uniform, healthy field must produce zero Inspection Areas, got: {zones}"
