import numpy as np

from cropmerge.anomaly.spatial import score_frame
from cropmerge.anomaly.structural import score_structural_cells
from cropmerge.config import load_config
from cropmerge.pipeline.frame_analysis import analyze_single_frame


class _FakeEmbedder:
    """Deterministic stand-in so this test isn't coupled to a real
    embedding backend — only checks analyze_single_frame's own fusion
    logic against the same score_frame/score_structural_cells calls
    processor.py used to make inline before the extraction."""

    def embed_tiles(self, bgr, mask, rows, cols):
        emb = np.zeros((rows, cols, 8), dtype=np.float32)
        emb[0:3, 5:8, 0] = 1.0
        emb[3:, :, 1] = 1.0
        return emb


def _synthetic_frame():
    h, w = 160, 160
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:] = (40, 160, 40)
    bgr[10:50, 110:150] = (50, 90, 150)
    field = np.ones((h, w), dtype=bool)
    label = np.full((h, w), "CROP", dtype=object)
    label[10:50, 110:150] = "BARE_SOIL"
    crop_mask = field.copy()
    return bgr, field, label, crop_mask


def test_analyze_single_frame_matches_direct_calls():
    cfg = load_config()
    bgr, field, label, crop_mask = _synthetic_frame()
    embedder = _FakeEmbedder()

    cells, heat, sres = analyze_single_frame(bgr, field, label, crop_mask, embedder, cfg, 8, 8)

    # Reproduce the pre-extraction inline sequence directly and compare.
    emb = embedder.embed_tiles(bgr, field, 8, 8)
    expected_cells, expected_heat = score_frame(bgr, field, label, emb, cfg)
    expected_cells, expected_sres = score_structural_cells(expected_cells, bgr, field, label, crop_mask, cfg)
    for c in expected_cells:
        c.anomaly_score = float(max(c.appearance_anomaly_score, c.structural_anomaly_score))
        c.row_visibility = expected_sres.row_visibility

    assert np.allclose(heat, expected_heat)
    assert sres.row_visibility == expected_sres.row_visibility
    assert len(cells) == len(expected_cells)
    for got, want in zip(cells, expected_cells):
        assert got.anomaly_score == want.anomaly_score
        assert got.appearance_anomaly_score == want.appearance_anomaly_score
        assert got.structural_anomaly_score == want.structural_anomaly_score


def test_analyze_single_frame_applies_quality_weight():
    cfg = load_config()
    bgr, field, label, crop_mask = _synthetic_frame()
    embedder = _FakeEmbedder()

    full_cells, full_heat, _ = analyze_single_frame(bgr, field, label, crop_mask, embedder, cfg, 8, 8, quality_weight=1.0)
    dim_cells, dim_heat, _ = analyze_single_frame(bgr, field, label, crop_mask, embedder, cfg, 8, 8, quality_weight=0.5)

    assert np.allclose(dim_heat, full_heat * 0.5)
    for full_c, dim_c in zip(full_cells, dim_cells):
        assert dim_c.anomaly_score == full_c.anomaly_score * 0.5
