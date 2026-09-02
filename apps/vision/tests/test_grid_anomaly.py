import numpy as np

from cropmerge.anomaly.grid import build_grid, normalize_scores, robust_baseline
from cropmerge.anomaly.spatial import score_frame
from cropmerge.anomaly.temporal import aggregate_zones, relative_location
from cropmerge.config import load_config
from cropmerge.pipeline.schemas import RelativeLocation


def test_grid_size():
    cells = build_grid(100, 200, 4, 5)
    assert len(cells) == 20


def test_normalize_constant():
    z = normalize_scores(np.ones(5))
    assert np.allclose(z, 0)


def test_robust_baseline():
    assert robust_baseline([1, 2, 100]) == 2


def test_relative_location():
    assert relative_location(0.1, 0.1) == RelativeLocation.NW
    assert relative_location(0.5, 0.5) == RelativeLocation.C
    assert relative_location(0.9, 0.9) == RelativeLocation.SE


def test_spatial_and_temporal():
    cfg = load_config()
    cfg["temporal"]["min_persistent_frames"] = 2
    cfg["anomaly"]["medium_threshold"] = 0.1
    h, w = 160, 160
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:] = (40, 160, 40)
    # anomaly corner
    bgr[10:50, 110:150] = (50, 90, 150)
    field = np.ones((h, w), dtype=bool)
    label = np.full((h, w), "CROP", dtype=object)
    label[10:50, 110:150] = "BARE_SOIL"
    emb = np.zeros((8, 8, 32), dtype=np.float32)
    # different embedding in NE cells
    emb[0:3, 5:8, 0] = 1.0
    emb[3:, :, 1] = 1.0
    cells, heat = score_frame(bgr, field, label, emb, cfg)
    assert heat.shape == (8, 8)
    assert any(c.valid for c in cells)

    frame_cells = [cells, cells, cells]
    zones = aggregate_zones(frame_cells, [0.0, 0.5, 1.0], [1.0, 1.0, 1.0], (h, w), cfg)
    assert isinstance(zones, list)
