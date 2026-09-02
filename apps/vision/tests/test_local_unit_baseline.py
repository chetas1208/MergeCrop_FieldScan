"""Real, constructed proof that the per-management-unit local occupancy
baseline (cropmerge.anomaly.structural._local_unit_baselines, wired into
score_structural_cells via the `management_units` param) finds a real gap
inside a small healthy crop block that a whole-field baseline MISSES when
a much larger low-occupancy residue block sits next to it and drags the
global median down.
"""
from __future__ import annotations

import numpy as np

from cropmerge.anomaly.grid import build_grid
from cropmerge.anomaly.structural import score_structural_cells
from cropmerge.config import load_config
from cropmerge.features.management_units import segment_management_units


def _small_crop_block_next_to_large_residue_block(h: int = 300, w: int = 300):
    """Left third: healthy green crop, EXCEPT one small interior patch that
    is a real bare-soil gap. Right two-thirds: a much larger textured
    residue block (near-zero occupancy) -- big enough to drag a whole-field
    median occupancy baseline down near zero, well below the crop block's
    own real baseline (~0.9)."""
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    crop_w = w // 3
    bgr[:, :crop_w] = (30, 170, 30)  # BGR green, healthy crop
    # Real interior gap: a soil-colored rectangle inside the crop block.
    bgr[120:180, 20:80] = (90, 140, 180)
    for y in range(h):
        shade = 1.0 if (y // 3) % 2 == 0 else 0.55
        bgr[y, crop_w:] = tuple(int(c * shade) for c in (90, 140, 180))  # striped tan residue

    field = np.ones((h, w), dtype=bool)
    label = np.full((h, w), "CROP", dtype=object)
    label[:, crop_w:] = "BARE_SOIL"
    crop_mask = label == "CROP"
    return bgr, field, label, crop_mask


def _build_cells(bgr, field):
    h, w = bgr.shape[:2]
    cells = build_grid(h, w, 10, 10)
    for c in cells:
        fm = field[c.y0 : c.y1, c.x0 : c.x1]
        c.field_fraction = float(np.mean(fm)) if fm.size else 0.0
        c.valid = c.field_fraction >= 0.25
        c.features = {"bare_soil": 0.0, "coverage_delta": 0.0}
    return cells


def _gap_cell(cells):
    """The cell whose footprint covers the interior gap patch [120:180, 20:80]."""
    for c in cells:
        if c.y0 <= 150 < c.y1 and c.x0 <= 50 < c.x1:
            return c
    raise AssertionError("no cell covers the constructed gap patch")


def test_local_unit_baseline_flags_a_gap_the_global_baseline_misses():
    cfg = load_config()
    bgr, field, label, crop_mask = _small_crop_block_next_to_large_residue_block()

    seg = segment_management_units(bgr, field, label, rows=10, cols=10)

    cells_without_units = _build_cells(bgr, field)
    cells_without_units, _ = score_structural_cells(
        cells_without_units, bgr, field, label, crop_mask, cfg, management_units=None
    )
    gap_without_units = _gap_cell(cells_without_units)

    cells_with_units = _build_cells(bgr, field)
    cells_with_units, _ = score_structural_cells(
        cells_with_units, bgr, field, label, crop_mask, cfg, management_units=seg
    )
    gap_with_units = _gap_cell(cells_with_units)

    # The whole-field median is dragged toward the (much larger) near-zero
    # residue block, so the gap's occupancy deficit relative to it is
    # small/near-zero -- exactly the false-negative this campaign phase
    # exists to fix.
    assert gap_without_units.features["row_occupancy_deficit"] < 0.35

    # Against the crop unit's OWN local baseline, the same real gap reads
    # as a large, correctly-detected deficit.
    assert gap_with_units.features["row_occupancy_deficit"] > 0.5
    assert gap_with_units.structural_anomaly_score > gap_without_units.structural_anomaly_score


def test_healthy_crop_cells_are_not_penalized_by_local_baseline():
    """A local baseline must not manufacture false positives on cells that
    ARE representative of their own unit -- only real outliers within the
    unit should score high."""
    cfg = load_config()
    bgr, field, label, crop_mask = _small_crop_block_next_to_large_residue_block()
    seg = segment_management_units(bgr, field, label, rows=10, cols=10)

    cells = _build_cells(bgr, field)
    cells, _ = score_structural_cells(cells, bgr, field, label, crop_mask, cfg, management_units=seg)

    healthy = [
        c
        for c in cells
        if c.valid and c.x1 <= 100 and not (c.y0 <= 150 < c.y1 and c.x0 <= 50 < c.x1)
    ]
    assert healthy
    assert all(c.structural_anomaly_score < 0.4 for c in healthy)


def test_residue_block_never_gets_a_crop_style_local_baseline():
    """Residue-block cells must never be assigned a local baseline from an
    ACTIVE_CROP unit -- they simply keep the whole-field fallback."""
    cfg = load_config()
    bgr, field, label, crop_mask = _small_crop_block_next_to_large_residue_block()
    seg = segment_management_units(bgr, field, label, rows=10, cols=10)

    cells = _build_cells(bgr, field)
    cells, _ = score_structural_cells(cells, bgr, field, label, crop_mask, cfg, management_units=seg)

    residue_cells = [c for c in cells if c.valid and c.x0 >= 100]
    assert residue_cells
    # None of these should carry a spuriously high occupancy-deficit-driven
    # score purely from being compared to a crop unit's high baseline.
    assert all(c.features["row_occupancy_deficit"] < 0.3 for c in residue_cells)
