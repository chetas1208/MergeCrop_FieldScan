"""Real, constructed proof that passing a per-unit row angle
(unit_row_angles, from cropmerge.features.unit_row_geometry, threaded
through cropmerge.anomaly.structural.score_structural_cells) recovers gap
continuity detection that the single whole-frame row angle misses when the
frame's OVERALL row_visibility reads LOW -- dragged down by a large,
genuinely rowless residue/noise region sitting next to a small but clearly
row-structured crop block.
"""
from __future__ import annotations

import numpy as np

from cropmerge.anomaly.grid import build_grid
from cropmerge.anomaly.structural import score_structural_cells
from cropmerge.config import load_config
from cropmerge.features.management_units import segment_management_units
from cropmerge.features.unit_row_geometry import analyze_management_unit_rows

_H, _W = 300, 300
_CROP_W = 100
_MARGIN = 10
_N_STRIPES = 4
_ROWS = _COLS = 20


def _stripe_x_ranges() -> list[tuple[int, int]]:
    usable = _CROP_W - 2 * _MARGIN
    spacing = usable / _N_STRIPES
    ranges = []
    for i in range(_N_STRIPES):
        cx = int(_MARGIN + spacing * i + spacing / 2)
        ranges.append((max(0, cx - 3), min(_CROP_W, cx + 3)))
    return ranges


def _crop_rows_next_to_noisy_residue_with_a_real_gap():
    """Left: vertical crop-row stripes with a real gap cut into stripe #1's
    middle. Right: a large random-noise "residue" region, large and
    incoherent enough that the WHOLE-FRAME structure-tensor coherence reads
    LOW even though the crop block's own rows are perfectly clear."""
    bgr = np.zeros((_H, _W, 3), dtype=np.uint8)
    bgr[:, :_CROP_W] = (30, 170, 30)
    label = np.full((_H, _W), "CROP", dtype=object)
    label[:, _CROP_W:] = "BARE_SOIL"
    field = np.ones((_H, _W), dtype=bool)

    stripes = _stripe_x_ranges()
    crop_mask = np.zeros((_H, _W), dtype=bool)
    for x0, x1 in stripes:
        crop_mask[_MARGIN : _H - _MARGIN, x0:x1] = True

    gap_x0, gap_x1 = stripes[1]
    gap_y0, gap_y1 = 144, 156
    crop_mask[gap_y0:gap_y1, gap_x0:gap_x1] = False
    bgr[gap_y0:gap_y1, gap_x0:gap_x1] = (90, 140, 180)

    rng = np.random.default_rng(0)
    noise = rng.random((_H, _W - _CROP_W)) > 0.5
    bgr[:, _CROP_W:] = np.where(noise[..., None], (60, 110, 150), (110, 160, 190)).astype(np.uint8)

    return bgr, field, label, crop_mask


def _build_cells(field):
    cells = build_grid(_H, _W, _ROWS, _COLS)
    for c in cells:
        fm = field[c.y0 : c.y1, c.x0 : c.x1]
        c.field_fraction = float(np.mean(fm)) if fm.size else 0.0
        c.valid = c.field_fraction >= 0.25
        c.features = {"bare_soil": 0.0, "coverage_delta": 0.0}
    return cells


def _gap_cell(cells):
    """Cell (row=10, col=2) sits centered on the constructed gap -- verified
    directly against _sample_along_row/_gap_score_along_row while building
    this fixture (see Decisions.md, Phase 8 entry, for the real numbers)."""
    for c in cells:
        if c.row == 10 and c.col == 2:
            return c
    raise AssertionError("expected grid cell (10, 2) to exist")


def test_whole_frame_row_visibility_is_low_despite_clear_crop_rows():
    """Sanity precondition: the noisy residue block really does drag the
    frame's overall row-visibility reading down to LOW, which is exactly
    the situation unit_row_angles exists to route around."""
    cfg = load_config()
    bgr, field, label, crop_mask = _crop_rows_next_to_noisy_residue_with_a_real_gap()
    cells = _build_cells(field)

    _, sres = score_structural_cells(cells, bgr, field, label, crop_mask, cfg)

    assert sres.row_visibility == "LOW"


def test_unit_row_angle_recovers_gap_continuity_the_whole_frame_angle_misses():
    cfg = load_config()
    bgr, field, label, crop_mask = _crop_rows_next_to_noisy_residue_with_a_real_gap()
    seg = segment_management_units(bgr, field, label, rows=_ROWS, cols=_COLS)
    row_results = analyze_management_unit_rows(crop_mask, field, seg)
    unit_row_angles = {
        r.unit_id: r.row_geometry.dominant_angle_deg for r in row_results if r.row_geometry is not None
    }
    assert unit_row_angles, "expected the confident ACTIVE_CROP unit to yield a row angle"

    without_cells = _build_cells(field)
    without_cells, _ = score_structural_cells(
        without_cells, bgr, field, label, crop_mask, cfg, management_units=seg, unit_row_angles=None
    )
    gap_without = _gap_cell(without_cells)

    with_cells = _build_cells(field)
    with_cells, _ = score_structural_cells(
        with_cells, bgr, field, label, crop_mask, cfg, management_units=seg, unit_row_angles=unit_row_angles
    )
    gap_with = _gap_cell(with_cells)

    # Whole-frame LOW visibility gates continuity sampling off entirely --
    # the real gap is invisible without the per-unit angle.
    assert gap_without.features["continuity_evidence"] == 0.0
    assert gap_without.features["gap_length_score"] == 0.0

    # With the unit's own confident angle, the same real gap is detected.
    assert gap_with.features["continuity_evidence"] > 0.9
    assert gap_with.features["gap_length_score"] > 0.9
    assert gap_with.structural_anomaly_score > gap_without.structural_anomaly_score
