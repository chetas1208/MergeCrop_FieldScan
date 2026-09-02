"""Per-management-unit row geometry (Phase 4 of the "Make FarmTech
Load-Bearing" campaign).

Why per-unit, not whole-field: cropmerge/features/row_geometry.py's
analyze_row_geometry() already runs Hough + structure-tensor analysis on a
single binary mask -- but running it on the WHOLE field mixes crop rows
from an ACTIVE_CROP block with the (nonexistent, or differently-angled)
"rows" of a residue/soil/non-crop block, producing a meaningless blended
angle. Restricting the input mask to one management unit's footprint AND to
the same fine-grained crop-plant mask (`crop_mask`) cropmerge.anomaly.
structural already uses for its own row-visibility Hough crosscheck (NOT
the coarse per-region label_map, which carries no internal row/gap
structure) fixes that at the source.

Deliberately conservative per the campaign's explicit instruction ("never
mixing residue+crop"): row geometry is computed ONLY for units classified
ACTIVE_CROP with confidence at or above `min_confidence` -- every other
unit (RESIDUE_STUBBLE/BARE_SOIL/NON_CROP/UNKNOWN, or a low-confidence
ACTIVE_CROP call) is skipped outright with a real reason string, never
given a fabricated or best-effort row angle.

SHADOW-ONLY / STANDALONE for this round, same discipline as
management_units.py's own first commit: this module is independently
computable and testable but not yet wired into cropmerge/anomaly/structural.py
or the live scoring path. Per-unit local row baselines (Phase 5-9 of the
campaign) are real, valuable follow-on work that deserves its own
benchmarking against this foundation -- not done here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cropmerge.anomaly.grid import build_grid
from cropmerge.features.management_units import UnitSegmentationResult, UnitType
from cropmerge.features.row_geometry import RowGeometryConfig, RowGeometryResult, analyze_row_geometry

# A unit's own confidence (from unit_type_confidence()) must reach this
# before its row geometry is trusted enough to compute at all.
DEFAULT_MIN_UNIT_CONFIDENCE = 0.3

# A per-unit crop mask with fewer than this many nonzero pixels can't
# support a meaningful Hough/structure-tensor analysis (mirrors the
# >= 100 px floor management_units.compute_cell_features() already uses
# for its own per-cell structure-tensor call).
MIN_CROP_PIXELS = 100


@dataclass(frozen=True)
class UnitRowGeometry:
    unit_id: int
    row_geometry: RowGeometryResult | None
    skipped_reason: str | None = None


def _unit_crop_mask(
    crop_mask: np.ndarray,
    field_mask: np.ndarray,
    cell_unit_ids: list[list[int]],
    grid_cells,
    unit_id: int,
) -> np.ndarray:
    """Boolean mask, image-resolution, True only where a pixel is BOTH
    inside a cell belonging to `unit_id` AND actual crop-plant presence
    (the same fine-grained `crop_mask` cropmerge.anomaly.structural already
    uses for its own row-visibility Hough crosscheck -- see
    `mask = crop_mask.astype(bool) & field_mask.astype(bool)` in that
    module) -- never the whole cell rectangle, which can straddle a real
    crop/non-crop edge, and never the coarse per-region label_map, which
    carries no internal row/gap structure to detect."""
    base = crop_mask.astype(bool) & field_mask.astype(bool)
    mask = np.zeros(base.shape[:2], dtype=bool)
    for cell in grid_cells:
        if cell_unit_ids[cell.row][cell.col] != unit_id:
            continue
        region = mask[cell.y0 : cell.y1, cell.x0 : cell.x1]
        region |= base[cell.y0 : cell.y1, cell.x0 : cell.x1]
    return mask


def analyze_management_unit_rows(
    crop_mask: np.ndarray,
    field_mask: np.ndarray,
    segmentation: UnitSegmentationResult,
    *,
    min_confidence: float = DEFAULT_MIN_UNIT_CONFIDENCE,
    row_cfg: RowGeometryConfig | None = None,
) -> list[UnitRowGeometry]:
    """Run row-geometry analysis independently for each management unit,
    restricted to confident ACTIVE_CROP units only. Returns one
    UnitRowGeometry per unit in `segmentation.units` (same order), so
    callers can always zip() them back together -- skipped units carry
    `row_geometry=None` and a real, specific `skipped_reason`."""
    h, w = field_mask.shape[:2]
    grid_cells = build_grid(h, w, segmentation.rows, segmentation.cols)

    results: list[UnitRowGeometry] = []
    for unit in segmentation.units:
        if unit.unit_type != UnitType.ACTIVE_CROP:
            results.append(
                UnitRowGeometry(
                    unit_id=unit.unit_id,
                    row_geometry=None,
                    skipped_reason=f"unit_type={unit.unit_type.value}, not ACTIVE_CROP",
                )
            )
            continue
        if unit.confidence < min_confidence:
            results.append(
                UnitRowGeometry(
                    unit_id=unit.unit_id,
                    row_geometry=None,
                    skipped_reason=f"confidence={unit.confidence:.2f} below min_confidence={min_confidence}",
                )
            )
            continue

        unit_mask = _unit_crop_mask(crop_mask, field_mask, segmentation.cell_unit_ids, grid_cells, unit.unit_id)
        if int(np.count_nonzero(unit_mask)) < MIN_CROP_PIXELS:
            results.append(
                UnitRowGeometry(
                    unit_id=unit.unit_id,
                    row_geometry=None,
                    skipped_reason=f"only {int(np.count_nonzero(unit_mask))}px of crop-plant area, "
                    f"below the {MIN_CROP_PIXELS}px floor",
                )
            )
            continue

        geometry = analyze_row_geometry(unit_mask, row_cfg)
        results.append(UnitRowGeometry(unit_id=unit.unit_id, row_geometry=geometry, skipped_reason=None))

    return results
