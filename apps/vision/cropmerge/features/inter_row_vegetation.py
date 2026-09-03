"""Row-aware inter-row vegetation classification (close-canopy /
visible-inter-row-vegetation slice of the "Make FarmTech Load-Bearing"
campaign).

Why this exists on top of cropmerge.features.weed_pressure's
understory_vegetation_fraction: that module classifies vegetation-in-soil
using the coarse whole-image BARE_SOIL label, which mixes real inter-row
space with headlands, gaps, and any other non-crop-labelled soil. This
module instead uses the ACTUAL detected row-line geometry
(cropmerge.features.row_geometry, run per confident ACTIVE_CROP management
unit via cropmerge.features.unit_row_geometry) to classify each pixel by
its perpendicular distance to the nearest detected row line -- "on-row" vs
"inter-row" -- which is what a genuine "visible vegetation growing between
the crop rows" claim actually requires.

HARD GATE, same physical-limitation caveat as weed_pressure.py: this is
only ever computed for a unit whose row geometry was confidently detected
(unit_row_geometry.analyze_management_unit_rows already enforces
ACTIVE_CROP + confidence + pixel-count gates before returning a
RowGeometryResult at all -- see analyze_inter_row_vegetation's own None
return below for the same gate applied here). A unit with no confident row
geometry gets NO inter-row classification, never a fabricated or guessed
row band. NEVER call the resulting metric "weed density" in UI copy or
docs -- it is a vegetation-PRESENCE proxy in the inter-row space only, with
zero species or weed-vs-volunteer-crop discrimination.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cropmerge.features.rgb_indices import excess_green, vegetation_mask
from cropmerge.features.row_geometry import RowGeometryResult

# Perpendicular half-width (pixels, at whatever resolution the caller's
# unit_mask/row_geometry offsets are in) treated as "on the row" around
# each detected row-line's offset. A prior, not a calibrated field
# measurement -- no real row-width/plant-spacing ground truth exists yet
# to derive this from (see management_units.py's own calibration-data
# docstring for the same honesty pattern). Tune once real field row-width
# data exists.
DEFAULT_ROW_BAND_HALF_WIDTH_PX = 15.0


@dataclass(frozen=True)
class InterRowVegetationResult:
    row_band_fraction: float  # fraction of the unit's footprint classified "on-row"
    inter_row_fraction: float  # fraction classified "inter-row"
    inter_row_vegetation_fraction: float  # of the inter-row area only, fraction showing vegetation signal
    visible_inter_row_vegetation_fraction_of_unit: float  # same vegetated inter-row area, as a fraction of the WHOLE unit


def classify_row_band(
    unit_mask: np.ndarray,
    row_geometry: RowGeometryResult,
    *,
    row_band_half_width_px: float = DEFAULT_ROW_BAND_HALF_WIDTH_PX,
) -> np.ndarray:
    """Boolean mask, same shape as unit_mask, True where a pixel (within
    unit_mask) is within `row_band_half_width_px` of the nearest detected
    row line's perpendicular offset. Empty mask (never a guess) when the
    row geometry has no dominant angle or no detected row-line clusters."""
    if row_geometry.dominant_angle_deg is None or not row_geometry.row_clusters:
        return np.zeros_like(unit_mask, dtype=bool)

    theta = np.radians(row_geometry.dominant_angle_deg)
    nx, ny = -np.sin(theta), np.cos(theta)  # unit normal to the row direction, matches row_geometry's own convention

    ys, xs = np.nonzero(unit_mask)
    band_mask = np.zeros_like(unit_mask, dtype=bool)
    if ys.size == 0:
        return band_mask

    proj = xs.astype(np.float64) * nx + ys.astype(np.float64) * ny
    offsets = np.array([c.offset_px for c in row_geometry.row_clusters], dtype=np.float64)
    dist = np.min(np.abs(proj[:, None] - offsets[None, :]), axis=1)
    on_row = dist <= row_band_half_width_px

    band_mask[ys[on_row], xs[on_row]] = True
    return band_mask


def analyze_inter_row_vegetation(
    bgr: np.ndarray,
    unit_mask: np.ndarray,
    row_geometry: RowGeometryResult,
    *,
    row_band_half_width_px: float = DEFAULT_ROW_BAND_HALF_WIDTH_PX,
    exg_threshold: float = 0.05,
) -> InterRowVegetationResult | None:
    """Returns None (never a fabricated result) when the unit's row
    geometry has no dominant angle or no detected row-line clusters, or the
    unit footprint is empty -- the same confidence gate
    unit_row_geometry.analyze_management_unit_rows already applies before a
    RowGeometryResult reaches this function at all."""
    if row_geometry.dominant_angle_deg is None or not row_geometry.row_clusters:
        return None
    if not np.any(unit_mask):
        return None

    row_band = classify_row_band(unit_mask, row_geometry, row_band_half_width_px=row_band_half_width_px)
    inter_row = unit_mask & ~row_band

    unit_area = int(np.count_nonzero(unit_mask))
    row_band_fraction = float(np.count_nonzero(row_band)) / unit_area
    inter_row_fraction = float(np.count_nonzero(inter_row)) / unit_area

    if not np.any(inter_row):
        return InterRowVegetationResult(
            row_band_fraction=row_band_fraction,
            inter_row_fraction=inter_row_fraction,
            inter_row_vegetation_fraction=0.0,
            visible_inter_row_vegetation_fraction_of_unit=0.0,
        )

    exg = excess_green(bgr)
    veg = vegetation_mask(exg, exg_threshold)
    vegetated_inter_row = veg & inter_row

    return InterRowVegetationResult(
        row_band_fraction=row_band_fraction,
        inter_row_fraction=inter_row_fraction,
        inter_row_vegetation_fraction=float(np.mean(vegetated_inter_row[inter_row])),
        visible_inter_row_vegetation_fraction_of_unit=float(np.count_nonzero(vegetated_inter_row)) / unit_area,
    )
