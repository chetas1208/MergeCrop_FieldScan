"""Understory/inter-row vegetation proxy — a real, honest, physically-
grounded weed-pressure signal.

HARD PHYSICAL LIMITATION, stated plainly because it constrains everything
here: RGB overhead imagery cannot see through a closed crop canopy. This
module can only detect vegetation growing in areas the segmentation did
NOT classify as crop canopy (bare soil, gaps, inter-row space) -- it
CANNOT measure weeds growing underneath established crop foliage, which
is not observable in visible-light overhead imagery at all, regardless of
model quality. Most useful before canopy closure or where rows are
visibly gapped. Never call this "weed density under the crop" or
"sub-canopy weed detection" in UI copy or docs -- that overclaims what is
physically measurable here.
"""

from __future__ import annotations

import numpy as np

from cropmerge.features.rgb_indices import excess_green, vegetation_mask
from cropmerge.pipeline.schemas import SemanticClass


def understory_vegetation_fraction(
    bgr: np.ndarray,
    field_mask: np.ndarray,
    label_map: np.ndarray,
    exg_threshold: float = 0.05,
) -> dict[str, float]:
    """Fraction of BARE_SOIL-classified field area that still shows
    vegetation signal (ExG above threshold) -- candidate weed/volunteer
    growth in gaps and inter-row space, not underneath canopy.

    Returns:
        soilFraction: fraction of the analyzable field classified BARE_SOIL.
        understoryVegetationFraction: of that soil area, the fraction that
            still shows vegetation signal (0 when there's no soil to check).
        vegetatedSoilFractionOfField: the same vegetated-soil area, but as a
            fraction of the WHOLE field (not just the soil portion) -- useful
            for comparing across observations with different soil exposure.
    """
    field = field_mask.astype(bool)
    if not np.any(field):
        return {"soilFraction": 0.0, "understoryVegetationFraction": 0.0, "vegetatedSoilFractionOfField": 0.0}

    soil_mask = (label_map == SemanticClass.BARE_SOIL.value) & field
    soil_fraction = float(np.mean(soil_mask[field]))
    if not np.any(soil_mask):
        return {"soilFraction": soil_fraction, "understoryVegetationFraction": 0.0, "vegetatedSoilFractionOfField": 0.0}

    exg = excess_green(bgr)
    veg = vegetation_mask(exg, exg_threshold)
    vegetated_soil = veg & soil_mask

    return {
        "soilFraction": soil_fraction,
        "understoryVegetationFraction": float(np.mean(vegetated_soil[soil_mask])),
        "vegetatedSoilFractionOfField": float(np.mean(vegetated_soil[field])),
    }
