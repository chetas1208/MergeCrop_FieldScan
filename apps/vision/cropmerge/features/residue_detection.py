"""Cheap heuristic residue/stubble vs. bare-soil vs. living-vegetation
evidence.

Motivated by real feedback: harvested residue/stubble was being classified
as CROP by the existing pipeline, which made downstream crop-coverage
numbers on that block meaningless (a residue block reading as high crop
coverage tells a farmer nothing useful).

HONESTY CONSTRAINT, stated because it shapes every threshold below:
separating dry plant residue from bare soil using RGB color alone is
genuinely uncertain -- both are commonly tan/brown, and the actual
distinguishing cue (residue's fine linear/directional texture from cut
stalks lying on the ground, vs. bare soil's comparatively smoother/more
uniform surface) is a weak, noisy signal from a single RGB frame, not a
reliable classifier. This module is deliberately conservative: when
vegetation, chromaticity, and texture evidence don't clearly agree, it
returns UNKNOWN rather than forcing a confident residue-vs-soil call. Never
present UNKNOWN cells as confidently classified in any downstream UI.

Not a trained classifier. Not validated against ground-truth residue/soil
labels. An engineering heuristic pending real field validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from cropmerge.features.rgb_indices import excess_green, vegetation_mask
from cropmerge.features.texture import texture_features


class ResidueClass(str, Enum):
    LIVING_VEGETATION = "living_vegetation"
    LIKELY_RESIDUE = "likely_residue"
    LIKELY_BARE_SOIL = "likely_bare_soil"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResidueThresholds:
    """Engineering heuristics, not validated constants -- tune from real
    field review, same caveat as every other threshold in this codebase."""

    vegetation_exg_threshold: float = 0.05
    living_vegetation_min_frac: float = 0.15  # region-level: ExG-positive pixel fraction
    # HSV hue range (OpenCV 0-179 scale) considered tan/brown/dry-material-like.
    brown_hue_low: int = 8
    brown_hue_high: int = 32
    brown_saturation_min: int = 20
    brown_saturation_max: int = 160
    brown_value_min: int = 60
    brown_min_frac: float = 0.4  # fraction of region pixels needing brown/tan hue
    # Texture split between residue (higher, from linear stalk edges) and
    # smoother bare soil.
    residue_edge_density_min: float = 0.06
    soil_edge_density_max: float = 0.03


@dataclass(frozen=True)
class ResidueEvidence:
    classification: ResidueClass
    living_vegetation_fraction: float
    brown_hue_fraction: float
    edge_density: float
    confidence_note: str


def classify_residue_evidence(
    bgr: np.ndarray,
    mask: np.ndarray | None = None,
    thresholds: ResidueThresholds | None = None,
) -> ResidueEvidence:
    t = thresholds or ResidueThresholds()
    h, w = bgr.shape[:2]
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    mask = mask.astype(bool)

    if not np.any(mask):
        return ResidueEvidence(
            classification=ResidueClass.UNKNOWN,
            living_vegetation_fraction=0.0,
            brown_hue_fraction=0.0,
            edge_density=0.0,
            confidence_note="empty region",
        )

    exg = excess_green(bgr)
    veg = vegetation_mask(exg, t.vegetation_exg_threshold)
    living_fraction = float(np.mean(veg[mask]))

    if living_fraction >= t.living_vegetation_min_frac:
        return ResidueEvidence(
            classification=ResidueClass.LIVING_VEGETATION,
            living_vegetation_fraction=living_fraction,
            brown_hue_fraction=0.0,
            edge_density=0.0,
            confidence_note="ExG-positive vegetation fraction above threshold",
        )

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    brown = (
        (hue >= t.brown_hue_low)
        & (hue <= t.brown_hue_high)
        & (sat >= t.brown_saturation_min)
        & (sat <= t.brown_saturation_max)
        & (val >= t.brown_value_min)
    )
    brown_fraction = float(np.mean(brown[mask]))

    tex = texture_features(bgr, mask)
    edge_density = tex["edge_density"]

    if brown_fraction < t.brown_min_frac:
        return ResidueEvidence(
            classification=ResidueClass.UNKNOWN,
            living_vegetation_fraction=living_fraction,
            brown_hue_fraction=brown_fraction,
            edge_density=edge_density,
            confidence_note="low living-vegetation signal but not clearly tan/brown either",
        )

    if edge_density >= t.residue_edge_density_min:
        return ResidueEvidence(
            classification=ResidueClass.LIKELY_RESIDUE,
            living_vegetation_fraction=living_fraction,
            brown_hue_fraction=brown_fraction,
            edge_density=edge_density,
            confidence_note="tan/brown hue with texture consistent with cut stalks/residue",
        )
    if edge_density <= t.soil_edge_density_max:
        return ResidueEvidence(
            classification=ResidueClass.LIKELY_BARE_SOIL,
            living_vegetation_fraction=living_fraction,
            brown_hue_fraction=brown_fraction,
            edge_density=edge_density,
            confidence_note="tan/brown hue with smooth texture consistent with bare soil",
        )
    return ResidueEvidence(
        classification=ResidueClass.UNKNOWN,
        living_vegetation_fraction=living_fraction,
        brown_hue_fraction=brown_fraction,
        edge_density=edge_density,
        confidence_note="tan/brown hue but texture evidence is ambiguous between residue and bare soil",
    )
