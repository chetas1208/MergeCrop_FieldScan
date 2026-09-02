"""
Planting-uniformity / plant-spacing metrics.

Classical formulation from planter-uniformity literature (commonly cited as
Kachman & Smith, "Alternative measures of accuracy in plant spacing for
planters using single seed metering," Trans. ASAE 1995 -- see
docs/FARMTECH_RESEARCH_BASIS.md for the citation-confidence note), plus a
FieldScan-specific "relative spacing irregularity" fallback for when the
nominal (intended) spacing is not known.

CONTRACT -- READ BEFORE CALLING:
Every function below assumes the *caller* has already confirmed
cropmerge.features.observability.classify_observability(...) returned
"PLANT_RESOLVABLE" for the imagery the input positions/spacings came from.
Miss Index / Multiple Index / Quality-of-Feed Index are only meaningful when
individual plants are reliably, individually resolvable (Wang et al. 2023 --
see docs/FARMTECH_RESEARCH_BASIS.md) and a nominal spacing is actually known.
This module contains no plant detector and has no way to check the
observability tier itself -- it operates purely on numbers the caller
supplies. Do NOT call these functions on spacings derived from unresolved,
low-confidence, or canopy-only imagery; the numbers will look precise
without being backed by resolvable plants.

This module does not implement a plant detector (out of scope for this
task -- see docs/FARMTECH_RESEARCH_BASIS.md, "Explicitly deferred").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlantingUniformityResult:
    miss_index: float
    multiple_index: float
    quality_of_feed_index: float
    nominal_spacing: float
    n: int


@dataclass(frozen=True)
class RelativeSpacingIrregularity:
    """FieldScan-specific fallback result when nominal spacing is unknown.
    NOT part of the Kachman & Smith formulation -- uses the empirical
    median interplant distance in place of an agronomic nominal spacing, so
    it answers "how irregular is this stand relative to itself," not "how
    close is this stand to its intended spacing."."""

    median_spacing: float
    large_gap_fraction_pct: float  # spacings > 1.5 * median_spacing
    tight_cluster_fraction_pct: float  # spacings <= 0.5 * median_spacing
    coefficient_of_variation: float  # std/median, unitless dispersion
    n: int


def interplant_distances(plant_center_positions: np.ndarray) -> np.ndarray:
    """Convert plant-center positions along a single row (e.g. x-pixel
    coordinates, or any consistent 1D linear-unit coordinate; positions
    need not be pre-sorted) into consecutive interplant distances.

    Returns an empty array when fewer than 2 positions are given -- a
    single (or zero) plant position has no interplant distance to report.
    """
    x = np.sort(np.asarray(plant_center_positions, dtype=np.float64))
    if x.size < 2:
        return np.zeros(0, dtype=np.float64)
    return np.diff(x)


def miss_index(spacings: np.ndarray, nominal_spacing: float) -> float:
    """Kachman & Smith Miss Index: percentage of interplant spacings more
    than 1.5x the nominal spacing (a "miss" -- an under-planted gap).

    Miss Index = 100 * count(spacing > 1.5 * nominal) / N
    """
    s = np.asarray(spacings, dtype=np.float64)
    if s.size == 0 or nominal_spacing <= 0:
        return 0.0
    return float(100.0 * np.count_nonzero(s > 1.5 * nominal_spacing) / s.size)


def multiple_index(spacings: np.ndarray, nominal_spacing: float) -> float:
    """Kachman & Smith Multiple Index: percentage of interplant spacings at
    most 0.5x the nominal spacing (a "multiple" -- two+ plants effectively
    at one station).

    Multiple Index = 100 * count(spacing <= 0.5 * nominal) / N
    """
    s = np.asarray(spacings, dtype=np.float64)
    if s.size == 0 or nominal_spacing <= 0:
        return 0.0
    return float(100.0 * np.count_nonzero(s <= 0.5 * nominal_spacing) / s.size)


def quality_of_feed_index(spacings: np.ndarray, nominal_spacing: float) -> float:
    """Kachman & Smith Quality-of-Feed Index (QFI):

    QFI = 100 - Miss Index - Multiple Index

    100 is a perfectly uniform stand at the nominal spacing; lower values
    indicate more misses and/or multiples.
    """
    return 100.0 - miss_index(spacings, nominal_spacing) - multiple_index(spacings, nominal_spacing)


def planting_uniformity(spacings: np.ndarray, nominal_spacing: float) -> PlantingUniformityResult:
    """Bundle Miss/Multiple/QFI for a known nominal spacing. Requires
    PLANT_RESOLVABLE observability and a known nominal spacing -- see module
    docstring."""
    s = np.asarray(spacings, dtype=np.float64)
    return PlantingUniformityResult(
        miss_index=miss_index(s, nominal_spacing),
        multiple_index=multiple_index(s, nominal_spacing),
        quality_of_feed_index=quality_of_feed_index(s, nominal_spacing),
        nominal_spacing=float(nominal_spacing),
        n=int(s.size),
    )


def relative_spacing_irregularity(spacings: np.ndarray) -> RelativeSpacingIrregularity:
    """Fallback planting-uniformity summary for when the nominal (intended)
    spacing is NOT known. Uses median(spacings) in place of a nominal
    spacing and applies the same 1.5x / 0.5x Kachman & Smith multipliers,
    plus a coefficient of variation. This is a FieldScan-specific
    substitution, not the original Kachman & Smith metric (which requires a
    known, externally-supplied nominal spacing) -- still requires
    PLANT_RESOLVABLE observability, since it is meaningless without
    individually resolvable plants.
    """
    s = np.asarray(spacings, dtype=np.float64)
    if s.size == 0:
        return RelativeSpacingIrregularity(0.0, 0.0, 0.0, 0.0, 0)
    median = float(np.median(s))
    if median <= 0:
        return RelativeSpacingIrregularity(median, 0.0, 0.0, 0.0, int(s.size))
    large = float(100.0 * np.count_nonzero(s > 1.5 * median) / s.size)
    tight = float(100.0 * np.count_nonzero(s <= 0.5 * median) / s.size)
    cv = float(np.std(s) / median)
    return RelativeSpacingIrregularity(
        median_spacing=median,
        large_gap_fraction_pct=large,
        tight_cluster_fraction_pct=tight,
        coefficient_of_variation=cv,
        n=int(s.size),
    )
