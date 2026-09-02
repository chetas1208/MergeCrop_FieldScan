"""
Observability router.

A pure gate that classifies how much agricultural detail is actually
resolvable in a frame/mask given cheap geometric evidence (row-geometry
coherence and line support -- see cropmerge/features/row_geometry.py), and
therefore which downstream claims are allowed:

  PLANT_RESOLVABLE -- individual plants are (claimed, by the caller-supplied
      plant-detection evidence) individually resolvable and countable.
      REQUIRED before calling cropmerge.features.spacing_metrics functions
      (Miss Index / Multiple Index / Quality-of-Feed Index / relative
      spacing irregularity). Grounded in Wang et al. 2023's framing that
      plant-level claims need sufficient resolution/growth stage -- see
      docs/FARMTECH_RESEARCH_BASIS.md.

  ROW_RESOLVABLE -- individual plants are not confirmed resolvable, but row
      structure (orientation + multiple consistent parallel row lines) is
      confidently detected. REQUIRED before row-continuity / "gap-in-the-row"
      claims (a gap requires a resolvable row on both sides of the empty
      span, per Wang et al. 2023 -- a bare-pixel run alone is not a gap
      claim).

  CANOPY_ONLY -- neither plants nor row structure are confidently resolved.
      Only coverage / fragmentation / soil-exposure evidence (e.g. ExG/VARI
      vegetation fraction, occupancy deficit) is valid at this tier -- no
      row-continuity or spacing claim may be made.

This module does not itself compute row_coherence / row_line_support_px /
num_consistent_rows -- see cropmerge.features.row_geometry.analyze_row_geometry,
which produces exactly these fields on its RowGeometryResult
(structure_tensor_coherence, row_line_support_px, num_consistent_rows).

Scope note: this is a pure classification function. It is NOT wired into
cropmerge/pipeline/processor.py yet (see docs/FARMTECH_RESEARCH_BASIS.md,
"Explicitly deferred") -- callers (both today's tests and a future pipeline
integration) are responsible for actually checking the returned tier before
invoking gated downstream code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ObservabilityTier = Literal["PLANT_RESOLVABLE", "ROW_RESOLVABLE", "CANOPY_ONLY"]


@dataclass(frozen=True)
class ObservabilityThresholds:
    """Named thresholds for the observability router. Mirrors
    configs/farmtech_v1.yaml `observability:` block; that YAML is the
    deployed source of truth, this dataclass is the code-side typed
    default. Keeping thresholds here (not scattered as magic numbers in
    classify_observability) so they can be reviewed/tuned in one place.
    """

    # ROW_RESOLVABLE requires the structure-tensor coherence (0..1, from
    # row_geometry.structure_tensor_orientation) to reach this level.
    min_row_coherence: float = 0.35

    # ROW_RESOLVABLE requires the strongest clustered Hough row line to have
    # at least this much summed pixel support (row_geometry.RowGeometryResult
    # .row_line_support_px).
    min_row_line_support_px: float = 60.0

    # ROW_RESOLVABLE requires at least this many distinct, angle/position-
    # consistent row-line clusters (row_geometry.RowGeometryResult
    # .num_consistent_rows) -- a single line is not enough to call "rows"
    # resolvable (could be a fence line, a shadow edge, road boundary, etc).
    min_consistent_rows: int = 2

    # PLANT_RESOLVABLE additionally requires per-row plant-level detection
    # evidence reaching these levels. No plant detector exists in this
    # codebase yet (see docs/FARMTECH_RESEARCH_BASIS.md) -- these thresholds
    # exist so the tier and its contract are already defined for whenever
    # one is wired in; until then this tier is reachable only if the caller
    # explicitly supplies qualifying plant-detection evidence.
    min_plant_detections_per_row: int = 3
    min_plant_detection_confidence: float = 0.6


def classify_observability(
    row_coherence: float,
    row_line_support_px: float,
    num_consistent_rows: int,
    *,
    plant_detection_count: int = 0,
    plant_detection_confidence: float = 0.0,
    thresholds: ObservabilityThresholds | None = None,
) -> ObservabilityTier:
    """Classify the observability tier for one frame/region.

    Args:
        row_coherence: structure-tensor coherence in [0, 1] (higher =
            stronger, more consistent local gradient orientation --
            evidence of real row structure vs. noise). See
            row_geometry.RowGeometryResult.structure_tensor_coherence.
        row_line_support_px: summed pixel length of the strongest
            angle/position-consistent Hough row-line cluster. See
            row_geometry.RowGeometryResult.row_line_support_px.
        num_consistent_rows: count of distinct row-line clusters meeting
            the row-geometry module's own min-support threshold. See
            row_geometry.RowGeometryResult.num_consistent_rows.
        plant_detection_count: (optional) number of individually resolved
            plant detections available for the row(s) in question. Defaults
            to 0 -- i.e. "no plant detector ran," which can never reach
            PLANT_RESOLVABLE, by design.
        plant_detection_confidence: (optional) confidence (0..1) associated
            with those plant detections.
        thresholds: override the default ObservabilityThresholds.

    Returns:
        "PLANT_RESOLVABLE", "ROW_RESOLVABLE", or "CANOPY_ONLY".
    """
    t = thresholds or ObservabilityThresholds()

    row_resolvable = (
        row_coherence >= t.min_row_coherence
        and row_line_support_px >= t.min_row_line_support_px
        and num_consistent_rows >= t.min_consistent_rows
    )

    plant_resolvable = (
        row_resolvable
        and plant_detection_count >= t.min_plant_detections_per_row
        and plant_detection_confidence >= t.min_plant_detection_confidence
    )

    if plant_resolvable:
        return "PLANT_RESOLVABLE"
    if row_resolvable:
        return "ROW_RESOLVABLE"
    return "CANOPY_ONLY"
