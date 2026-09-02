"""Evidence-family consensus gate for Inspection Zone priority.

Motivated by real user feedback on a real field photo: some flagged regions
looked no different from unflagged ones, and manual review showed the flagged
ones were often "appearance differs from baseline" alone, with no
corroborating structural evidence (no row-continuity break, no gap, no
fragmentation). A single fused anomaly score treats "the color/texture/DINO
embedding looks a bit different" the same as "the crop canopy visibly breaks
here" -- those deserve different confidence, not the same threshold.

Architecture note (why this is a 2-family gate, not the 4-family split a
recommendation elsewhere in this campaign sketched out): cropmerge/anomaly
/spatial.py's score_frame() already fuses DINO + color + texture + coverage +
vegetation into ONE appearance_anomaly_score -- there is no separately
tracked "coverage family" independent of appearance in this codebase's
current architecture, so treating them as two independent families would be
double-counting the same underlying signal, not real corroboration.
appearance_anomaly_score and structural_anomaly_score (from
cropmerge/anomaly/structural.py's independent row/gap/fragmentation
pipeline) ARE genuinely computed independently of each other. Splitting
coverage out as its own real independent signal would need score_frame()
itself restructured to stop fusing it into appearance -- a real, separate,
larger change, not done here.

This module only ever CAPS priority downward (appearance-only evidence
never reaches "high" without structural or strong-persistence backing) --
it never raises a zone's priority or invents evidence that isn't there.
"""

from __future__ import annotations

from dataclasses import dataclass

from cropmerge.pipeline.schemas import ZoneEvidence


@dataclass(frozen=True)
class ConsensusThresholds:
    """Engineering thresholds, not scientifically calibrated constants --
    tune from real benchmark/review data, same caveat as every other
    threshold in this codebase's anomaly-scoring config."""

    structure_min: float = 0.15
    appearance_min: float = 0.15
    strong_structure_min: float = 0.35
    min_persistence_for_strong_structure: float = 0.6


@dataclass(frozen=True)
class ConsensusResult:
    structure_has_signal: bool
    appearance_has_signal: bool
    families_agreeing: int  # 0, 1, or 2
    strong_structure_alone: bool  # strong structural signal + high persistence
    high_confidence_supported: bool  # families_agreeing >= 2 OR strong_structure_alone


def evaluate_consensus(
    evidence: ZoneEvidence, thresholds: ConsensusThresholds | None = None
) -> ConsensusResult:
    t = thresholds or ConsensusThresholds()
    structure_signal = evidence.structural_anomaly_score or 0.0
    appearance_signal = evidence.appearance_anomaly_score or 0.0
    persistence = evidence.persistence or 0.0

    structure_has_signal = structure_signal >= t.structure_min
    appearance_has_signal = appearance_signal >= t.appearance_min
    families_agreeing = int(structure_has_signal) + int(appearance_has_signal)
    strong_structure_alone = (
        structure_signal >= t.strong_structure_min and persistence >= t.min_persistence_for_strong_structure
    )

    return ConsensusResult(
        structure_has_signal=structure_has_signal,
        appearance_has_signal=appearance_has_signal,
        families_agreeing=families_agreeing,
        strong_structure_alone=strong_structure_alone,
        high_confidence_supported=families_agreeing >= 2 or strong_structure_alone,
    )
