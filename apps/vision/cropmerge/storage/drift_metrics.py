"""Analysis-drift comparison between two FieldTriageReports produced from
the same source video/image at the same sample_fps and config (typically:
an original source vs. a lossy-compressed candidate derivative).

This is the acceptance gate the storage campaign requires to exist BEFORE
any lossy source-replacement compression is ever turned on
(CROPMERGE_LOSSY_SOURCE_REPLACEMENT stays off regardless of what this module
measures -- it only produces the numbers a human/policy decision would be
based on, it does not itself gate anything).

LIMITATION (documented, not hidden): FieldTriageReport does not currently
expose raw per-frame segmentation masks, only report-level aggregates
(mean_crop_coverage, mean_bare_soil, class_coverage fractions) and
Inspection Zone geometry (bbox_norm/centroid_norm, image-relative
normalized rectangles). True pixel-level mask IoU per observation would
need cropmerge/pipeline/processor.py to optionally retain and return raw
label maps -- a real but separate follow-up (would materially increase
report/output size, so it should stay opt-in). This module instead compares
what a completed report actually contains: coverage fractions and
Inspection Zone bbox overlap/type/priority agreement. That is real,
report-level drift, just not literal per-pixel segmentation drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cropmerge.pipeline.schemas import FieldTriageReport, InspectionZone


def _bbox_iou(a: dict[str, float], b: dict[str, float]) -> float:
    """IoU of two normalized (x, y, w, h) rectangles in [0, 1] image space."""
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a["w"]) * max(0.0, a["h"])
    area_b = max(0.0, b["w"]) * max(0.0, b["h"])
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


@dataclass(frozen=True)
class ZoneMatch:
    original: InspectionZone
    candidate: InspectionZone | None
    iou: float
    type_agrees: bool
    priority_agrees: bool


def match_zones(
    original_zones: list[InspectionZone],
    candidate_zones: list[InspectionZone],
    min_iou: float = 0.3,
) -> list[ZoneMatch]:
    """Greedy best-IoU matching from each original zone to an unclaimed
    candidate zone. A zone with no candidate reaching `min_iou` is recorded
    as unmatched (candidate=None, iou=0.0) -- this is exactly the "a
    compressed artifact that drops a real finding" failure mode the
    campaign's high-priority-recall gate is meant to catch."""
    remaining = list(candidate_zones)
    matches: list[ZoneMatch] = []
    for orig in original_zones:
        best_idx, best_iou = -1, 0.0
        for idx, cand in enumerate(remaining):
            iou = _bbox_iou(orig.bbox_norm, cand.bbox_norm)
            if iou > best_iou:
                best_idx, best_iou = idx, iou
        if best_idx >= 0 and best_iou >= min_iou:
            cand = remaining.pop(best_idx)
            matches.append(
                ZoneMatch(
                    original=orig, candidate=cand, iou=best_iou,
                    type_agrees=orig.primary_type == cand.primary_type,
                    priority_agrees=orig.review_priority == cand.review_priority,
                )
            )
        else:
            matches.append(ZoneMatch(original=orig, candidate=None, iou=0.0, type_agrees=False, priority_agrees=False))
    return matches


@dataclass(frozen=True)
class DriftReport:
    crop_coverage_mae: float
    soil_fraction_mae: float
    field_fraction_mae: float
    zone_count_original: int
    zone_count_candidate: int
    zone_matches: list[ZoneMatch] = field(default_factory=list)
    type_agreement: float = 1.0
    priority_agreement: float = 1.0
    high_priority_recall: float = 1.0
    mean_matched_iou: float = 1.0

    def to_dict(self) -> dict:
        return {
            "cropCoverageMae": self.crop_coverage_mae,
            "soilFractionMae": self.soil_fraction_mae,
            "fieldFractionMae": self.field_fraction_mae,
            "zoneCountOriginal": self.zone_count_original,
            "zoneCountCandidate": self.zone_count_candidate,
            "typeAgreement": self.type_agreement,
            "priorityAgreement": self.priority_agreement,
            "highPriorityRecall": self.high_priority_recall,
            "meanMatchedIou": self.mean_matched_iou,
        }

    def to_markdown_row(self, candidate_name: str) -> str:
        return (
            f"| {candidate_name} | {self.crop_coverage_mae:.4f} | {self.soil_fraction_mae:.4f} | "
            f"{self.type_agreement:.2%} | {self.priority_agreement:.2%} | {self.high_priority_recall:.2%} | "
            f"{self.mean_matched_iou:.3f} |"
        )


def compare_reports(original: FieldTriageReport, candidate: FieldTriageReport, min_iou: float = 0.3) -> DriftReport:
    matches = match_zones(original.inspection_zones, candidate.inspection_zones, min_iou=min_iou)
    n = len(matches)
    type_agreement = sum(1 for m in matches if m.candidate is not None and m.type_agrees) / n if n else 1.0
    priority_agreement = sum(1 for m in matches if m.candidate is not None and m.priority_agrees) / n if n else 1.0

    from cropmerge.pipeline.schemas import ReviewPriority

    high_orig = [m for m in matches if m.original.review_priority == ReviewPriority.HIGH]
    high_recall = (
        sum(1 for m in high_orig if m.candidate is not None) / len(high_orig) if high_orig else 1.0
    )
    matched_ious = [m.iou for m in matches if m.candidate is not None]
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 1.0

    return DriftReport(
        crop_coverage_mae=abs(original.field.mean_crop_coverage - candidate.field.mean_crop_coverage),
        soil_fraction_mae=abs(original.field.mean_bare_soil - candidate.field.mean_bare_soil),
        field_fraction_mae=abs(original.field.mean_field_fraction - candidate.field.mean_field_fraction),
        zone_count_original=len(original.inspection_zones),
        zone_count_candidate=len(candidate.inspection_zones),
        zone_matches=matches,
        type_agreement=type_agreement,
        priority_agreement=priority_agreement,
        high_priority_recall=high_recall,
        mean_matched_iou=mean_iou,
    )


@dataclass(frozen=True)
class DriftGate:
    """Engineering safety thresholds -- NOT published agronomic constants,
    see docs/STORAGE_COMPRESSION_RESEARCH.md. Tune from real benchmark data,
    don't treat these defaults as scientifically calibrated."""

    max_crop_coverage_mae: float = 0.01
    max_soil_fraction_mae: float = 0.01
    min_type_agreement: float = 0.98
    min_high_priority_recall: float = 1.0


def passes_gate(drift: DriftReport, gate: DriftGate | None = None) -> tuple[bool, list[str]]:
    """Returns (passes, reasons_for_failure). A candidate that fails should
    never be used to replace a source -- see CROPMERGE_LOSSY_SOURCE_REPLACEMENT,
    which stays off regardless of this result until real footage has been
    benchmarked and this gate has actually been exercised against it."""
    gate = gate or DriftGate()
    reasons = []
    if drift.crop_coverage_mae > gate.max_crop_coverage_mae:
        reasons.append(f"crop coverage MAE {drift.crop_coverage_mae:.4f} > {gate.max_crop_coverage_mae}")
    if drift.soil_fraction_mae > gate.max_soil_fraction_mae:
        reasons.append(f"soil fraction MAE {drift.soil_fraction_mae:.4f} > {gate.max_soil_fraction_mae}")
    if drift.type_agreement < gate.min_type_agreement:
        reasons.append(f"type agreement {drift.type_agreement:.2%} < {gate.min_type_agreement:.2%}")
    if drift.high_priority_recall < gate.min_high_priority_recall:
        reasons.append(f"high-priority recall {drift.high_priority_recall:.2%} < {gate.min_high_priority_recall:.2%}")
    return len(reasons) == 0, reasons
