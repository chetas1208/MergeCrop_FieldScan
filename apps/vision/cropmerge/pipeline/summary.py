"""Deterministic Field Analysis Summary — template-generated from already
computed report fields. Never an LLM output: every sentence here traces to a
real number already in the report. The optional local-LLM enrichment layer
(cropmerge/enrich/) may polish this prose but must not invent evidence beyond
what this module already established.
"""

from __future__ import annotations

from cropmerge.pipeline.schemas import (
    FieldAnalysisSummary,
    FieldTriageReport,
    InspectionZone,
    ReviewPriority,
)

_LOCATION_LABELS = {
    "northwest": "northwest",
    "north": "north",
    "northeast": "northeast",
    "west": "west",
    "center": "center",
    "east": "east",
    "southwest": "southwest",
    "south": "south",
    "southeast": "southeast",
}


def _location_label(zone: InspectionZone) -> str:
    return _LOCATION_LABELS.get(zone.relative_location.value, zone.relative_location.value)


def _observation_fraction(zone: InspectionZone) -> str | None:
    if zone.persistent_observations is not None and zone.total_observations:
        return f"{zone.persistent_observations} of {zone.total_observations} usable observations"
    if zone.frames_seen:
        return f"{zone.frames_seen} frames"
    return None


def _headline(scheduled: int, usable: int, zones: list[InspectionZone], high: list[InspectionZone]) -> str:
    if scheduled == 0:
        return "No observations were scheduled for this run."
    coverage_note = f"{usable} of {scheduled} scheduled observations were usable."
    if not zones:
        return f"No significant inspection areas were found. {coverage_note}"
    if high:
        top = high[0]
        return (
            f"The strongest review target is {top.primary_signal_label.lower()} in the "
            f"{_location_label(top)} of the field. {coverage_note}"
        )
    return f"{len(zones)} inspection area{'s' if len(zones) != 1 else ''} were flagged for review. {coverage_note}"


def _highest_priority_findings(top: InspectionZone) -> list[str]:
    findings = [f"Strongest evidence in this run: {top.primary_signal_label.lower()}."]
    obs = _observation_fraction(top)
    if obs:
        findings.append(f"Observed in {obs}.")
    if top.evidence.crop_coverage_delta is not None and top.evidence.crop_coverage_delta > 0:
        findings.append(f"Crop coverage is lower here by roughly {round(top.evidence.crop_coverage_delta * 100)}% relative to nearby field structure.")
    if top.evidence.soil_exposure_delta is not None and top.evidence.soil_exposure_delta > 0:
        findings.append(f"Visible soil is higher here by roughly {round(top.evidence.soil_exposure_delta * 100)}% relative to nearby field structure.")
    return findings


def build_field_analysis_summary(report: FieldTriageReport) -> FieldAnalysisSummary:
    scheduled = len(report.frame_quality)
    usable = sum(1 for f in report.frame_quality if f.usable)
    limited = scheduled - usable

    zones = report.inspection_zones
    high = [z for z in zones if z.review_priority == ReviewPriority.HIGH]
    medium = [z for z in zones if z.review_priority == ReviewPriority.MEDIUM]
    low = [z for z in zones if z.review_priority == ReviewPriority.LOW]

    ranked = sorted(zones, key=lambda z: z.review_score if z.review_score else z.anomaly_score, reverse=True)
    top = ranked[0] if ranked else None

    key_findings: list[str] = []
    if scheduled > 0:
        key_findings.append(f"{usable} of {scheduled} scheduled observations were usable for analysis.")
    if report.field.detected:
        key_findings.append(
            f"Estimated crop coverage across the analyzed field is "
            f"{round(report.field.mean_crop_coverage * 100)}%."
        )
    if zones:
        key_findings.append(
            f"{len(zones)} inspection area{'s' if len(zones) != 1 else ''} identified "
            f"({len(high)} high, {len(medium)} medium, {len(low)} low priority)."
        )
        if top is not None:
            key_findings.extend(_highest_priority_findings(top))
    else:
        key_findings.append("No persistent visual differences were found above the current review threshold.")

    persistent = [z for z in zones if (z.persistent_observations or 0) >= 2]
    if zones and persistent:
        key_findings.append(
            f"{len(persistent)} of {len(zones)} inspection area{'s' if len(persistent) != 1 else ''} "
            f"persisted across multiple observations rather than appearing once."
        )

    return FieldAnalysisSummary(
        scheduled_observations=scheduled,
        usable_observations=usable,
        limited_observations=limited,
        high_priority_count=len(high),
        medium_priority_count=len(medium),
        low_priority_count=len(low),
        highest_priority_zone_id=top.id if top is not None else None,
        headline=_headline(scheduled, usable, zones, high),
        key_findings=key_findings,
        limitations=list(report.limitations),
    )
