from __future__ import annotations

from cropmerge.anomaly.grid import GridCell
from cropmerge.anomaly.structural import ZONE_TYPE_LABELS

REVIEW_SCORE_ANOMALY_WEIGHT = 0.65
REVIEW_SCORE_PERSISTENCE_WEIGHT = 0.35


def combined_review_score(anomaly: float, persistence: float) -> float:
    """Review priority from max(appearance, structural) + persistence — not health confidence."""
    base = anomaly
    return REVIEW_SCORE_ANOMALY_WEIGHT * base + REVIEW_SCORE_PERSISTENCE_WEIGHT * persistence


def _pct_delta(value: float) -> str:
    sign = "+" if value > 0 else "−" if value < 0 else ""
    return f"{sign}{abs(value):.0%}"


def _pct_share(value: float) -> str:
    return f"{max(0.0, min(1.0, value)):.0%}"


def _loc_label(location: str | None) -> str:
    if not location:
        return "this patch"
    return location.replace("_", " ").replace("-", " ").lower()


def explain_cell(cell: GridCell, cfg: dict, persistence: float | None = None) -> list[str]:
    """Legacy wrapper — delegates to explain_zone."""
    return explain_zone(cell, cfg, persistence=persistence)


def explain_zone(
    cell: GridCell,
    cfg: dict,
    *,
    persistence: float | None = None,
    persistent_obs: int | None = None,
    total_obs: int | None = None,
    primary_type: str = "general_visual_variation",
    mean_struct: float | None = None,
    mean_appear: float | None = None,
) -> list[str]:
    """Human-readable inspection zone reasons with structural vs appearance separation."""
    min_c = float(cfg.get("anomaly", {}).get("reason_contribution_min", 0.06))
    reasons: list[str] = []
    feats = cell.features
    contrib = cell.contributions
    struct = mean_struct if mean_struct is not None else cell.structural_anomaly_score
    appear = mean_appear if mean_appear is not None else cell.appearance_anomaly_score

    if primary_type in {"stand_gap", "row_discontinuity"}:
        cont = float(feats.get("continuity_evidence", 0))
        if cont >= 0.2:
            reasons.append("Crop-row continuity breaks in this region")
        before = float(feats.get("row_continuity_before", cont))
        after = float(feats.get("row_continuity_after", cont))
        if before >= 0.3 and after >= 0.3:
            reasons.append("Rows/vegetation continue into and out of this region")
        occ_def = float(feats.get("row_occupancy_deficit", feats.get("crop_coverage_deficit", 0)))
        if occ_def >= 0.15:
            reasons.append(
                f"Visible vegetation occupancy is {_pct_share(occ_def)} below nearby rows"
            )
        soil = float(feats.get("soil_exposure_delta", feats.get("bare_soil", 0)))
        if soil >= 0.1:
            reasons.append(f"Additional exposed soil is visible ({_pct_share(soil)} increase)")

    elif primary_type == "sparse_canopy":
        frag = float(feats.get("fragmentation_score", 0))
        occ_def = float(feats.get("row_occupancy_deficit", 0))
        if frag >= 0.08:
            reasons.append("Crop canopy appears fragmented compared to surrounding field")
        if occ_def >= 0.15:
            reasons.append(f"Local crop occupancy is {_pct_share(occ_def)} below field baseline")

    elif primary_type == "exposed_soil":
        bare = float(feats.get("bare_soil", 0))
        reasons.append(
            f"Exposed soil visible ({_pct_share(bare)} of cell) — not automatically a planter skip"
        )

    if contrib.get("color", 0) >= min_c or (appear >= 0.35 and struct < 0.35):
        lab = float(feats.get("color_difference", 0.0))
        reasons.append(
            f"Color differs from surrounding crop (Lab distance {lab:.2f} — appearance only)"
        )

    if contrib.get("dino", 0) >= min_c:
        emb = float(feats.get("embedding_difference", 0.0))
        reasons.append(f"DINO visual features differ from field baseline (distance {emb:.2f})")

    if contrib.get("texture", 0) >= min_c:
        tex = float(feats.get("texture_difference", 0.0))
        reasons.append(f"Surface texture differs from neighboring crop ({tex:.2f})")

    if contrib.get("coverage", 0) >= min_c and primary_type not in {
        "stand_gap",
        "row_discontinuity",
        "sparse_canopy",
    }:
        delta = float(feats.get("coverage_delta", 0.0))
        crop = float(feats.get("crop_coverage", 0.0))
        if delta < -0.05:
            reasons.append(
                f"Canopy thinner than field norm ({_pct_delta(delta)}; {_pct_share(crop)} crop cover here)"
            )

    if struct < 0.25 and appear >= 0.35:
        reasons.append("No strong row discontinuity detected — primarily appearance-based")

    if persistence is not None and persistent_obs is not None and total_obs is not None:
        reasons.append(
            f"Pattern persists across {persistent_obs} of {total_obs} usable observations "
            f"(persistence {_pct_share(persistence)})"
        )
    elif persistence is not None and persistence >= 0.5:
        reasons.append(f"Not a one-frame glitch — persistence ~{_pct_share(persistence)}")

    if not reasons:
        label = ZONE_TYPE_LABELS.get(primary_type, "Visual variation")
        reasons.append(f"Flagged for review: {label}")

    return reasons


def review_priority(score: float, persistence: float, cfg: dict) -> str:
    hi = float(cfg.get("anomaly", {}).get("high_threshold", 0.62))
    mid = float(cfg.get("anomaly", {}).get("medium_threshold", 0.38))
    combined = combined_review_score(score, persistence)
    if combined >= hi:
        return "high"
    if combined >= mid:
        return "medium"
    return "low"


def recommendation(
    priority: str,
    *,
    location: str | None = None,
    anomaly_score: float = 0.0,
    persistence: float = 0.0,
    frames_seen: int = 0,
    primary_signal: str = "Visual variation",
    structural_score: float = 0.0,
    appearance_score: float = 0.0,
) -> str:
    loc = _loc_label(location)
    review = combined_review_score(anomaly_score, persistence)
    frame_note = f" across {frames_seen} observations" if frames_seen else ""

    if structural_score >= 0.5 and structural_score > appearance_score:
        action = (
            f"Review this area for possible crop discontinuity or sparse canopy — "
            f"structural signal is stronger than color/texture{frame_note}."
        )
    elif priority == "high":
        action = (
            f"Fly the {loc} sector on your next pass — review score {review:.0%}{frame_note}. "
            f"Primary signal: {primary_signal}."
        )
    elif priority == "medium":
        action = (
            f"Bookmark the {loc} area for the next flight (review score {review:.0%}). "
            f"{primary_signal}."
        )
    else:
        action = f"Low-key variation in the {loc} — optional drive-by."

    return action + " Recommended action: review this area more closely. Not a diagnosis."
