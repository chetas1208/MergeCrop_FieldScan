"""
Structural crop discontinuity detection — row gaps, sparse canopy, fragmentation.

Separate from appearance-based anomaly (color/DINO/texture).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from cropmerge.anomaly.grid import GridCell
from cropmerge.features.management_units import UnitSegmentationResult, UnitType
from cropmerge.features.rgb_indices import excess_green, vegetation_mask
from cropmerge.features.row_geometry import (
    RowGeometryConfig,
    analyze_row_geometry,
    structure_tensor_orientation,
)
from cropmerge.pipeline.schemas import SemanticClass

log = logging.getLogger("cropmerge.anomaly.structural")

NON_CROP_CLASSES = {
    SemanticClass.ROAD_PATH.value,
    SemanticClass.TREE_VEGETATION.value,
    SemanticClass.WATER.value,
    SemanticClass.INFRASTRUCTURE.value,
}


@dataclass
class StructuralFrameResult:
    row_visibility: str  # HIGH | MEDIUM | LOW
    row_angle_deg: float | None
    gap_mask: np.ndarray
    fragmentation_mask: np.ndarray
    occupancy: np.ndarray


def build_occupancy_map(
    bgr: np.ndarray,
    field_mask: np.ndarray,
    label_map: np.ndarray,
    crop_mask: np.ndarray | None,
) -> np.ndarray:
    """Fuse segmentation crop mask + ExG vegetation into soft occupancy [0,1]."""
    h, w = bgr.shape[:2]
    exg = excess_green(bgr)
    exg_n = np.clip((exg - exg.min()) / (exg.max() - exg.min() + 1e-6), 0, 1)
    veg = vegetation_mask(exg, 0.05).astype(np.float32)

    seg_crop = np.zeros((h, w), dtype=np.float32)
    if crop_mask is not None:
        seg_crop = crop_mask.astype(np.float32)
    elif label_map is not None:
        seg_crop = (label_map == SemanticClass.CROP.value).astype(np.float32)

    occ = 0.25 * seg_crop + 0.45 * veg + 0.30 * exg_n
    # Segmentation often over-labels crop; trust vegetation signal when they disagree
    disagree = (seg_crop > 0.4) & (veg < 0.35)
    occ[disagree] = np.minimum(occ[disagree], veg[disagree] * 0.6 + exg_n[disagree] * 0.4)
    occ[~field_mask] = 0.0
    if label_map is not None:
        for cls in NON_CROP_CLASSES:
            occ[label_map == cls] = 0.0
    return np.clip(occ, 0.0, 1.0)


def _structure_tensor_orientation(occ: np.ndarray, field_mask: np.ndarray) -> tuple[float | None, float]:
    """Dominant row direction via structure tensor on occupancy gradients.

    Thin wrapper: the actual formula lives once in
    cropmerge.features.row_geometry.structure_tensor_orientation (shared with
    the FarmTech row-geometry module, which independently implemented the
    same math) — kept as a local guard here only to preserve this call
    site's field_mask-sum short-circuit and its exact historical eps
    (1e-6, vs. the shared function's own default of 1e-9), so live
    row-visibility numbers are byte-identical to before this consolidation.
    """
    m = field_mask.astype(np.float32)
    if m.sum() < 100:
        return None, 0.0
    return structure_tensor_orientation(occ, m, eps=1e-6)


def _edge_exclusion_mask(field_mask: np.ndarray, px: int) -> np.ndarray:
    """True inside field but away from boundary (exclude headlands/edges)."""
    if not np.any(field_mask):
        return field_mask
    dist = cv2.distanceTransform(field_mask.astype(np.uint8), cv2.DIST_L2, 5)
    return dist >= px


def _sample_along_row(
    occ: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
    length: int,
    step: float = 1.0,
) -> np.ndarray:
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    h, w = occ.shape
    samples = []
    for i in range(-length, length + 1):
        x = int(cx + i * step * cos_a)
        y = int(cy + i * step * sin_a)
        if 0 <= x < w and 0 <= y < h:
            samples.append(float(occ[y, x]))
        else:
            samples.append(np.nan)
    return np.array(samples, dtype=np.float64)


def _gap_score_along_row(samples: np.ndarray, min_run: int = 3) -> tuple[float, float, float]:
    """
    Before/after continuity test on 1D occupancy profile.
    Returns (gap_score, before_mean, after_mean).
    """
    valid = ~np.isnan(samples)
    if valid.sum() < min_run * 3:
        return 0.0, 0.0, 0.0
    s = np.nan_to_num(samples, nan=0.0)
    baseline = float(np.percentile(s[valid], 75))
    if baseline < 0.15:
        return 0.0, 0.0, 0.0

    best = 0.0
    before_m = after_m = 0.0
    n = len(s)
    for start in range(min_run, n - min_run * 2):
        for end in range(start + min_run, min(n - min_run, start + n // 3)):
            gap = s[start:end]
            if float(np.mean(gap)) > baseline * 0.65:
                continue
            before = s[max(0, start - min_run) : start]
            after = s[end : min(n, end + min_run)]
            if len(before) < min_run or len(after) < min_run:
                continue
            b_mean = float(np.mean(before))
            a_mean = float(np.mean(after))
            if b_mean < baseline * 0.55 or a_mean < baseline * 0.55:
                continue
            deficit = 1.0 - float(np.mean(gap)) / max(b_mean, a_mean, 1e-6)
            extent = (end - start) / max(n, 1)
            score = deficit * (0.6 + 0.4 * min(extent * 8, 1.0))
            if score > best:
                best = score
                before_m, after_m = b_mean, a_mean
    return float(np.clip(best, 0, 1)), before_m, after_m


def _fragmentation_map(occ: np.ndarray, field_mask: np.ndarray, win: int = 15) -> np.ndarray:
    """Local hole/fragmentation vs smoothed neighborhood baseline."""
    m = field_mask.astype(np.float32)
    occ_f = occ * m
    k = max(win | 1, 3)
    local_mean = cv2.blur(occ_f, (k, k))
    local_sq = cv2.blur(occ_f * occ_f, (k, k))
    local_var = np.maximum(local_sq - local_mean**2, 0)
    # Low occupancy + high local variance → fragmented
    deficit = np.clip(local_mean - occ_f, 0, 1)
    frag = deficit * np.clip(local_var * 20, 0, 1)
    frag[~field_mask] = 0
    return frag.astype(np.float32)


def _hough_crosscheck_row_visibility(
    row_vis: str,
    angle: float | None,
    crop_mask: np.ndarray | None,
    field_mask: np.ndarray,
    min_consistent_rows: int,
) -> tuple[str, float | None]:
    """Opt-in corroborating check (config: structural.farmtech_hough_crosscheck_enabled,
    default off, unbenchmarked): when the structure-tensor coherence alone
    scored a frame "LOW", an independent Hough-line-based method
    (cropmerge.features.row_geometry.analyze_row_geometry) may still find
    consistent, well-supported row lines that a single aggregate coherence
    statistic can miss (e.g. two dominant orientations partially cancelling
    each other in the tensor sum). Only ever upgrades LOW -> MEDIUM, never
    downgrades and never claims HIGH — this is corroboration for a
    borderline case, not a replacement for the primary coherence signal.
    No-op (returns inputs unchanged) when row_vis isn't "LOW" or no crop
    mask is available to run Hough line detection on.
    """
    if row_vis != "LOW" or crop_mask is None:
        return row_vis, angle
    mask = (crop_mask.astype(bool) & field_mask.astype(bool)).astype(np.uint8)
    if mask.sum() < 100:
        return row_vis, angle
    geo = analyze_row_geometry(mask, RowGeometryConfig())
    if geo.num_consistent_rows >= min_consistent_rows:
        return "MEDIUM", geo.dominant_angle_deg if angle is None else angle
    return row_vis, angle


def detect_structural_frame(
    bgr: np.ndarray,
    field_mask: np.ndarray,
    label_map: np.ndarray,
    crop_mask: np.ndarray | None,
    cfg: dict,
) -> StructuralFrameResult:
    scfg = cfg.get("structural", {})
    edge_px = int(scfg.get("edge_exclusion_px", 12))
    row_coherence_min = float(scfg.get("row_visibility_min_coherence", 0.12))
    hough_crosscheck_enabled = bool(scfg.get("farmtech_hough_crosscheck_enabled", False))
    hough_crosscheck_min_rows = int(scfg.get("farmtech_hough_crosscheck_min_rows", 2))

    occ = build_occupancy_map(bgr, field_mask, label_map, crop_mask)
    angle, coherence = _structure_tensor_orientation(occ, field_mask)
    if coherence >= row_coherence_min * 1.8:
        row_vis = "HIGH"
    elif coherence >= row_coherence_min:
        row_vis = "MEDIUM"
    else:
        row_vis = "LOW"

    if hough_crosscheck_enabled:
        row_vis, angle = _hough_crosscheck_row_visibility(
            row_vis, angle, crop_mask, field_mask, hough_crosscheck_min_rows
        )

    interior = _edge_exclusion_mask(field_mask, edge_px)
    frag = _fragmentation_map(occ, interior)
    gap_mask = np.zeros_like(occ, dtype=bool)

    if angle is not None and row_vis != "LOW":
        h, w = occ.shape
        step = max(h, w) // 40
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                if not interior[y, x]:
                    continue
                profile = _sample_along_row(occ, x, y, angle, length=step * 2)
                gs, _, _ = _gap_score_along_row(profile)
                if gs > 0.35:
                    gap_mask[y - step // 2 : y + step // 2, x - step // 2 : x + step // 2] = True

    return StructuralFrameResult(
        row_visibility=row_vis,
        row_angle_deg=angle,
        gap_mask=gap_mask,
        fragmentation_mask=frag,
        occupancy=occ,
    )


def _cell_management_unit_id(cell: GridCell, management_units: UnitSegmentationResult | None) -> int | None:
    """Raw unit id for a cell's grid position, or None if unavailable/out of
    range/unassigned (-1). No confidence/type filtering here -- callers that
    need that (e.g. _local_unit_baselines) filter on top of this."""
    if management_units is None or cell.row >= len(management_units.cell_unit_ids):
        return None
    row_ids = management_units.cell_unit_ids[cell.row]
    if cell.col >= len(row_ids):
        return None
    uid = row_ids[cell.col]
    return uid if uid != -1 else None


def _local_unit_baselines(
    cells: list[GridCell],
    occ: np.ndarray,
    field_mask: np.ndarray,
    management_units: UnitSegmentationResult | None,
    cfg: dict,
) -> tuple[dict[int, float], dict[tuple[int, int], int]]:
    """Per-management-unit median occupancy, for cells confidently belonging
    to an ACTIVE_CROP unit with enough same-unit peers to be statistically
    meaningful. Returns (unit_baselines, cell->unit_id) -- a cell absent
    from cell->unit_id (no management_units, no confident unit, or too few
    peers) falls back to the whole-field baseline exactly as before this
    was added; this is additive, never a behavior change on its own."""
    cell_unit_id: dict[tuple[int, int], int] = {}
    if management_units is None:
        return {}, cell_unit_id

    mucfg = cfg.get("management_units", {})
    min_conf = float(mucfg.get("min_confidence_for_local_baseline", 0.3))
    min_peers = int(mucfg.get("min_peer_cells_for_local_baseline", 3))

    unit_by_id = {u.unit_id: u for u in management_units.units}
    occ_by_unit: dict[int, list[float]] = {}
    pos_by_unit: dict[tuple[int, int], int] = {}
    for cell in cells:
        if not cell.valid or cell.row >= len(management_units.cell_unit_ids):
            continue
        row_ids = management_units.cell_unit_ids[cell.row]
        if cell.col >= len(row_ids):
            continue
        uid = row_ids[cell.col]
        unit = unit_by_id.get(uid)
        if unit is None or unit.unit_type != UnitType.ACTIVE_CROP or unit.confidence < min_conf:
            continue
        region = field_mask[cell.y0 : cell.y1, cell.x0 : cell.x1]
        if not np.any(region):
            continue
        v = float(np.mean(occ[cell.y0 : cell.y1, cell.x0 : cell.x1][region]))
        pos_by_unit[(cell.row, cell.col)] = uid
        occ_by_unit.setdefault(uid, []).append(v)

    unit_baselines = {uid: float(np.median(vals)) for uid, vals in occ_by_unit.items() if len(vals) >= min_peers}
    cell_unit_id = {pos: uid for pos, uid in pos_by_unit.items() if uid in unit_baselines}
    return unit_baselines, cell_unit_id


def score_structural_cells(
    cells: list[GridCell],
    bgr: np.ndarray,
    field_mask: np.ndarray,
    label_map: np.ndarray,
    crop_mask: np.ndarray | None,
    cfg: dict,
    management_units: UnitSegmentationResult | None = None,
    unit_row_angles: dict[int, float] | None = None,
) -> tuple[list[GridCell], StructuralFrameResult]:
    """Attach structural scores + features to grid cells.

    management_units (cropmerge.features.management_units), when provided,
    gates a LOCAL occupancy baseline: an ACTIVE_CROP cell with enough
    same-unit ACTIVE_CROP peers is compared against ITS OWN unit's median
    occupancy, not the whole field's -- see _local_unit_baselines(). Fixes,
    at the baseline arithmetic level (not just the zone-label level
    classify_zone_type() already gates), a healthy crop block sitting next
    to a harvested residue block dragging the whole-field baseline down.

    unit_row_angles (unit_id -> dominant_angle_deg, from
    cropmerge.features.unit_row_geometry.analyze_management_unit_rows),
    when provided, makes gap-continuity sampling use THAT unit's own
    detected row angle instead of the single whole-frame angle -- and,
    since a unit only appears in this dict when its own row geometry was
    confidently detected, a cell inside it samples for gap continuity even
    if the frame's overall row_visibility is LOW (e.g. dragged down by a
    residue block with no rows at all).
    """
    scfg = cfg.get("structural", {})
    w_occ = float(scfg.get("weights", {}).get("occupancy_deficit", 0.30))
    w_cont = float(scfg.get("weights", {}).get("continuity_evidence", 0.25))
    w_gap = float(scfg.get("weights", {}).get("gap_extent", 0.20))
    w_frag = float(scfg.get("weights", {}).get("fragmentation", 0.15))
    w_soil = float(scfg.get("weights", {}).get("soil_exposure", 0.10))
    wsum = w_occ + w_cont + w_gap + w_frag + w_soil

    result = detect_structural_frame(bgr, field_mask, label_map, crop_mask, cfg)
    occ = result.occupancy
    h, w = bgr.shape[:2]
    from cropmerge.features.rgb_indices import excess_green, vegetation_mask

    exg = excess_green(bgr)
    veg = vegetation_mask(exg, 0.05)

    # Field baseline occupancy inside valid cells
    valid_occs = [
        float(np.mean(occ[cell.y0 : cell.y1, cell.x0 : cell.x1][field_mask[cell.y0 : cell.y1, cell.x0 : cell.x1]]))
        for cell in cells
        if cell.valid
    ]
    baseline = float(np.median(valid_occs)) if valid_occs else 0.5
    unit_baselines, cell_unit_id = _local_unit_baselines(cells, occ, field_mask, management_units, cfg)

    for cell in cells:
        if not cell.valid:
            cell.structural_anomaly_score = 0.0
            continue

        local_baseline = unit_baselines.get(cell_unit_id.get((cell.row, cell.col)), baseline)

        region = field_mask[cell.y0 : cell.y1, cell.x0 : cell.x1]
        lm = label_map[cell.y0 : cell.y1, cell.x0 : cell.x1]
        local_occ = occ[cell.y0 : cell.y1, cell.x0 : cell.x1]
        mean_occ = float(np.mean(local_occ[region])) if np.any(region) else 0.0
        local_veg = float(np.mean(veg[cell.y0 : cell.y1, cell.x0 : cell.x1][region])) if np.any(region) else 0.0
        bare = float(np.mean((lm == SemanticClass.BARE_SOIL.value) & region)) if np.any(region) else 0.0
        crop_frac = float(np.mean((lm == SemanticClass.CROP.value) & region)) if np.any(region) else 0.0

        occ_deficit = max(0.0, local_baseline - mean_occ) / max(local_baseline, 0.15)
        occ_deficit = max(occ_deficit, max(0.0, local_baseline - local_veg) / max(local_baseline, 0.15))
        # Low vegetation inside crop-labelled region → structural deficit (seg-independent)
        if local_veg < local_baseline * 0.55 and local_baseline >= 0.35:
            occ_deficit = max(occ_deficit, (local_baseline - local_veg) / max(local_baseline, 0.15))
            score_floor = 0.4 + 0.5 * (1.0 - local_veg / max(local_baseline, 0.15))
        else:
            score_floor = 0.0
        frag_val = float(np.mean(result.fragmentation_mask[cell.y0 : cell.y1, cell.x0 : cell.x1][region]))
        gap_val = float(np.mean(result.gap_mask[cell.y0 : cell.y1, cell.x0 : cell.x1][region]))

        continuity = 0.0
        gap_extent = gap_val
        unit_id = _cell_management_unit_id(cell, management_units)
        unit_angle = unit_row_angles.get(unit_id) if unit_row_angles and unit_id is not None else None
        if unit_angle is not None:
            cell_angle, angle_visible = unit_angle, True
        else:
            cell_angle, angle_visible = result.row_angle_deg, result.row_visibility != "LOW"
        if cell_angle is not None and angle_visible:
            cx = (cell.x0 + cell.x1) / 2
            cy = (cell.y0 + cell.y1) / 2
            profile = _sample_along_row(occ, cx, cy, cell_angle, length=max(h, w) // 16)
            gs, before, after = _gap_score_along_row(profile)
            continuity = gs
            gap_extent = max(gap_extent, gs)

        soil_delta = max(0.0, bare - float(cell.features.get("bare_soil", 0.0)))

        # Same real-field regression as classify_zone_type's occ_deficit
        # gate above (see that comment for the full trace): a real stand
        # gap removes crop, so it shows up in BOTH the along-row profile
        # AND the cell's own local occupancy. A single noisy 1D sample
        # crossing ordinary canopy texture (no real occupancy deficit)
        # must not carry full weight in the SCORE either, not just the
        # label -- otherwise the same false positives still rank as
        # HIGH-priority findings under a different name. `continuity`/
        # `gap_extent` stay stored on the cell at their raw value (still
        # useful diagnostic signal for explain_zone's reasons), only their
        # contribution to this cell's score is corroboration-scaled.
        corroboration = float(np.clip(occ_deficit / 0.15, 0.0, 1.0))
        score = (
            w_occ * np.clip(occ_deficit, 0, 1)
            + w_cont * continuity * corroboration
            + w_gap * np.clip(gap_extent, 0, 1) * corroboration
            + w_frag * np.clip(frag_val * 3, 0, 1)
            + w_soil * np.clip(soil_delta * 2, 0, 1)
        ) / max(wsum, 1e-6)

        # Interior bare-soil or low-veg island
        if bare >= 0.2 and local_baseline >= 0.45 and crop_frac < 0.5:
            score_floor = max(score_floor, 0.35 + 0.45 * bare)
        if local_veg < 0.25 and local_baseline >= 0.5:
            score_floor = max(score_floor, 0.45)

        if score_floor > 0:
            score = max(float(score), score_floor)

        cell.structural_anomaly_score = float(np.clip(score, 0.0, 1.0))
        cell.features.update(
            {
                "row_occupancy_deficit": occ_deficit,
                "gap_length_score": gap_extent,
                "continuity_evidence": continuity,
                "fragmentation_score": frag_val,
                "soil_exposure_delta": soil_delta,
                "crop_coverage_deficit": occ_deficit,
                "row_continuity_before": continuity,
                "row_continuity_after": continuity,
            }
        )

    return cells, result


_NON_ACTIVE_CROP_UNIT_TYPES = {"residue_stubble", "bare_soil", "non_crop"}
_UNIT_GATE_MIN_CONFIDENCE = 0.3


def classify_zone_type(
    cell: GridCell,
    row_visibility: str,
    unit_type: str | None = None,
    unit_confidence: float = 0.0,
) -> str:
    """Map cell features → InspectionZoneType value.

    unit_type/unit_confidence come from cropmerge.features.management_units
    (populated per-cell in pipeline/frame_analysis.py) and gate crop-specific
    findings: a cell whose management unit is confidently RESIDUE_STUBBLE/
    BARE_SOIL/NON_CROP contains no standing crop, so it can never
    legitimately produce "stand_gap"/"sparse_canopy" — those findings claim
    crop that should be present isn't. Real regression this fixes: a
    harvested residue block was being flagged as a crop gap purely because
    it looks different from a green-crop whole-field baseline.
    """
    raw = _raw_zone_type(cell, row_visibility)
    if (
        unit_type in _NON_ACTIVE_CROP_UNIT_TYPES
        and unit_confidence >= _UNIT_GATE_MIN_CONFIDENCE
        and raw in {"stand_gap", "sparse_canopy", "row_discontinuity"}
    ):
        return "general_visual_variation" if unit_type == "non_crop" else "exposed_soil"
    return raw


def _raw_zone_type(cell: GridCell, row_visibility: str) -> str:
    """Original feature→type mapping, pre management-unit gating."""
    struct = getattr(cell, "structural_anomaly_score", 0.0)
    appear = getattr(cell, "appearance_anomaly_score", cell.anomaly_score)
    feats = cell.features
    continuity = float(feats.get("continuity_evidence", 0))
    frag = float(feats.get("fragmentation_score", 0))
    occ_def = float(feats.get("row_occupancy_deficit", 0))
    bare = float(feats.get("bare_soil", 0))
    soil_d = float(feats.get("soil_exposure_delta", 0))

    if struct < 0.25 and appear < 0.25:
        return "general_visual_variation"

    # Real-field regression (2026-09-03, traced from real user report of
    # "identical-looking regions flagged in some spots but not others" on a
    # healthy, uniform soybean field): _sample_along_row()'s single 1D
    # profile through a cell can dip through ordinary within-canopy texture
    # (leaf clumping, shadow) and read as a near-perfect "gap" even when the
    # cell's own 2D occupancy is completely normal -- verified directly on
    # samples/real/real_soybean_field.jpg: 15 of 38 valid cells scored
    # continuity>=0.63 (several at 1.0) while their occ_deficit sat at
    # 0.0-0.12, scattered essentially at random depending on exactly where
    # the sampling line crossed a natural micro-dip. A genuine stand gap
    # removes crop, so it shows up in BOTH the along-row profile AND the
    # cell's own local occupancy -- requiring occ_deficit corroboration
    # (thresholds match explain.py's own existing "worth mentioning"
    # deficit bar) cut that same field's stand-gap-eligible cells from 15
    # to 1, keeping the one with a real, substantial 23% occupancy deficit.
    if continuity >= 0.45 and row_visibility in {"HIGH", "MEDIUM"} and occ_def >= 0.15:
        return "stand_gap"
    if continuity >= 0.30 and row_visibility in {"HIGH", "MEDIUM"} and occ_def >= 0.08:
        return "row_discontinuity"
    if bare >= 0.25 and occ_def >= 0.2 and struct >= 0.25:
        return "stand_gap"
    if frag >= 0.12 and occ_def >= 0.25:
        return "sparse_canopy"
    if bare >= 0.2 or soil_d >= 0.15:
        return "exposed_soil"
    if float(feats.get("color_difference", 0)) > 0.08 and struct < 0.35:
        return "color_variation"
    if float(feats.get("texture_difference", 0)) > 0.08 and struct < 0.35:
        return "texture_variation"
    if struct >= 0.4 and appear < 0.35:
        return "sparse_canopy"
    if appear >= 0.4 and struct < 0.35:
        return "color_variation"
    return "general_visual_variation"


ZONE_TYPE_LABELS = {
    "stand_gap": "Possible stand gap / crop discontinuity",
    "row_discontinuity": "Possible row discontinuity",
    "sparse_canopy": "Possible fragmented or sparse canopy",
    "exposed_soil": "Increased exposed soil",
    "color_variation": "Unusual crop appearance (color)",
    "texture_variation": "Unusual surface texture",
    "water_like_region": "Water-like region",
    "general_visual_variation": "General visual variation",
}
