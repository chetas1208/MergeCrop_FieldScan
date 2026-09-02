from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cropmerge.anomaly.explain import (
    combined_review_score,
    explain_zone,
    recommendation,
    review_priority,
)
from cropmerge.anomaly.grid import GridCell
from cropmerge.anomaly.structural import ZONE_TYPE_LABELS, classify_zone_type
from cropmerge.pipeline.schemas import (
    InspectionZone,
    InspectionZoneType,
    RelativeLocation,
    ReviewPriority,
    ZoneEvidence,
)


def relative_location(cx: float, cy: float) -> RelativeLocation:
    """cx, cy in [0,1] image-normalized."""
    col = 0 if cx < 1 / 3 else (2 if cx > 2 / 3 else 1)
    row = 0 if cy < 1 / 3 else (2 if cy > 2 / 3 else 1)
    grid = [
        [RelativeLocation.NW, RelativeLocation.N, RelativeLocation.NE],
        [RelativeLocation.W, RelativeLocation.C, RelativeLocation.E],
        [RelativeLocation.SW, RelativeLocation.S, RelativeLocation.SE],
    ]
    return grid[row][col]


@dataclass
class _Track:
    id: str
    centroids: list[tuple[float, float]] = field(default_factory=list)
    appearance_scores: list[float] = field(default_factory=list)
    structural_scores: list[float] = field(default_factory=list)
    combined_scores: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    cells: list[GridCell] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    reg_confs: list[float] = field(default_factory=list)


def _centroid_norm(cell: GridCell, h: int, w: int) -> tuple[float, float]:
    cx = ((cell.x0 + cell.x1) / 2) / max(w, 1)
    cy = ((cell.y0 + cell.y1) / 2) / max(h, 1)
    return cx, cy


def _bbox_norm(cell: GridCell, h: int, w: int) -> tuple[float, float, float, float]:
    return cell.x0 / w, cell.y0 / h, (cell.x1 - cell.x0) / w, (cell.y1 - cell.y0) / h


def _cell_review_base(cell: GridCell) -> float:
    return float(max(cell.appearance_anomaly_score, cell.structural_anomaly_score))


def aggregate_zones(
    frame_cells: list[list[GridCell]],
    timestamps: list[float],
    quality_weights: list[float],
    frame_shape: tuple[int, int],
    cfg: dict,
    registration_confidences: list[float] | None = None,
) -> list[InspectionZone]:
    """Link high-scoring cells across time → persistent inspection zones."""
    tcfg = cfg.get("temporal", {})
    acfg = cfg.get("anomaly", {})
    min_frames = int(tcfg.get("min_persistent_frames", 3))
    link_dist = float(tcfg.get("centroid_link_distance", 0.12))
    mid_thr = float(acfg.get("medium_threshold", 0.40))
    score_gate = mid_thr * 0.75
    h, w = frame_shape
    reg_confs = registration_confidences or [1.0] * len(timestamps)
    usable_frames = sum(1 for qw in quality_weights if qw >= 0.25)

    tracks: list[_Track] = []
    next_id = 1

    for fi, (cells, ts, qw) in enumerate(zip(frame_cells, timestamps, quality_weights)):
        reg_c = reg_confs[fi] if fi < len(reg_confs) else 1.0
        frame_w = float(qw) * (0.5 + 0.5 * reg_c)
        ranked = sorted(
            [
                c
                for c in cells
                if c.valid and _cell_review_base(c) >= score_gate
            ],
            key=_cell_review_base,
            reverse=True,
        )
        candidates = ranked[:8]
        used_tracks: set[int] = set()
        for cell in sorted(candidates, key=_cell_review_base, reverse=True):
            cx, cy = _centroid_norm(cell, h, w)
            best_i, best_d = -1, 1e9
            for i, tr in enumerate(tracks):
                if i in used_tracks or not tr.centroids:
                    continue
                px, py = tr.centroids[-1]
                d = float(np.hypot(cx - px, cy - py))
                if d < best_d:
                    best_d, best_i = d, i
            base = _cell_review_base(cell)
            weighted = base * frame_w
            if best_i >= 0 and best_d <= link_dist:
                tr = tracks[best_i]
                tr.centroids.append((cx, cy))
                tr.appearance_scores.append(cell.appearance_anomaly_score * frame_w)
                tr.structural_scores.append(cell.structural_anomaly_score * frame_w)
                tr.combined_scores.append(weighted)
                tr.weights.append(frame_w)
                tr.timestamps.append(ts)
                tr.cells.append(cell)
                tr.bboxes.append(_bbox_norm(cell, h, w))
                tr.reg_confs.append(reg_c)
                used_tracks.add(best_i)
            else:
                tid = f"IZ-{next_id:03d}"
                next_id += 1
                tracks.append(
                    _Track(
                        id=tid,
                        centroids=[(cx, cy)],
                        appearance_scores=[cell.appearance_anomaly_score * frame_w],
                        structural_scores=[cell.structural_anomaly_score * frame_w],
                        combined_scores=[weighted],
                        weights=[frame_w],
                        timestamps=[ts],
                        cells=[cell],
                        bboxes=[_bbox_norm(cell, h, w)],
                        reg_confs=[reg_c],
                    )
                )

    zones: list[InspectionZone] = []
    for tr in tracks:
        if len(tr.timestamps) < min_frames:
            continue
        w_arr = np.maximum(tr.weights, 1e-3)
        mean_appear = float(np.average(tr.appearance_scores, weights=w_arr))
        mean_struct = float(np.average(tr.structural_scores, weights=w_arr))
        mean_base = float(max(mean_appear, mean_struct))
        if mean_base < mid_thr * 0.85:
            continue

        total_obs = max(usable_frames, len(timestamps))
        persistent_obs = len(tr.timestamps)
        persistence = float(
            np.clip(
                np.sum(w_arr) / max(sum(quality_weights), 1e-3),
                0.0,
                1.0,
            )
        )

        cx = float(np.mean([c[0] for c in tr.centroids]))
        cy = float(np.mean([c[1] for c in tr.centroids]))
        rep_idx = int(np.argmax(tr.combined_scores))
        rep = tr.cells[rep_idx]
        row_vis = rep.row_visibility or "LOW"
        type_key = classify_zone_type(rep, row_vis)
        try:
            primary_type = InspectionZoneType(type_key)
        except ValueError:
            primary_type = InspectionZoneType.GENERAL_VISUAL_VARIATION
        signal_label = ZONE_TYPE_LABELS.get(type_key, "General visual variation")

        review_score = combined_review_score(mean_base, persistence)
        pri = review_priority(mean_base, persistence, cfg)
        cov_deltas = [c.features.get("coverage_delta", 0.0) for c in tr.cells]
        mean_reg = float(np.mean(tr.reg_confs)) if tr.reg_confs else None

        reasons = explain_zone(
            rep,
            cfg,
            persistence=persistence,
            persistent_obs=persistent_obs,
            total_obs=total_obs,
            primary_type=type_key,
            mean_struct=mean_struct,
            mean_appear=mean_appear,
        )
        if not reasons:
            continue

        bb = tr.bboxes[rep_idx]
        zones.append(
            InspectionZone(
                id=tr.id,
                review_priority=ReviewPriority(pri),
                anomaly_score=round(mean_base, 4),
                appearance_anomaly_score=round(mean_appear, 4),
                structural_anomaly_score=round(mean_struct, 4),
                review_score=round(review_score, 4),
                persistence_score=round(persistence, 4),
                primary_type=primary_type,
                primary_signal_label=signal_label,
                persistent_observations=persistent_obs,
                total_observations=total_obs,
                first_seen_ms=round(min(tr.timestamps) * 1000, 1),
                last_seen_ms=round(max(tr.timestamps) * 1000, 1),
                first_seen_sec=round(min(tr.timestamps), 3),
                last_seen_sec=round(max(tr.timestamps), 3),
                frames_seen=persistent_obs,
                relative_location=relative_location(cx, cy),
                centroid_norm={"x": round(cx, 4), "y": round(cy, 4)},
                bbox_norm={
                    "x": round(bb[0], 4),
                    "y": round(bb[1], 4),
                    "w": round(bb[2], 4),
                    "h": round(bb[3], 4),
                },
                evidence=ZoneEvidence(
                    crop_coverage_delta=round(float(np.mean(cov_deltas)), 4),
                    color_difference=round(float(rep.features.get("color_difference", 0)), 4),
                    vegetation_difference=round(
                        float(rep.features.get("vegetation_difference", 0)), 4
                    ),
                    texture_difference=round(float(rep.features.get("texture_difference", 0)), 4),
                    embedding_difference=round(
                        float(rep.features.get("embedding_difference", 0)), 4
                    ),
                    persistence=round(persistence, 4),
                    appearance_anomaly_score=round(mean_appear, 4),
                    structural_anomaly_score=round(mean_struct, 4),
                    row_continuity_before=round(
                        float(rep.features.get("row_continuity_before", 0)), 4
                    ),
                    row_continuity_after=round(
                        float(rep.features.get("row_continuity_after", 0)), 4
                    ),
                    gap_extent_normalized=round(
                        float(rep.features.get("gap_length_score", 0)), 4
                    ),
                    soil_exposure_delta=round(
                        float(rep.features.get("soil_exposure_delta", 0)), 4
                    ),
                    fragmentation_score=round(
                        float(rep.features.get("fragmentation_score", 0)), 4
                    ),
                    registration_confidence=round(mean_reg, 4) if mean_reg is not None else None,
                ),
                reasons=reasons,
                recommendation=recommendation(
                    pri,
                    location=relative_location(cx, cy).value,
                    anomaly_score=mean_base,
                    persistence=persistence,
                    frames_seen=persistent_obs,
                    primary_signal=signal_label,
                    structural_score=mean_struct,
                    appearance_score=mean_appear,
                ),
            )
        )

    zones.sort(
        key=lambda z: (
            {"high": 2, "medium": 1, "low": 0}[z.review_priority.value],
            z.review_score,
            z.structural_anomaly_score,
            z.persistence_score,
        ),
        reverse=True,
    )
    return zones[:12]
