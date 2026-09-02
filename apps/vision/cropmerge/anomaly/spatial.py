"""
Spatial visual anomaly inside the field mask.

Real analysis:
  - per-cell RGB/ExG/Lab/texture stats
  - embedding cosine distance to robust field baseline
  - robust z-scores (median/MAD) per feature
  - optional IsolationForest on valid cells
  - weighted fusion → exploratory visual anomaly score [0,1]
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from cropmerge.anomaly.grid import GridCell, build_grid, normalize_scores
from cropmerge.features.dinov3 import cosine_distance
from cropmerge.features.rgb_indices import color_stats, excess_green, lab_distance, vegetation_mask
from cropmerge.features.texture import texture_features
from cropmerge.pipeline.schemas import SemanticClass


def _robust_z(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median/MAD z-score; invalid → 0. Higher = more unusual."""
    out = np.zeros_like(values, dtype=np.float64)
    if not np.any(valid):
        return out
    v = values[valid]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) + 1e-6
    z = np.abs(values - med) / (1.4826 * mad)
    out[valid] = z[valid]
    return out


def _local_robust_z(
    values: np.ndarray,
    valid: np.ndarray,
    rows_idx: np.ndarray,
    cols_idx: np.ndarray,
    radius: int,
    min_neighbors: int = 4,
) -> np.ndarray:
    """Median/MAD z-score computed against each cell's own spatial
    neighborhood (Chebyshev radius in grid cells), not the whole-field
    baseline. _robust_z()'s single global median/MAD implicitly assumes the
    field is otherwise uniform with only small localized anomalies -- that
    assumption breaks down for a field with large, deliberate, roughly
    equal-sized treatment zones (e.g. a research plot split into thirds):
    the global median lands somewhere between the zones, so cells in EVERY
    zone read as "different from the blended average," and which specific
    cells cross a flagging threshold becomes sensitive to grid-cell noise
    rather than genuine within-zone anomalies -- cells that look identical
    to their immediate neighbors can end up on opposite sides of the
    threshold. Comparing each cell to its local neighborhood instead means
    a cell surrounded by visually similar cells is not flagged just because
    it differs from a different part of the field.

    Cells with fewer than `min_neighbors` valid neighbors fall back to 0
    (no local signal) rather than a noisy estimate from too few samples --
    the global z-score (blended in by the caller) still covers those.
    """
    out = np.zeros_like(values, dtype=np.float64)
    if not np.any(valid):
        return out
    valid_positions = np.flatnonzero(valid)
    for i in valid_positions:
        r0, c0 = rows_idx[i], cols_idx[i]
        neighbor_mask = valid & (np.abs(rows_idx - r0) <= radius) & (np.abs(cols_idx - c0) <= radius)
        neighbor_mask[i] = False
        if int(np.sum(neighbor_mask)) < min_neighbors:
            continue
        local_vals = values[neighbor_mask]
        med = float(np.median(local_vals))
        mad = float(np.median(np.abs(local_vals - med))) + 1e-6
        out[i] = abs(values[i] - med) / (1.4826 * mad)
    return out


def _to_unit(z: np.ndarray, valid: np.ndarray, clip: float = 4.0) -> np.ndarray:
    """Map robust z to [0,1] via clipped scale + within-frame percentile norm."""
    x = np.clip(z / clip, 0.0, 1.0)
    if np.any(valid):
        # blend global z-scale with relative ranking among valid cells
        rel = np.zeros_like(x)
        vv = x[valid]
        rel[valid] = normalize_scores(vv)
        x = 0.55 * x + 0.45 * rel
    x[~valid] = 0.0
    return x


def score_frame(
    bgr: np.ndarray,
    field_mask: np.ndarray,
    label_map: np.ndarray,
    embeddings: np.ndarray,
    cfg: dict,
) -> tuple[list[GridCell], np.ndarray]:
    gcfg = cfg.get("grid", {})
    acfg = cfg.get("anomaly", {})
    fcfg = cfg.get("features", {})
    rows = int(gcfg.get("rows", 8))
    cols = int(gcfg.get("cols", 8))
    min_frac = float(gcfg.get("min_field_fraction", 0.25))
    exg_thr = float(fcfg.get("vegetation_exg_threshold", 0.05))
    weights = acfg.get("weights", {})
    w_dino = float(weights.get("dino", 0.40))
    w_cov = float(weights.get("coverage", 0.25))
    w_col = float(weights.get("color", 0.20))
    w_veg = float(weights.get("vegetation", 0.10))
    w_tex = float(weights.get("texture", 0.05))
    use_iforest = bool(acfg.get("use_isolation_forest", True))
    iforest_weight = float(acfg.get("isolation_forest_weight", 0.20))
    # See _local_robust_z()'s docstring: blends in a spatially-local
    # baseline alongside the whole-field one so a cell isn't flagged just
    # for differing from a DIFFERENT part of a multi-zone field (e.g. a
    # research plot split into large treatment strips). 0.0 reproduces the
    # old pure-global behavior exactly; 1.0 would be pure-local.
    local_baseline_weight = float(acfg.get("local_baseline_weight", 0.5))
    local_baseline_radius = int(acfg.get("local_baseline_radius", 2))

    h, w = bgr.shape[:2]
    cells = build_grid(h, w, rows, cols)
    rows_idx = np.array([c.row for c in cells], dtype=np.int32)
    cols_idx = np.array([c.col for c in cells], dtype=np.int32)
    exg = excess_green(bgr)
    veg = vegetation_mask(exg, exg_thr)

    field_stats = color_stats(bgr, field_mask)
    field_tex = texture_features(bgr, field_mask)
    field_cov = (
        float(np.mean((label_map == SemanticClass.CROP.value) & field_mask))
        if np.any(field_mask)
        else 0.0
    )
    field_veg = float(np.mean(veg & field_mask)) if np.any(field_mask) else 0.0

    valid_emb = [
        embeddings[r, c]
        for r in range(rows)
        for c in range(cols)
        if float(np.linalg.norm(embeddings[r, c])) > 1e-6
    ]
    if valid_emb:
        baseline_emb = np.median(np.stack(valid_emb, axis=0), axis=0)
        baseline_emb = baseline_emb / (np.linalg.norm(baseline_emb) + 1e-8)
    else:
        baseline_emb = np.zeros(embeddings.shape[-1], dtype=np.float32)

    n = rows * cols
    cov_raw = np.zeros(n, dtype=np.float64)
    col_raw = np.zeros(n, dtype=np.float64)
    veg_raw = np.zeros(n, dtype=np.float64)
    tex_raw = np.zeros(n, dtype=np.float64)
    emb_raw = np.zeros(n, dtype=np.float64)
    valid_mask = np.zeros(n, dtype=bool)
    feat_matrix = []  # for isolation forest

    for i, cell in enumerate(cells):
        fm = field_mask[cell.y0 : cell.y1, cell.x0 : cell.x1]
        frac = float(np.mean(fm)) if fm.size else 0.0
        cell.field_fraction = frac
        cell.valid = frac >= min_frac
        valid_mask[i] = cell.valid
        if not cell.valid:
            feat_matrix.append(np.zeros(6, dtype=np.float64))
            continue

        region_mask = np.zeros((h, w), dtype=bool)
        region_mask[cell.y0 : cell.y1, cell.x0 : cell.x1] = fm
        lm = label_map[cell.y0 : cell.y1, cell.x0 : cell.x1]
        crop_cov = float(np.mean((lm == SemanticClass.CROP.value) & fm)) if np.any(fm) else 0.0
        bare = float(np.mean((lm == SemanticClass.BARE_SOIL.value) & fm)) if np.any(fm) else 0.0
        st = color_stats(bgr, region_mask)
        tx = texture_features(bgr, region_mask)
        local_exg = float(np.mean(exg[cell.y0 : cell.y1, cell.x0 : cell.x1][fm])) if np.any(fm) else 0.0
        local_veg = (
            float(np.mean(veg[cell.y0 : cell.y1, cell.x0 : cell.x1][fm])) if np.any(fm) else 0.0
        )
        emb = embeddings[cell.row, cell.col].astype(np.float64)
        cell.embedding = emb

        cov_delta = crop_cov - field_cov
        color_diff = lab_distance(st, field_stats)
        veg_diff = abs(local_veg - field_veg)
        tex_diff = abs(tx["variance"] - field_tex["variance"]) + abs(
            tx["edge_density"] - field_tex["edge_density"]
        )
        if np.linalg.norm(emb) > 1e-6:
            emb_u = emb / (np.linalg.norm(emb) + 1e-8)
            e_diff = cosine_distance(emb_u.astype(np.float32), baseline_emb.astype(np.float32))
        else:
            e_diff = 0.0

        # Also use absolute ExG deviation from field median ExG
        field_exg = field_stats.get("exg_mean", 0.0)
        exg_diff = abs(local_exg - field_exg)

        cell.features = {
            "crop_coverage": crop_cov,
            "bare_soil": bare,
            "exg_mean": st["exg_mean"],
            "veg_frac": local_veg,
            "coverage_delta": cov_delta,
            "color_difference": color_diff,
            "vegetation_difference": veg_diff,
            "texture_difference": tex_diff,
            "embedding_difference": e_diff,
            "exg_difference": exg_diff,
            **{f"tex_{k}": v for k, v in tx.items()},
        }

        cov_raw[i] = abs(cov_delta)
        col_raw[i] = color_diff
        veg_raw[i] = veg_diff + 0.5 * exg_diff
        tex_raw[i] = tex_diff
        emb_raw[i] = e_diff
        feat_matrix.append(
            np.array(
                [cov_delta, color_diff, veg_diff, tex_diff, e_diff, bare],
                dtype=np.float64,
            )
        )

    # Robust z-scores (not naive min-max alone), blending a whole-field
    # baseline with a spatially-local one -- see _local_robust_z()'s
    # docstring for why the local component matters.
    def _blended_z(raw: np.ndarray) -> np.ndarray:
        global_z = _robust_z(raw, valid_mask)
        if local_baseline_weight <= 0.0:
            return global_z
        local_z = _local_robust_z(raw, valid_mask, rows_idx, cols_idx, local_baseline_radius)
        return (1.0 - local_baseline_weight) * global_z + local_baseline_weight * local_z

    cov_u = _to_unit(_blended_z(cov_raw), valid_mask)
    col_u = _to_unit(_blended_z(col_raw), valid_mask)
    veg_u = _to_unit(_blended_z(veg_raw), valid_mask)
    tex_u = _to_unit(_blended_z(tex_raw), valid_mask)
    emb_u = _to_unit(_blended_z(emb_raw), valid_mask)

    iforest_u = np.zeros(n, dtype=np.float64)
    X = np.asarray(feat_matrix, dtype=np.float64)
    n_valid = int(valid_mask.sum())
    if use_iforest and n_valid >= 8:
        Xv = X[valid_mask]
        # standardize
        mu = Xv.mean(axis=0)
        sd = Xv.std(axis=0) + 1e-6
        Xn = (X - mu) / sd
        clf = IsolationForest(
            n_estimators=100,
            contamination=min(0.2, max(0.05, 3.0 / n_valid)),
            random_state=0,
        )
        clf.fit(Xn[valid_mask])
        # decision_function: higher = more normal → invert
        dec = clf.decision_function(Xn)
        # map to [0,1] anomaly
        d = -dec
        iforest_u = _to_unit(_robust_z(d, valid_mask), valid_mask)

    wsum = w_dino + w_cov + w_col + w_veg + w_tex
    if use_iforest and n_valid >= 8:
        # renormalize to leave room for iforest
        scale = max(1.0 - iforest_weight, 0.5)
        w_dino, w_cov, w_col, w_veg, w_tex = [
            w * scale for w in (w_dino, w_cov, w_col, w_veg, w_tex)
        ]
        wsum = w_dino + w_cov + w_col + w_veg + w_tex + iforest_weight
    else:
        iforest_weight = 0.0

    heat = np.zeros((rows, cols), dtype=np.float32)
    for i, cell in enumerate(cells):
        if not cell.valid:
            continue
        contrib = {
            "dino": float(emb_u[i]) * w_dino,
            "coverage": float(cov_u[i]) * w_cov,
            "color": float(col_u[i]) * w_col,
            "vegetation": float(veg_u[i]) * w_veg,
            "texture": float(tex_u[i]) * w_tex,
        }
        if iforest_weight > 0:
            contrib["isolation"] = float(iforest_u[i]) * iforest_weight
        score = float(np.clip(sum(contrib.values()) / max(wsum, 1e-6), 0.0, 1.0))
        cell.appearance_anomaly_score = score
        cell.anomaly_score = score
        cell.contributions = contrib
        heat[cell.row, cell.col] = score

    # Spatial smooth heatmap (3x3) to reduce salt/pepper
    if heat.size >= 9:
        k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
        k /= k.sum()
        pad = np.pad(heat, 1, mode="edge")
        sm = np.zeros_like(heat)
        for r in range(rows):
            for c in range(cols):
                sm[r, c] = float(np.sum(pad[r : r + 3, c : c + 3] * k))
        # only apply smooth on valid cells
        for cell in cells:
            if cell.valid:
                cell.anomaly_score = float(
                    0.65 * cell.anomaly_score + 0.35 * sm[cell.row, cell.col]
                )
                cell.appearance_anomaly_score = cell.anomaly_score
                heat[cell.row, cell.col] = cell.anomaly_score

    return cells, heat
