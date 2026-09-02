"""
Crop-row geometry from a binary crop mask.

Implements the geometry half of the row-detection approach described in
Ullah, Islam & Bais 2024 (Computers and Electronics in Agriculture):
morphological cleanup -> skeleton thinning -> Probabilistic Hough Transform
for dominant row-line candidates, clustered by angle and perpendicular
position. A second, independent orientation estimator (structure-tensor
gradient coherence, same formula already used in
cropmerge/anomaly/structural.py) is provided as a comparable cross-check.

See docs/FARMTECH_RESEARCH_BASIS.md for the paper trail and
docs/FARMTECH_METHOD_TRACEABILITY.md for the formula -> code -> runtime-role
table.

Scope note: this module is intentionally NOT wired into
cropmerge/pipeline/processor.py. It operates purely on a caller-supplied
numpy mask/image and is designed to be independently unit tested. Wiring it
into the live per-frame pipeline, and training the ~14M-param lightweight
row-segmentation model whose output this module would eventually consume
instead of today's heuristic/segmentation crop mask, are both explicitly
deferred (see docs/FARMTECH_RESEARCH_BASIS.md, "Explicitly deferred").

Guo-Hall thinning availability: Guo-Hall thinning requires cv2.ximgproc,
which ships in opencv-contrib-python(-headless), NOT in the
opencv-python-headless dependency this project actually installs (see
apps/vision/pyproject.toml). When cv2.ximgproc is unavailable, this module
falls back to a pure-Python/numpy Zhang-Suen thinning implementation
(Zhang & Suen, 1984) and reports which method actually ran via
RowGeometryResult.thinning_method / guo_hall_available -- it never silently
substitutes one method for the other without saying so.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger("cropmerge.features.row_geometry")

try:
    import cv2.ximgproc as _cv2_ximgproc  # type: ignore

    _HAS_XIMGPROC = hasattr(_cv2_ximgproc, "thinning") and hasattr(_cv2_ximgproc, "THINNING_GUOHALL")
except Exception:  # pragma: no cover - import-time environment probe
    _cv2_ximgproc = None
    _HAS_XIMGPROC = False

_WARNED_NO_XIMGPROC = False


def guo_hall_available() -> bool:
    """Whether cv2.ximgproc's Guo-Hall thinning is importable in this env."""
    return _HAS_XIMGPROC


@dataclass(frozen=True)
class RowGeometryConfig:
    """Named row-geometry parameters (mirrors configs/farmtech_v1.yaml
    `row_geometry:` block; that YAML is the source of truth for deployed
    values, this dataclass is the code-side default/typed view)."""

    # Morphological cleanup applied to the input mask before thinning.
    morph_open_px: int = 3
    morph_close_px: int = 5

    # cv2.HoughLinesP (Probabilistic Hough Transform) parameters.
    hough_rho_px: float = 1.0
    hough_theta_deg: float = 1.0
    hough_threshold: int = 30
    hough_min_line_length_px: int = 20
    hough_max_line_gap_px: int = 10

    # Row-line clustering: lines within `angle_cluster_tol_deg` of the
    # robust dominant angle are treated as row-aligned; row-aligned lines
    # whose perpendicular-offset positions are within `row_position_tol_px`
    # of each other are merged into one row-line cluster (one detected row).
    angle_cluster_tol_deg: float = 12.0
    row_position_tol_px: float = 15.0

    # A row-line cluster only counts as a "consistent row" if its combined
    # (summed) Hough segment length reaches this many pixels of support.
    min_row_support_px: float = 60.0

    # Masks larger than this on their longest side are downsampled (nearest-
    # neighbor, binary-preserving) before thinning, then support_px/offset_px
    # are scaled back up to original-image units. Real-world necessity, not
    # a theoretical one: the Zhang-Suen fallback thinning (used whenever
    # opencv-contrib's Guo-Hall isn't installed -- see thin_mask()) is
    # numpy-vectorized but still does up to 400 full-array passes; on an
    # uncapped ~4000x2250 real drone/overhead photo this measured well over
    # three minutes, effectively hanging any caller. Row geometry (angle,
    # coherence, line counts) is a topological/statistical property that
    # doesn't need native resolution -- an 800px cap keeps it fast and
    # accurate for this purpose.
    max_dimension_px: int = 800


@dataclass(frozen=True)
class RowLineCluster:
    """One detected row line, merged from one or more colinear/parallel
    Hough segments that share the dominant row angle and a perpendicular
    position."""

    angle_deg: float  # median angle (deg, mod 180) of the segments in this cluster
    offset_px: float  # median perpendicular-offset position of the cluster
    support_px: float  # summed segment length backing this row (pixel evidence)
    line_count: int  # number of Hough segments merged into this cluster


@dataclass(frozen=True)
class RowGeometryResult:
    thinning_method: str  # "guo_hall" or "zhang_suen_fallback"
    guo_hall_available: bool
    skeleton_pixel_count: int
    hough_line_count: int
    dominant_angle_deg: float | None  # robust (circular-median) Hough row angle
    row_clusters: list[RowLineCluster] = field(default_factory=list)
    num_consistent_rows: int = 0  # clusters meeting min_row_support_px
    row_line_support_px: float = 0.0  # strongest cluster's summed support
    structure_tensor_angle_deg: float | None = None
    structure_tensor_coherence: float = 0.0
    angle_agreement_deg: float | None = None  # |Hough angle - structure-tensor angle|, circular


# ---------------------------------------------------------------------------
# Mask cleanup
# ---------------------------------------------------------------------------


def clean_mask(mask: np.ndarray, cfg: RowGeometryConfig | None = None) -> np.ndarray:
    """Binarize + morphological open/close to remove speckle noise and close
    small gaps before skeletonization. Pure function: input mask untouched."""
    cfg = cfg or RowGeometryConfig()
    binary = (mask > 0).astype(np.uint8)
    if cfg.morph_open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_open_px, cfg.morph_open_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    if cfg.morph_close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_close_px, cfg.morph_close_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    return binary


# ---------------------------------------------------------------------------
# Thinning: Guo-Hall (preferred) with a documented Zhang-Suen fallback
# ---------------------------------------------------------------------------


def _zhang_suen_thinning(binary: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Zhang & Suen (1984), "A fast parallel algorithm for thinning digital
    patterns," Communications of the ACM 27(3). Classic two-subiteration
    topological thinning. Vectorized with numpy; operates on a {0,1} uint8
    array and returns a boolean skeleton.

    Not the same algorithm as Guo-Hall (Guo & Hall 1989) and not guaranteed
    numerically identical -- both are standard 1px-skeleton thinning
    algorithms with comparable qualitative output. Used ONLY when
    cv2.ximgproc (opencv-contrib) is unavailable; see module docstring.
    """
    img = (binary > 0).astype(np.uint8).copy()
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2 = p[0:-2, 1:-1]
            p3 = p[0:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, 0:-2]
            p8 = p[1:-1, 0:-2]
            p9 = p[0:-2, 0:-2]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            b_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            a_count = np.zeros_like(img, dtype=np.int32)
            for i in range(8):
                a_count += ((neighbors[i] == 0) & (neighbors[i + 1] == 1)).astype(np.int32)
            common = (img == 1) & (b_count >= 2) & (b_count <= 6) & (a_count == 1)
            if step == 0:
                cond = common & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = common & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            if np.any(cond):
                img[cond] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def thin_mask(mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Skeletonize a cleaned binary mask to ~1px centerlines.

    Returns (skeleton_bool_mask, method) where method is "guo_hall" when
    cv2.ximgproc.thinning(..., THINNING_GUOHALL) ran, or
    "zhang_suen_fallback" when opencv-contrib is unavailable in this
    environment and the pure-Python fallback ran instead. The method used
    is always reported explicitly -- callers must not assume Guo-Hall ran.
    """
    global _WARNED_NO_XIMGPROC
    binary = (mask > 0).astype(np.uint8)
    if _HAS_XIMGPROC:
        u8 = binary * 255
        skel = _cv2_ximgproc.thinning(u8, thinningType=_cv2_ximgproc.THINNING_GUOHALL)
        return skel > 0, "guo_hall"
    if not _WARNED_NO_XIMGPROC:
        log.warning(
            "cv2.ximgproc (opencv-contrib) is not installed in this environment "
            "(this project depends on opencv-python-headless, not "
            "opencv-contrib-python-headless) -- Guo-Hall thinning is unavailable. "
            "Falling back to a pure-Python Zhang-Suen thinning implementation. "
            "Install opencv-contrib-python-headless to use the paper-matching "
            "Guo-Hall method from Ullah, Islam & Bais 2024."
        )
        _WARNED_NO_XIMGPROC = True
    return _zhang_suen_thinning(binary), "zhang_suen_fallback"


# ---------------------------------------------------------------------------
# Probabilistic Hough Transform for row-line candidates
# ---------------------------------------------------------------------------


def hough_row_lines(skeleton: np.ndarray, cfg: RowGeometryConfig | None = None) -> np.ndarray:
    """Probabilistic Hough Transform (cv2.HoughLinesP) over a skeleton mask.
    Returns an (N, 4) float array of [x1, y1, x2, y2] segments (empty (0, 4)
    array if none found)."""
    cfg = cfg or RowGeometryConfig()
    u8 = (skeleton.astype(np.uint8)) * 255
    lines = cv2.HoughLinesP(
        u8,
        rho=cfg.hough_rho_px,
        theta=np.radians(cfg.hough_theta_deg),
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_line_length_px,
        maxLineGap=cfg.hough_max_line_gap_px,
    )
    if lines is None:
        return np.zeros((0, 4), dtype=np.float64)
    return lines.reshape(-1, 4).astype(np.float64)


def line_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Undirected line angle in degrees, mod 180 (a line and its reverse
    have the same angle)."""
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)


def _line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def _line_midpoint(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _circular_distance_deg(a: float, b: float, period: float = 180.0) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


def robust_median_angle_deg(angles_deg: np.ndarray, period: float = 180.0) -> float:
    """Discrete circular median of mod-`period` angles: the sample point
    with least total circular distance to every other sample.

    A plain numeric median is wrong for undirected line angles because it
    doesn't understand the wraparound at 0/period (e.g. angles 1 deg and
    179 deg are nearly identical directions, 2 deg apart circularly, but 178
    deg apart numerically). This discrete circular median is robust to
    outliers (like a numeric median) and correct across the wrap.
    """
    angles = np.asarray(angles_deg, dtype=np.float64) % period
    if angles.size == 0:
        raise ValueError("robust_median_angle_deg requires at least one angle")
    if angles.size == 1:
        return float(angles[0])
    diffs = np.abs(angles[:, None] - angles[None, :]) % period
    diffs = np.minimum(diffs, period - diffs)
    total = diffs.sum(axis=1)
    return float(angles[int(np.argmin(total))])


def _build_cluster(indices: list[int], offsets: np.ndarray, lengths: np.ndarray, angles: np.ndarray) -> RowLineCluster:
    idx = np.asarray(indices, dtype=int)
    return RowLineCluster(
        angle_deg=robust_median_angle_deg(angles[idx]),
        offset_px=float(np.median(offsets[idx])),
        support_px=float(np.sum(lengths[idx])),
        line_count=int(idx.size),
    )


def cluster_row_lines(
    lines: np.ndarray, cfg: RowGeometryConfig | None = None
) -> tuple[float | None, list[RowLineCluster]]:
    """Cluster Hough segments by angle then perpendicular position.

    1. Compute the robust (circular-median) dominant angle across all
       segments, then re-estimate it using only segments within
       `angle_cluster_tol_deg` of that first estimate (one round of
       trimming -- keeps a handful of near-perpendicular outlier segments
       from skewing the dominant-row-direction estimate).
    2. Project each row-aligned segment's midpoint onto the axis
       perpendicular to the dominant angle to get a 1D "which row" offset.
    3. Merge segments whose offsets are within `row_position_tol_px` of the
       next-nearest offset (sorted 1D gap clustering) into one
       RowLineCluster per physical row line.

    Returns (dominant_angle_deg | None, clusters) -- None / [] when `lines`
    is empty.
    """
    cfg = cfg or RowGeometryConfig()
    if lines.shape[0] == 0:
        return None, []

    angles = np.array([line_angle_deg(*l) for l in lines])
    dominant = robust_median_angle_deg(angles)
    close = np.array([_circular_distance_deg(a, dominant, 180.0) <= cfg.angle_cluster_tol_deg for a in angles])
    if np.any(close):
        dominant = robust_median_angle_deg(angles[close])
        close = np.array([_circular_distance_deg(a, dominant, 180.0) <= cfg.angle_cluster_tol_deg for a in angles])

    if not np.any(close):
        return dominant, []

    theta = np.radians(dominant)
    # Unit normal to the dominant row direction: separates parallel rows.
    nx, ny = -np.sin(theta), np.cos(theta)

    kept = np.nonzero(close)[0]
    offsets = np.array([_line_midpoint(*lines[i])[0] * nx + _line_midpoint(*lines[i])[1] * ny for i in kept])
    lengths = np.array([_line_length(*lines[i]) for i in kept])
    kept_angles = angles[kept]

    order = np.argsort(offsets)
    clusters: list[RowLineCluster] = []
    current = [int(order[0])]
    for j in order[1:]:
        j = int(j)
        if offsets[j] - offsets[current[-1]] <= cfg.row_position_tol_px:
            current.append(j)
        else:
            clusters.append(_build_cluster(current, offsets, lengths, kept_angles))
            current = [j]
    clusters.append(_build_cluster(current, offsets, lengths, kept_angles))
    return dominant, clusters


# ---------------------------------------------------------------------------
# Structure-tensor orientation/coherence (second, independent method)
# ---------------------------------------------------------------------------


def structure_tensor_orientation(
    image: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-9
) -> tuple[float | None, float]:
    """Dominant orientation + coherence via the 2D structure tensor of image
    gradients, evaluated over an optional weight mask (e.g. the field mask).

    gx, gy = Sobel gradients; J = [[sum(gx^2*w), sum(gx*gy*w)],
                                     [sum(gx*gy*w), sum(gy^2*w)]]
    theta = 0.5*atan2(2*Jxy, Jxx-Jyy)
    coherence = sqrt((Jxx-Jyy)^2 + 4*Jxy^2) / (Jxx+Jyy+eps)  in [0, 1]

    theta as computed above is the dominant *gradient* direction; the row
    direction runs perpendicular to it (rows are locally constant along
    their own axis, so the intensity gradient points across them), hence
    the + 90 deg rotation before returning. Returns (row_angle_deg, 0.0) ->
    (None, 0.0) when there is no usable signal (empty/uniform weighted
    region).

    This is a comparable, independent cross-check to the Hough-based
    dominant_angle_deg in RowGeometryResult -- the two methods measure
    orientation completely differently (voting over discrete line segments
    vs. an aggregate gradient-covariance statistic), so agreement between
    them is meaningful corroborating evidence.
    """
    g = image.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    w = np.ones_like(g) if weight is None else weight.astype(np.float32)

    jxx = float(np.sum(gx * gx * w))
    jyy = float(np.sum(gy * gy * w))
    jxy = float(np.sum(gx * gy * w))
    if jxx + jyy < 1e-6:
        return None, 0.0

    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    coherence = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + eps))
    row_angle = float(np.degrees(theta + np.pi / 2) % 180.0)
    return row_angle, coherence


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def analyze_row_geometry(mask: np.ndarray, cfg: RowGeometryConfig | None = None) -> RowGeometryResult:
    """Run the full row-geometry pipeline on a single binary crop mask:
    cleanup -> thinning (Guo-Hall or documented fallback) -> Probabilistic
    Hough Transform -> angle/position clustering, plus the independent
    structure-tensor orientation/coherence cross-check.

    Pure function: does not read cropmerge.config or touch
    cropmerge/pipeline/processor.py (not wired into the live pipeline yet,
    see module docstring).

    Masks larger than `cfg.max_dimension_px` on their longest side are
    downsampled before thinning/Hough (see RowGeometryConfig.max_dimension_px
    for why this matters in practice, not just in theory) -- support_px and
    offset_px on the returned clusters are scaled back up so callers always
    see original-image pixel units regardless of internal downsampling.
    """
    cfg = cfg or RowGeometryConfig()
    scale = 1.0
    h, w = mask.shape[:2]
    longest = max(h, w)
    if longest > cfg.max_dimension_px > 0:
        scale = cfg.max_dimension_px / longest
        small_size = (max(1, round(w * scale)), max(1, round(h * scale)))  # cv2.resize wants (w, h)
        mask = cv2.resize((mask > 0).astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST)

    cleaned = clean_mask(mask, cfg)
    skeleton, thin_method = thin_mask(cleaned)
    lines = hough_row_lines(skeleton, cfg)
    dominant_angle, clusters = cluster_row_lines(lines, cfg)

    if scale != 1.0:
        inv_scale = 1.0 / scale
        clusters = [
            RowLineCluster(
                angle_deg=c.angle_deg,
                offset_px=c.offset_px * inv_scale,
                support_px=c.support_px * inv_scale,
                line_count=c.line_count,
            )
            for c in clusters
        ]

    strong = [c for c in clusters if c.support_px >= cfg.min_row_support_px]

    st_angle, st_coherence = structure_tensor_orientation(cleaned.astype(np.float32))
    agreement = None
    if dominant_angle is not None and st_angle is not None:
        agreement = _circular_distance_deg(dominant_angle, st_angle, 180.0)

    return RowGeometryResult(
        thinning_method=thin_method,
        guo_hall_available=guo_hall_available(),
        skeleton_pixel_count=int(np.count_nonzero(skeleton)),
        hough_line_count=int(lines.shape[0]),
        dominant_angle_deg=dominant_angle,
        row_clusters=strong,
        num_consistent_rows=len(strong),
        row_line_support_px=max((c.support_px for c in strong), default=0.0),
        structure_tensor_angle_deg=st_angle,
        structure_tensor_coherence=st_coherence,
        angle_agreement_deg=agreement,
    )
