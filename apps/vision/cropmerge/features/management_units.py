"""Management-unit segmentation via cheap grid-cell region merging.

Real feedback (Sept 2 field photo review): a field image commonly contains
multiple visually and agronomically distinct blocks -- different varieties,
treatments, or (as in the reported case) harvested residue sitting next to
standing canopy. Comparing anomaly evidence against a single whole-image
baseline treats a genuine block boundary as noise; the cropmerge/anomaly
/spatial.py local-baseline fix (2026-09-02) already mitigates this at the
INDIVIDUAL-CELL level (compare each cell to its own neighborhood), but does
not identify or report the actual management-unit boundaries themselves,
which is real information a farmer/agronomist can use directly (and a
building block for per-unit row detection, per-unit statistics, etc).

ARCHITECTURE, deliberately simple per the campaign's own instruction ("If
scikit-image SLIC/RAG is already a safe dependency, benchmark it. Otherwise
build a simple contiguous tile/region approach... Do not add heavy models
just for unit segmentation"): this reuses the existing grid-cell
infrastructure (cropmerge/anomaly/grid.py) rather than a general-purpose
superpixel library. Adjacent grid cells are merged into one management unit
when a boundary "cost" (weighted feature-difference across vegetation
fraction, residue-vs-soil-vs-crop signal, color, and row orientation) is
below a threshold -- a textbook region-adjacency-graph merge via
union-find, not a trained model.

SHADOW-ONLY / STANDALONE for this integration round, matching every other
new capability shipped tonight: this module does NOT alter any existing
Inspection Area, anomaly score, or field-summary computation. It is a new,
independently computable, independently testable diagnostic. Wiring
management-unit boundaries INTO the anomaly-scoring pipeline (e.g. gating
comparisons to within-unit baselines instead of the whole field) is real,
valuable follow-on work that deserves its own benchmarking against labeled
management-unit boundaries this session does not have -- not done here.

Engineering thresholds/weights below are priors, not scientifically
calibrated constants -- tune from real field review.

REAL CALIBRATION DATA (2026-09-02, run against the actual provided overhead
photo through the heuristic segmenter, not just the synthetic test fixtures
below): the default boundary_threshold=0.25 was too permissive for this
real image and collapsed everything into ONE unit. Sweeping the threshold
on that same real photo:

    threshold=0.02 -> 23 units (over-segmented -- noise-level splits)
    threshold=0.05 -> 10 units
    threshold=0.08 ->  3 units, areas [0.88, 0.10, 0.02]
    threshold=0.10 ->  1 unit  (under-segmented)

This confirms the algorithm IS sensitive to real feature differences (it is
not a no-op), but the synthetic test fixtures below use deliberately stark
color contrasts (pure green vs. pure tan) that are larger than this real
photo's actual zone-to-zone differences -- so a threshold tuned to "obviously
correct" on synthetic data is NOT automatically correct on real imagery.
0.25 stays the code default here (unchanged, not silently tuned from a
single uncontrolled data point) but ~0.08 is a more realistic real-world
starting point worth testing further; no ground-truth management-unit
labels exist yet to pick a definitively "correct" value from this one
sample. This whole finding is exactly why this module ships SHADOW-ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from cropmerge.anomaly.grid import GridCell, build_grid
from cropmerge.features.residue_detection import ResidueClass, classify_residue_evidence
from cropmerge.features.rgb_indices import color_stats, excess_green, vegetation_mask
from cropmerge.features.row_geometry import structure_tensor_orientation

_RESIDUE_SCORE = {
    ResidueClass.LIVING_VEGETATION: 0.0,
    ResidueClass.LIKELY_BARE_SOIL: 0.5,
    ResidueClass.LIKELY_RESIDUE: 1.0,
    ResidueClass.UNKNOWN: 0.25,
}


class UnitType(str, Enum):
    """What a management unit actually IS, agronomically -- the load-bearing
    gate for downstream anomaly scoring (a RESIDUE_STUBBLE/BARE_SOIL/NON_CROP
    unit must never produce a "Possible Crop Gap"/"Sparse Crop Coverage"
    finding, because it contains no standing crop to be sparse or gapped)."""

    ACTIVE_CROP = "active_crop"
    RESIDUE_STUBBLE = "residue_stubble"
    BARE_SOIL = "bare_soil"
    NON_CROP = "non_crop"
    UNKNOWN = "unknown"


# Vegetation fraction above which a unit is considered to have standing
# crop present, regardless of some residue-classified cells within it
# (e.g. crop rows over visible inter-row residue/mulch).
_ACTIVE_CROP_VEG_THRESHOLD = 0.20
_NON_CROP_LABEL_THRESHOLD = 0.5
_MIN_CELLS_FOR_FULL_CONFIDENCE = 6


def classify_unit_type(
    mean_vegetation_fraction: float,
    dominant_residue_class: ResidueClass,
    mean_non_crop_label_fraction: float,
) -> UnitType:
    """Deterministic, threshold-based classification -- no learned model,
    no confidence-free guess. Order matters: non-crop land cover (road,
    tree canopy, water, infrastructure) overrides everything else, since a
    unit can't simultaneously be a farmed block and a farm road."""
    if mean_non_crop_label_fraction >= _NON_CROP_LABEL_THRESHOLD:
        return UnitType.NON_CROP
    if mean_vegetation_fraction >= _ACTIVE_CROP_VEG_THRESHOLD:
        return UnitType.ACTIVE_CROP
    if dominant_residue_class == ResidueClass.LIKELY_RESIDUE:
        return UnitType.RESIDUE_STUBBLE
    if dominant_residue_class == ResidueClass.LIKELY_BARE_SOIL:
        return UnitType.BARE_SOIL
    return UnitType.UNKNOWN


def unit_type_confidence(
    members: list[CellFeatures],
    unit_type: UnitType,
    dominant_residue_class: ResidueClass,
) -> float:
    """Confidence based on real, observable internal consistency -- never a
    fabricated neural score. Two real signals, both in [0, 1]:

    1. Semantic agreement: fraction of member cells whose OWN residue_class
       (or vegetation regime, for ACTIVE_CROP) agrees with the unit's
       assigned type. A unit built from cells that mostly agree is more
       trustworthy than one built from a bare majority.
    2. Spatial support: cell_count relative to a minimum sample size
       (_MIN_CELLS_FOR_FULL_CONFIDENCE) -- a 1-cell "unit" is definitionally
       less confident than a well-sampled one, independent of agreement.

    Final confidence is the product of the two -- either one being weak
    caps the result, which is the honest behavior (a large but internally
    inconsistent unit should not read as confident, nor should a small but
    perfectly uniform one)."""
    if not members:
        return 0.0

    if unit_type == UnitType.ACTIVE_CROP:
        agree = sum(1 for m in members if m.vegetation_fraction >= _ACTIVE_CROP_VEG_THRESHOLD)
    elif unit_type == UnitType.RESIDUE_STUBBLE:
        agree = sum(1 for m in members if m.residue_class == ResidueClass.LIKELY_RESIDUE)
    elif unit_type == UnitType.BARE_SOIL:
        agree = sum(1 for m in members if m.residue_class == ResidueClass.LIKELY_BARE_SOIL)
    elif unit_type == UnitType.NON_CROP:
        agree = sum(1 for m in members if m.non_crop_label_fraction >= _NON_CROP_LABEL_THRESHOLD)
    else:
        agree = sum(1 for m in members if m.residue_class == ResidueClass.UNKNOWN)
    agreement_fraction = agree / len(members)

    spatial_support = min(1.0, len(members) / _MIN_CELLS_FOR_FULL_CONFIDENCE)

    return agreement_fraction * spatial_support


@dataclass(frozen=True)
class UnitBoundaryWeights:
    vegetation: float = 0.35
    residue: float = 0.25
    color: float = 0.20
    row_orientation: float = 0.20


_NON_CROP_LABELS = {"ROAD_PATH", "TREE_VEGETATION", "WATER", "INFRASTRUCTURE"}


@dataclass(frozen=True)
class CellFeatures:
    row: int
    col: int
    valid: bool
    vegetation_fraction: float
    residue_class: ResidueClass
    residue_score: float
    color_mean_bgr: tuple[float, float, float]
    row_angle_deg: float | None
    row_coherence: float
    crop_label_fraction: float = 0.0  # fraction of cell classified CROP by the segmentation label_map
    non_crop_label_fraction: float = 0.0  # ROAD_PATH/TREE_VEGETATION/WATER/INFRASTRUCTURE


@dataclass(frozen=True)
class ManagementUnit:
    unit_id: int
    cell_count: int
    area_fraction: float  # of the analyzable field, not the whole image
    mean_vegetation_fraction: float
    dominant_residue_class: str
    mean_row_angle_deg: float | None
    unit_type: UnitType = UnitType.UNKNOWN
    confidence: float = 0.0


@dataclass
class UnitSegmentationResult:
    rows: int
    cols: int
    cell_unit_ids: list[list[int]] = field(default_factory=list)  # [row][col] -> unit_id, -1 for invalid cells
    units: list[ManagementUnit] = field(default_factory=list)


def compute_cell_features(
    bgr: np.ndarray,
    field_mask: np.ndarray,
    label_map: np.ndarray,
    cells: list[GridCell],
    min_field_fraction: float = 0.25,
) -> list[CellFeatures]:
    exg_full = excess_green(bgr)
    veg_full = vegetation_mask(exg_full, 0.05)
    features: list[CellFeatures] = []

    for cell in cells:
        region_mask = np.zeros(field_mask.shape, dtype=bool)
        region_mask[cell.y0 : cell.y1, cell.x0 : cell.x1] = field_mask[cell.y0 : cell.y1, cell.x0 : cell.x1]
        cell_field_frac = float(np.mean(field_mask[cell.y0 : cell.y1, cell.x0 : cell.x1]))
        valid = cell_field_frac >= min_field_fraction

        if not valid or not np.any(region_mask):
            features.append(
                CellFeatures(
                    row=cell.row, col=cell.col, valid=False, vegetation_fraction=0.0,
                    residue_class=ResidueClass.UNKNOWN, residue_score=0.0,
                    color_mean_bgr=(0.0, 0.0, 0.0), row_angle_deg=None, row_coherence=0.0,
                )
            )
            continue

        veg_frac = float(np.mean(veg_full[region_mask]))
        residue = classify_residue_evidence(bgr, region_mask)
        stats = color_stats(bgr, region_mask)
        color_mean = (stats.get("b_mean", 0.0), stats.get("g_mean", 0.0), stats.get("r_mean", 0.0))

        if label_map is not None:
            region_labels = label_map[region_mask]
            crop_label_frac = float(np.mean(region_labels == "CROP")) if region_labels.size else 0.0
            non_crop_mask = np.isin(region_labels, list(_NON_CROP_LABELS))
            non_crop_label_frac = float(np.mean(non_crop_mask)) if region_labels.size else 0.0
        else:
            crop_label_frac, non_crop_label_frac = 0.0, 0.0

        crop_region = region_mask & (label_map == "CROP") if label_map is not None else region_mask
        angle, coherence = None, 0.0
        if np.count_nonzero(crop_region) >= 100:
            sub = crop_region[cell.y0 : cell.y1, cell.x0 : cell.x1].astype(np.float32)
            angle, coherence = structure_tensor_orientation(sub)

        features.append(
            CellFeatures(
                row=cell.row, col=cell.col, valid=True, vegetation_fraction=veg_frac,
                residue_class=residue.classification, residue_score=_RESIDUE_SCORE[residue.classification],
                color_mean_bgr=color_mean, row_angle_deg=angle, row_coherence=coherence,
                crop_label_fraction=crop_label_frac, non_crop_label_fraction=non_crop_label_frac,
            )
        )
    return features


def _circular_angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff) / 90.0  # normalized to [0, 1]


def boundary_cost(a: CellFeatures, b: CellFeatures, weights: UnitBoundaryWeights | None = None) -> float:
    """Weighted feature-difference between two adjacent cells -- LOW cost
    means "looks like the same management unit," HIGH cost means "likely a
    real boundary." Returns 1.0 (maximum, definite boundary) if either cell
    is invalid (no field/too little field fraction) -- an invalid cell
    never merges with anything."""
    w = weights or UnitBoundaryWeights()
    if not a.valid or not b.valid:
        return 1.0

    d_veg = abs(a.vegetation_fraction - b.vegetation_fraction)
    d_residue = abs(a.residue_score - b.residue_score)
    color_diff = np.array(a.color_mean_bgr) - np.array(b.color_mean_bgr)
    d_color = float(np.linalg.norm(color_diff)) / 255.0
    if a.row_angle_deg is not None and b.row_angle_deg is not None:
        d_theta = _circular_angle_diff(a.row_angle_deg, b.row_angle_deg)
    else:
        d_theta = 0.0

    return w.vegetation * d_veg + w.residue * d_residue + w.color * d_color + w.row_orientation * d_theta


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


def segment_management_units(
    bgr: np.ndarray,
    field_mask: np.ndarray,
    label_map: np.ndarray,
    *,
    rows: int = 8,
    cols: int = 8,
    boundary_threshold: float = 0.25,
    weights: UnitBoundaryWeights | None = None,
) -> UnitSegmentationResult:
    """Segment the field into contiguous management units via grid-cell
    region merging. Adjacent cells (4-connected) merge into the same unit
    when boundary_cost() is below `boundary_threshold`; the result is one
    or more connected components, each reported with real aggregate
    statistics -- never fabricated region names or counts."""
    h, w = bgr.shape[:2]
    cells = build_grid(h, w, rows, cols)
    features = compute_cell_features(bgr, field_mask, label_map, cells)
    feature_by_pos = {(f.row, f.col): f for f in features}

    n = len(cells)
    index_of = {(c.row, c.col): i for i, c in enumerate(cells)}
    uf = _UnionFind(n)

    for c in cells:
        i = index_of[(c.row, c.col)]
        fi = feature_by_pos[(c.row, c.col)]
        for dr, dc in ((0, 1), (1, 0)):  # right neighbor, down neighbor -- avoids double-processing pairs
            neighbor_pos = (c.row + dr, c.col + dc)
            if neighbor_pos not in index_of:
                continue
            j = index_of[neighbor_pos]
            fj = feature_by_pos[neighbor_pos]
            if boundary_cost(fi, fj, weights) < boundary_threshold:
                uf.union(i, j)

    root_of_cell = [uf.find(i) for i in range(n)]
    valid_cell_count = sum(1 for f in features if f.valid)

    root_to_unit_id: dict[int, int] = {}
    cell_unit_ids = [[-1] * cols for _ in range(rows)]
    unit_cells: dict[int, list[CellFeatures]] = {}

    for c, root, f in zip(cells, root_of_cell, features):
        if not f.valid:
            continue
        unit_id = root_to_unit_id.setdefault(root, len(root_to_unit_id))
        cell_unit_ids[c.row][c.col] = unit_id
        unit_cells.setdefault(unit_id, []).append(f)

    units: list[ManagementUnit] = []
    for unit_id, members in sorted(unit_cells.items()):
        residue_counts: dict[ResidueClass, int] = {}
        for m in members:
            residue_counts[m.residue_class] = residue_counts.get(m.residue_class, 0) + 1
        dominant = max(residue_counts, key=lambda k: residue_counts[k])
        angles = [m.row_angle_deg for m in members if m.row_angle_deg is not None]
        mean_veg = float(np.mean([m.vegetation_fraction for m in members]))
        mean_non_crop = float(np.mean([m.non_crop_label_fraction for m in members]))
        u_type = classify_unit_type(mean_veg, dominant, mean_non_crop)
        confidence = unit_type_confidence(members, u_type, dominant)
        units.append(
            ManagementUnit(
                unit_id=unit_id,
                cell_count=len(members),
                area_fraction=len(members) / valid_cell_count if valid_cell_count else 0.0,
                mean_vegetation_fraction=mean_veg,
                dominant_residue_class=dominant.value,
                mean_row_angle_deg=float(np.mean(angles)) if angles else None,
                unit_type=u_type,
                confidence=confidence,
            )
        )

    return UnitSegmentationResult(rows=rows, cols=cols, cell_unit_ids=cell_unit_ids, units=units)
