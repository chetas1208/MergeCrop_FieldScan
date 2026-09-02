from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GridCell:
    row: int
    col: int
    y0: int
    y1: int
    x0: int
    x1: int
    field_fraction: float
    valid: bool
    features: dict[str, float] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    anomaly_score: float = 0.0  # combined review base = max(appearance, structural)
    appearance_anomaly_score: float = 0.0
    structural_anomaly_score: float = 0.0
    contributions: dict[str, float] = field(default_factory=dict)
    row_visibility: str = "LOW"


def build_grid(h: int, w: int, rows: int, cols: int) -> list[GridCell]:
    cells: list[GridCell] = []
    for r in range(rows):
        for c in range(cols):
            y0 = int(r * h / rows)
            y1 = int((r + 1) * h / rows)
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)
            cells.append(GridCell(r, c, y0, y1, x0, x1, 0.0, False))
    return cells


def cell_mask(field_mask: np.ndarray, cell: GridCell) -> np.ndarray:
    return field_mask[cell.y0 : cell.y1, cell.x0 : cell.x1]


def robust_baseline(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(arr))


def normalize_scores(raw: np.ndarray) -> np.ndarray:
    """Min-max to [0,1]; constant → zeros."""
    if raw.size == 0:
        return raw
    lo, hi = float(np.min(raw)), float(np.max(raw))
    if hi - lo < 1e-9:
        return np.zeros_like(raw)
    return (raw - lo) / (hi - lo)
