from __future__ import annotations

from collections import deque

from cropmerge.anomaly.grid import GridCell
from cropmerge.anomaly.temporal import aggregate_zones
from cropmerge.pipeline.schemas import InspectionZone


class RollingZoneAggregator:
    """Reuses cropmerge.anomaly.temporal.aggregate_zones (unmodified,
    verbatim) over a bounded rolling window of recently analyzed live
    frames, instead of reinventing temporal consensus for live mode.
    """

    def __init__(self, cfg: dict, frame_shape: tuple[int, int], window_size: int = 20):
        self.cfg = cfg
        self.frame_shape = frame_shape
        self.window_size = window_size
        self._cells: deque[list[GridCell]] = deque(maxlen=window_size)
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._quality_weights: deque[float] = deque(maxlen=window_size)

    def add_frame(self, cells: list[GridCell], timestamp_sec: float, quality_weight: float = 1.0) -> None:
        self._cells.append(cells)
        self._timestamps.append(timestamp_sec)
        self._quality_weights.append(quality_weight)

    def current_zones(self) -> list[InspectionZone]:
        if not self._cells:
            return []
        return aggregate_zones(
            list(self._cells),
            list(self._timestamps),
            list(self._quality_weights),
            self.frame_shape,
            self.cfg,
        )
