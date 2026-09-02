from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from cropmerge.pipeline.schemas import SemanticClass


@dataclass
class SegmentMask:
    label: SemanticClass
    mask: np.ndarray  # bool HxW
    confidence: float = 1.0
    prompt: str | None = None
    track_id: int | None = None


@dataclass
class FrameSegmentation:
    frame_index: int
    timestamp_sec: float
    masks: list[SegmentMask] = field(default_factory=list)
    label_map: np.ndarray | None = None  # int HxW class ids
    field_mask: np.ndarray | None = None
    crop_mask: np.ndarray | None = None
    backend: str = "unknown"
    is_fallback: bool = False


class Segmenter(ABC):
    """Pluggable segmentation backend (SAM 3.1 / heuristic / mock)."""

    name: str = "base"

    @abstractmethod
    def segment_image(self, bgr: np.ndarray, frame_index: int = 0, timestamp_sec: float = 0.0) -> FrameSegmentation:
        ...

    def start_video(self, frames_bgr: list[np.ndarray]) -> None:
        """Optional video session init for tracking backends."""

    def track_video(self, frames_meta: list[tuple[int, float]]) -> list[FrameSegmentation]:
        """Optional: track across video after start_video."""
        raise NotImplementedError

    def close(self) -> None:
        return None
