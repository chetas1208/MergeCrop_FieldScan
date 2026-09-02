from cropmerge.segmentation.base import FrameSegmentation, SegmentMask, Segmenter
from cropmerge.segmentation.sam3 import create_segmenter, sam3_available

try:
    from cropmerge.segmentation.sam2 import sam2_available
except Exception:  # pragma: no cover
    def sam2_available() -> bool:
        return False

__all__ = [
    "FrameSegmentation",
    "SegmentMask",
    "Segmenter",
    "create_segmenter",
    "sam3_available",
    "sam2_available",
]
