import numpy as np

from cropmerge.config import load_config
from cropmerge.segmentation.heuristic import HeuristicSegmenter
from cropmerge.segmentation.postprocess import build_field_mask, class_fractions, resolve_overlaps
from cropmerge.pipeline.schemas import SemanticClass
from cropmerge.segmentation.base import SegmentMask


def test_heuristic_on_green_field():
    cfg = load_config()
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[:] = (30, 170, 40)
    img[80:100, :] = (70, 90, 110)  # road-ish
    seg = HeuristicSegmenter(cfg).segment_image(img, 0, 0.0)
    assert seg.is_fallback
    assert seg.field_mask is not None
    assert seg.label_map is not None
    fr = class_fractions(seg.label_map)
    assert fr["CROP"] > 0.2


def test_overlap_priority():
    h, w = 40, 40
    crop = np.ones((h, w), dtype=bool)
    road = np.zeros((h, w), dtype=bool)
    road[10:15, :] = True
    masks = [
        SegmentMask(SemanticClass.CROP, crop, 1.0),
        SegmentMask(SemanticClass.ROAD_PATH, road, 1.0),
    ]
    lm = resolve_overlaps(masks, (h, w))
    assert lm[12, 20] == SemanticClass.ROAD_PATH.value


def test_field_mask_largest_component():
    lm = np.full((50, 50), SemanticClass.UNKNOWN.value, dtype=object)
    lm[5:40, 5:40] = SemanticClass.CROP.value
    lm[0:3, 0:3] = SemanticClass.CROP.value
    fm = build_field_mask(lm)
    assert float(np.mean(fm)) > 0.3
