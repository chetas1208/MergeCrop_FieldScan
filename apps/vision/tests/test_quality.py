import numpy as np

from cropmerge.config import load_config
from cropmerge.video.quality import analyze_frame_quality


def test_sharp_frame_usable():
    cfg = load_config()
    # High-frequency pattern
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[::2, ::2] = 200
    img[1::2, 1::2] = 40
    fq, _ = analyze_frame_quality(img, 0, 0.0, cfg)
    assert fq.sharpness > 0.05
    assert 0 <= fq.exposure_score <= 1
    assert fq.quality_weight > 0


def test_blur_flagged():
    cfg = load_config()
    img = np.full((120, 160, 3), 128, dtype=np.uint8)
    fq, _ = analyze_frame_quality(img, 0, 0.0, cfg)
    assert fq.sharpness < 0.3
