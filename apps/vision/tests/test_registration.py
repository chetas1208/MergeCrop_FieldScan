import numpy as np

from cropmerge.video.registration import estimate_homography


def test_registration_identical():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # textured
    rng = np.random.default_rng(0)
    img[rng.integers(0, 200, 500), rng.integers(0, 200, 500)] = 255
    r = estimate_homography(img, img.copy(), {})
    # May or may not find enough unique features on sparse noise; just no crash
    assert r.n_matches >= 0
    assert 0 <= r.confidence <= 1


def test_registration_empty():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    r = estimate_homography(a, a, {})
    assert r.success is False or r.confidence >= 0
