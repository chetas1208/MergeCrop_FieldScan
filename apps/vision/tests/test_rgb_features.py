import numpy as np
import pytest

from cropmerge.features.rgb_indices import color_stats, excess_green, lab_distance, vari, vegetation_mask
from cropmerge.features.texture import texture_features


def test_exg_higher_on_green():
    green = np.zeros((32, 32, 3), dtype=np.uint8)
    green[:, :] = (20, 180, 20)  # BGR
    brown = np.zeros((32, 32, 3), dtype=np.uint8)
    brown[:, :] = (40, 80, 140)
    assert float(np.mean(excess_green(green))) > float(np.mean(excess_green(brown)))


def test_vegetation_mask():
    green = np.zeros((16, 16, 3), dtype=np.uint8)
    green[:] = (30, 200, 30)
    m = vegetation_mask(excess_green(green), 0.05)
    assert float(np.mean(m)) > 0.9


def test_vari_higher_on_green():
    # Gitelson et al. 2002: VARI = (G-R)/(G+R-B) — hand-calculable case.
    green = np.zeros((16, 16, 3), dtype=np.uint8)
    green[:] = (20, 180, 20)  # BGR -> R=20,G=180,B=20 -> (180-20)/(180+20-20)=0.888...
    brown = np.zeros((16, 16, 3), dtype=np.uint8)
    brown[:] = (40, 80, 140)  # BGR -> R=140,G=80,B=40 -> (80-140)/(80+140-40)=-0.333...
    v_green = float(np.mean(vari(green)))
    v_brown = float(np.mean(vari(brown)))
    assert v_green > v_brown
    assert v_green == pytest.approx(0.8888, rel=1e-3)
    assert v_brown == pytest.approx(-0.3333, rel=1e-3)


def test_color_stats_and_lab():
    a = np.zeros((20, 20, 3), dtype=np.uint8)
    a[:] = (50, 150, 50)
    b = np.zeros((20, 20, 3), dtype=np.uint8)
    b[:] = (100, 100, 100)
    sa, sb = color_stats(a), color_stats(b)
    assert sa["g_mean"] > sb["g_mean"]
    assert lab_distance(sa, sb) >= 0


def test_texture_nonzero():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    t = texture_features(img)
    assert t["variance"] >= 0
    assert 0 <= t["edge_density"] <= 1
