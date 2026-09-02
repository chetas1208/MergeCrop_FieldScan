from __future__ import annotations

import numpy as np

from cropmerge.features.residue_detection import ResidueClass, classify_residue_evidence

TAN_BGR = (90, 140, 180)  # brown/tan in BGR order, hue ~17 (within brown range)
GREEN_BGR = (40, 160, 40)


def _flat_patch(bgr_color: tuple[int, int, int], h: int = 60, w: int = 60) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr_color
    return img


def _striped_patch(bgr_color: tuple[int, int, int], h: int = 60, w: int = 60) -> np.ndarray:
    """Alternating bright/dark stripes of the same hue -- creates real edge
    density (linear texture), the residue proxy signal."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        shade = 1.0 if (y // 3) % 2 == 0 else 0.55
        img[y, :] = tuple(int(c * shade) for c in bgr_color)
    return img


def test_strongly_green_region_is_living_vegetation():
    bgr = _flat_patch(GREEN_BGR)
    mask = np.ones((60, 60), dtype=bool)

    result = classify_residue_evidence(bgr, mask)

    assert result.classification == ResidueClass.LIVING_VEGETATION
    assert result.living_vegetation_fraction > 0.5


def test_smooth_tan_region_is_likely_bare_soil():
    bgr = _flat_patch(TAN_BGR)
    mask = np.ones((60, 60), dtype=bool)

    result = classify_residue_evidence(bgr, mask)

    assert result.classification == ResidueClass.LIKELY_BARE_SOIL
    assert result.brown_hue_fraction > 0.5


def test_textured_tan_region_is_likely_residue():
    bgr = _striped_patch(TAN_BGR)
    mask = np.ones((60, 60), dtype=bool)

    result = classify_residue_evidence(bgr, mask)

    assert result.classification == ResidueClass.LIKELY_RESIDUE
    assert result.edge_density > 0.0


def test_empty_mask_is_unknown_not_an_error():
    bgr = _flat_patch(TAN_BGR)
    mask = np.zeros((60, 60), dtype=bool)

    result = classify_residue_evidence(bgr, mask)

    assert result.classification == ResidueClass.UNKNOWN
    assert "empty" in result.confidence_note


def test_non_brown_non_green_region_is_unknown():
    """Flat gray/blue (e.g. water, tarp, infrastructure) -- neither
    vegetation nor tan/brown -- must not be forced into residue or soil."""
    bgr = _flat_patch((160, 100, 60))  # blue-ish in BGR
    mask = np.ones((60, 60), dtype=bool)

    result = classify_residue_evidence(bgr, mask)

    assert result.classification == ResidueClass.UNKNOWN


def test_never_confidently_classifies_ambiguous_low_texture_signal():
    """A tan region with edge density sitting between the residue and soil
    thresholds must stay UNKNOWN rather than being forced either way."""
    from cropmerge.features.residue_detection import ResidueThresholds, classify_residue_evidence

    bgr = _striped_patch(TAN_BGR, h=60, w=60)
    mask = np.ones((60, 60), dtype=bool)
    # Force a threshold gap that this fixture's real edge_density should
    # fall inside of, proving the ambiguous branch is reachable.
    from cropmerge.features.texture import texture_features

    real_edge_density = texture_features(bgr, mask)["edge_density"]
    thresholds = ResidueThresholds(
        residue_edge_density_min=real_edge_density + 0.05,
        soil_edge_density_max=real_edge_density - 0.05,
    )

    result = classify_residue_evidence(bgr, mask, thresholds)

    assert result.classification == ResidueClass.UNKNOWN
    assert "ambiguous" in result.confidence_note
