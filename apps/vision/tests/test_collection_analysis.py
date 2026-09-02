from __future__ import annotations

from pathlib import Path

import cv2

from cropmerge.config import load_config
from cropmerge.eval.synthetic_field import make_synthetic_frame
from cropmerge.pipeline.collection_analysis import (
    CollectionImageInput,
    analyze_collection_images,
)


def _write_synthetic_image(path: Path, t: float) -> None:
    frame = make_synthetic_frame(t)
    cv2.imwrite(str(path), frame)


def test_analyzes_each_image_independently(tmp_path: Path):
    cfg = load_config()
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    _write_synthetic_image(img_a, 0.5)
    _write_synthetic_image(img_b, 1.5)

    images = [
        CollectionImageInput(image_id="img1", relative_group=None, path=img_a),
        CollectionImageInput(image_id="img2", relative_group=None, path=img_b),
    ]

    result = analyze_collection_images("col1", images, tmp_path / "outputs", cfg, segmentation_backend="heuristic")

    assert result.analyzed_count == 2
    assert result.failed_count == 0
    for r in result.image_results:
        assert r.status == "analyzed"
        assert r.crop_coverage is not None
        assert r.run_id == r.image_id  # run_id passed through explicitly


def test_failed_image_does_not_abort_the_rest(tmp_path: Path):
    cfg = load_config()
    good_img = tmp_path / "good.jpg"
    _write_synthetic_image(good_img, 0.5)

    images = [
        CollectionImageInput(image_id="img1", relative_group=None, path=good_img),
        CollectionImageInput(image_id="img2", relative_group=None, path=tmp_path / "does_not_exist.jpg"),
    ]

    result = analyze_collection_images("col1", images, tmp_path / "outputs", cfg, segmentation_backend="heuristic")

    assert result.analyzed_count == 1
    assert result.failed_count == 1
    failed = next(r for r in result.image_results if r.status == "failed")
    assert failed.error is not None
    assert failed.image_id == "img2"


def test_group_statistics_computed_per_group_with_robust_median_iqr(tmp_path: Path):
    cfg = load_config()
    images = []
    for i in range(4):
        path = tmp_path / f"a{i}.jpg"
        _write_synthetic_image(path, 0.3 + i * 0.2)
        images.append(CollectionImageInput(image_id=f"a{i}", relative_group="variation-a", path=path))
    for i in range(3):
        path = tmp_path / f"b{i}.jpg"
        _write_synthetic_image(path, 1.0 + i * 0.2)
        images.append(CollectionImageInput(image_id=f"b{i}", relative_group="variation-b", path=path))

    result = analyze_collection_images("col1", images, tmp_path / "outputs", cfg, segmentation_backend="heuristic")

    assert result.analyzed_count == 7
    groups = {g.group: g for g in result.group_statistics}
    assert set(groups.keys()) == {"variation-a", "variation-b"}
    assert groups["variation-a"].image_count == 4
    assert groups["variation-b"].image_count == 3
    for g in groups.values():
        assert g.crop_coverage_median is not None
        assert g.crop_coverage_iqr[0] <= g.crop_coverage_median <= g.crop_coverage_iqr[1]


def test_ungrouped_images_fall_into_ungrouped_bucket(tmp_path: Path):
    cfg = load_config()
    path = tmp_path / "a.jpg"
    _write_synthetic_image(path, 0.5)
    images = [CollectionImageInput(image_id="a", relative_group=None, path=path)]

    result = analyze_collection_images("col1", images, tmp_path / "outputs", cfg, segmentation_backend="heuristic")

    assert len(result.group_statistics) == 1
    assert result.group_statistics[0].group == "ungrouped"


def test_group_stats_skip_failed_images(tmp_path: Path):
    cfg = load_config()
    good_path = tmp_path / "good.jpg"
    _write_synthetic_image(good_path, 0.5)
    images = [
        CollectionImageInput(image_id="good", relative_group="variation-a", path=good_path),
        CollectionImageInput(image_id="bad", relative_group="variation-a", path=tmp_path / "missing.jpg"),
    ]

    result = analyze_collection_images("col1", images, tmp_path / "outputs", cfg, segmentation_backend="heuristic")

    assert result.group_statistics[0].image_count == 1  # only the successfully-analyzed image counted
