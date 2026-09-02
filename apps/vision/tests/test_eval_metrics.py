import numpy as np

from cropmerge.eval.metrics import compare_label_maps, confusion_matrix
from cropmerge.eval.synthetic_field import make_synthetic_labels, make_synthetic_pair
from cropmerge.pipeline.schemas import SemanticClass


def test_perfect_match_pixel_accuracy():
    gt = make_synthetic_labels(1.0)
    report = compare_label_maps(gt, gt)
    assert report["pixel_accuracy"] == 1.0
    assert report["mean_iou"] == 1.0


def test_confusion_matrix_shape():
    gt = make_synthetic_labels(0.0)
    pred = gt.copy()
    pred[0, 0] = SemanticClass.ROAD_PATH.value
    cm = confusion_matrix(pred, gt)
    assert cm.ndim == 2
    assert cm.sum() == gt.size


def test_synthetic_pair_shapes_match():
    bgr, labels = make_synthetic_pair(2.5)
    assert bgr.shape[:2] == labels.shape
    assert bgr.shape[2] == 3


def test_classes_present_in_synthetic_gt():
    labels = make_synthetic_labels(1.0)
    uniq = set(labels.reshape(-1).tolist())
    assert SemanticClass.CROP.value in uniq
    assert SemanticClass.BARE_SOIL.value in uniq
    assert SemanticClass.ROAD_PATH.value in uniq
    assert SemanticClass.TREE_VEGETATION.value in uniq
