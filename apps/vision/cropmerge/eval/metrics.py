"""Segmentation evaluation metrics against pixel-level ground truth."""
from __future__ import annotations

from typing import Any

import numpy as np

from cropmerge.pipeline.schemas import SemanticClass

EVAL_CLASSES: tuple[str, ...] = (
    SemanticClass.CROP.value,
    SemanticClass.BARE_SOIL.value,
    SemanticClass.ROAD_PATH.value,
    SemanticClass.TREE_VEGETATION.value,
    SemanticClass.WATER.value,
    SemanticClass.INFRASTRUCTURE.value,
    SemanticClass.UNKNOWN.value,
)


def _as_class_array(label_map: np.ndarray) -> np.ndarray:
    flat = label_map.reshape(-1)
    out = np.empty(flat.size, dtype=np.int16)
    class_to_idx = {name: i for i, name in enumerate(EVAL_CLASSES)}
    unknown_idx = class_to_idx[SemanticClass.UNKNOWN.value]
    for i, px in enumerate(flat):
        key = str(px)
        out[i] = class_to_idx.get(key, unknown_idx)
    return out.reshape(label_map.shape[:2])


def confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    classes: tuple[str, ...] = EVAL_CLASSES,
) -> np.ndarray:
    pred_idx = _as_class_array(pred)
    gt_idx = _as_class_array(gt)
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for p, g in zip(pred_idx.ravel(), gt_idx.ravel()):
        cm[g, p] += 1
    return cm


def per_class_scores(cm: np.ndarray) -> dict[str, dict[str, float | None]]:
    scores: dict[str, dict[str, float | None]] = {}
    for i, name in enumerate(EVAL_CLASSES):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        support = float(cm[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        union = tp + fp + fn
        iou = tp / union if union > 0 else None
        scores[name] = {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_pixels": support,
        }
    return scores


def summarize(cm: np.ndarray, classes: tuple[str, ...] = EVAL_CLASSES) -> dict[str, Any]:
    per_class = per_class_scores(cm)
    total = float(cm.sum())
    correct = float(np.trace(cm))
    pixel_accuracy = correct / total if total > 0 else 0.0

    ious = [v["iou"] for v in per_class.values() if v["iou"] is not None and v["support_pixels"] > 0]
    miou = float(np.mean(ious)) if ious else 0.0

    macro_f1_vals = [v["f1"] for v in per_class.values() if v["f1"] is not None and v["support_pixels"] > 0]
    macro_f1 = float(np.mean(macro_f1_vals)) if macro_f1_vals else 0.0

    return {
        "pixel_accuracy": round(pixel_accuracy, 4),
        "mean_iou": round(miou, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            k: {
                kk: (round(vv, 4) if isinstance(vv, float) else vv)
                for kk, vv in v.items()
            }
            for k, v in per_class.items()
            if v["support_pixels"] > 0
        },
        "confusion_matrix": {
            "classes": list(classes),
            "matrix": cm.astype(int).tolist(),
        },
    }


def compare_label_maps(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(f"Shape mismatch: pred {pred.shape[:2]} vs gt {gt.shape[:2]}")
    cm = confusion_matrix(pred, gt)
    return summarize(cm)
