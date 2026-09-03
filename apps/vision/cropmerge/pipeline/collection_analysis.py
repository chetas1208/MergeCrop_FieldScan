"""Collection-level analysis: run every valid image in a CAS-registered
image collection independently, then aggregate group-level statistics.

Each collection image is its OWN independent observation -- there is no
cross-image temporal zone tracking (unlike a video's 1 FPS observation
sequence). A collection image never overlaps physical area with another the
way consecutive video frames do, so no attempt is made to "stitch" or union
results into one field-wide number; only per-image results and per-group
robust statistics (median/IQR, not a naive mean that one bad image could
skew) are reported, per the campaign's own "do not double-count area across
overlapping images" and "do not display false precision" requirements.

CORRECTNESS-FIRST IMPLEMENTATION for tonight's field-readiness campaign:
calls FieldTriageProcessor.process() once per image, which (per its
existing, unmodified, heavily-tested internals) constructs its own
segmentation/embedding backend on every call -- this does NOT yet satisfy
the "one model-load lifecycle per collection" performance target from the
campaign spec. Doing that safely means threading an optional pre-built
segmenter/embedder through processor.py's core process() method, which is
the single most heavily used, most heavily tested entry point in this
codebase (the live production JobManager's own analysis path) -- a real,
valuable, separate change deserving its own careful, reviewed pass, not a
rushed edit at the end of an already-large session. See Decisions.md.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cropmerge.pipeline.processor import FieldTriageProcessor

log = logging.getLogger("cropmerge.pipeline.collection_analysis")


@dataclass(frozen=True)
class CollectionImageInput:
    image_id: str
    relative_group: str | None
    path: Path


@dataclass(frozen=True)
class CollectionImageResult:
    image_id: str
    relative_group: str | None
    status: str  # "analyzed" | "failed"
    error: str | None = None
    field_detected: bool | None = None
    crop_coverage: float | None = None
    bare_soil_fraction: float | None = None
    run_id: str | None = None  # FieldTriageReport.run_id, for artifact lookup


@dataclass(frozen=True)
class GroupStatistics:
    group: str
    image_count: int
    crop_coverage_median: float | None
    crop_coverage_iqr: tuple[float, float] | None
    bare_soil_median: float | None
    bare_soil_iqr: tuple[float, float] | None


@dataclass
class CollectionAnalysisResult:
    collection_id: str
    image_results: list[CollectionImageResult] = field(default_factory=list)
    group_statistics: list[GroupStatistics] = field(default_factory=list)

    @property
    def analyzed_count(self) -> int:
        return sum(1 for r in self.image_results if r.status == "analyzed")

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.image_results if r.status == "failed")


def _robust_stats(values: list[float]) -> tuple[float, tuple[float, float]]:
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    iqr = (float(np.percentile(arr, 25)), float(np.percentile(arr, 75)))
    return median, iqr


def _compute_group_statistics(image_results: list[CollectionImageResult]) -> list[GroupStatistics]:
    groups: dict[str, list[CollectionImageResult]] = defaultdict(list)
    for r in image_results:
        if r.status == "analyzed" and r.crop_coverage is not None:
            groups[r.relative_group or "ungrouped"].append(r)

    stats: list[GroupStatistics] = []
    for group_name in sorted(groups):
        results = groups[group_name]
        cov_median, cov_iqr = _robust_stats([r.crop_coverage for r in results])  # type: ignore[misc]
        soil_median, soil_iqr = _robust_stats([r.bare_soil_fraction or 0.0 for r in results])
        stats.append(
            GroupStatistics(
                group=group_name,
                image_count=len(results),
                crop_coverage_median=cov_median,
                crop_coverage_iqr=cov_iqr,
                bare_soil_median=soil_median,
                bare_soil_iqr=soil_iqr,
            )
        )
    return stats


def analyze_collection_images(
    collection_id: str,
    images: list[CollectionImageInput],
    output_root: Path,
    cfg: dict,
    *,
    segmentation_backend: str = "heuristic",
    dino_backend: str = "heuristic",
) -> CollectionAnalysisResult:
    """Run every image independently through the existing single-image
    analysis pipeline. Never lets one image's failure abort the whole
    collection -- a failed image is recorded with status="failed" and an
    error message, matching this codebase's established never-block-the-
    real-result pattern; every OTHER image still gets analyzed.
    """
    result = CollectionAnalysisResult(collection_id=collection_id)

    for img in images:
        try:
            report = FieldTriageProcessor(cfg).process(
                img.path,
                output_root,
                sample_fps=1.0,
                max_frames=1,
                segmentation_backend=segmentation_backend,
                dino_backend=dino_backend,
                skip_dino=True,
                run_id=img.image_id,
            )
            result.image_results.append(
                CollectionImageResult(
                    image_id=img.image_id,
                    relative_group=img.relative_group,
                    status="analyzed",
                    field_detected=report.field.detected,
                    crop_coverage=report.field.mean_crop_coverage,
                    bare_soil_fraction=report.field.mean_bare_soil,
                    run_id=report.run_id,
                )
            )
        except Exception as exc:
            log.exception("Collection image analysis failed: collection=%s image=%s", collection_id, img.image_id)
            result.image_results.append(
                CollectionImageResult(
                    image_id=img.image_id,
                    relative_group=img.relative_group,
                    status="failed",
                    error=str(exc),
                )
            )

    result.group_statistics = _compute_group_statistics(result.image_results)
    return result
