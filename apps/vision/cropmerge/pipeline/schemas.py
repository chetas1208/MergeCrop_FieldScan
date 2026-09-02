from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Named policy/module versions attached to every completed report so a
# result stays forensically reconstructible as these subsystems evolve (a
# result never becomes silently ambiguous about which formula/policy
# version produced it). Bump the relevant constant whenever that
# subsystem's actual behavior changes, not on every unrelated edit.
STORAGE_POLICY_VERSION = "fieldscan-storage-v1"
FARMTECH_VERSION = "farmtech-v1"
# Bump manually when processor.py's algorithm changes in a way not already
# captured by a configs/*.yaml value (e.g. a fusion-formula bugfix) --
# cropmerge/storage/analysis_cache.py combines this with a hash of the
# actual loaded config content, so config-only changes invalidate the
# result cache automatically without relying on someone remembering to
# bump a version string for those.
ANALYSIS_VERSION = "analysis-v1"


class ReviewPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RelativeLocation(str, Enum):
    NW = "northwest"
    N = "north"
    NE = "northeast"
    W = "west"
    C = "center"
    E = "east"
    SW = "southwest"
    S = "south"
    SE = "southeast"


class SemanticClass(str, Enum):
    FIELD = "FIELD"
    CROP = "CROP"
    BARE_SOIL = "BARE_SOIL"
    ROAD_PATH = "ROAD_PATH"
    TREE_VEGETATION = "TREE_VEGETATION"
    WATER = "WATER"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    UNKNOWN = "UNKNOWN"


class InspectionZoneType(str, Enum):
    STAND_GAP = "stand_gap"
    SPARSE_CANOPY = "sparse_canopy"
    EXPOSED_SOIL = "exposed_soil"
    COLOR_VARIATION = "color_variation"
    TEXTURE_VARIATION = "texture_variation"
    WATER_LIKE_REGION = "water_like_region"
    ROW_DISCONTINUITY = "row_discontinuity"
    GENERAL_VISUAL_VARIATION = "general_visual_variation"


class ZoneEvidence(BaseModel):
    crop_coverage_delta: float | None = None
    color_difference: float | None = None
    vegetation_difference: float | None = None
    texture_difference: float | None = None
    embedding_difference: float | None = None
    persistence: float | None = None
    appearance_anomaly_score: float | None = None
    structural_anomaly_score: float | None = None
    row_continuity_before: float | None = None
    row_continuity_after: float | None = None
    gap_extent_normalized: float | None = None
    soil_exposure_delta: float | None = None
    fragmentation_score: float | None = None
    registration_confidence: float | None = None


class InspectionZone(BaseModel):
    id: str
    review_priority: ReviewPriority
    anomaly_score: float = Field(ge=0, le=1)
    appearance_anomaly_score: float = Field(default=0.0, ge=0, le=1)
    structural_anomaly_score: float = Field(default=0.0, ge=0, le=1)
    review_score: float = Field(default=0.0, ge=0, le=1)
    persistence_score: float = Field(ge=0, le=1)
    primary_type: InspectionZoneType = InspectionZoneType.GENERAL_VISUAL_VARIATION
    primary_signal_label: str = "General visual variation"
    persistent_observations: int = 0
    total_observations: int = 0
    first_seen_ms: float
    last_seen_ms: float
    first_seen_sec: float
    last_seen_sec: float
    frames_seen: int
    relative_location: RelativeLocation
    centroid_norm: dict[str, float]
    bbox_norm: dict[str, float]
    evidence: ZoneEvidence
    reasons: list[str]
    recommendation: str


class CropCoverageDetail(BaseModel):
    estimated_fraction: float
    analyzable_fraction: float
    segmentation_confidence: float | None = None
    uncertain_fraction: float = 0.0
    bare_soil_fraction: float = 0.0
    non_crop_fraction: float = 0.0


class FieldBoundaryInfo(BaseModel):
    label: str = "Analysis Field Boundary"
    source: str = "Vision estimate"
    confidence: str = "Medium"
    derivation: str = "Segmentation union of crop, bare soil, and field classes with conservative cleanup"
    tooltip: str = (
        "Estimated boundary of the field region currently included in the visual analysis. "
        "This is derived from imagery and is not a surveyed, parcel, or property boundary."
    )


class FieldSummary(BaseModel):
    detected: bool
    mean_crop_coverage: float
    mean_bare_soil: float
    road_path_detected: bool
    tree_vegetation_detected: bool
    water_detected: bool
    infrastructure_detected: bool
    mean_field_fraction: float
    crop_coverage: CropCoverageDetail | None = None
    boundary: FieldBoundaryInfo | None = None
    row_visibility: str = "LOW"


class FarmTechVegetation(BaseModel):
    """Chromatic vegetation-evidence indices — never health/NDVI, see
    docs/FARMTECH_RESEARCH_BASIS.md."""

    exg_mean: float
    vari_mean: float


class FarmTechRowGeometry(BaseModel):
    """Row-structure diagnostic from cropmerge/features/row_geometry.py.
    method reports which thinning implementation actually ran ("guo_hall"
    or "zhang_suen_fallback") — never hidden, see FARMTECH_RESEARCH_BASIS.md."""

    method: str
    orientation_deg: float | None
    coherence: float
    row_count: int
    support_px: float


class FarmTechStructure(BaseModel):
    crop_occupancy: float
    soil_fraction: float
    fragmentation: float
    # Visible-vegetation-in-soil-area proxy — see
    # cropmerge/features/weed_pressure.py's module docstring for the hard
    # physical limitation this respects: vegetation occluded beneath closed
    # crop canopy is not observable in overhead RGB imagery, so this can
    # only ever be a "visible weed cover" signal, never a total/hidden one.
    understory_vegetation_fraction: float | None = None
    vegetated_soil_fraction_of_field: float | None = None
    # Residue/stubble vs bare-soil vs unknown, over the BARE_SOIL-classified
    # field area — see cropmerge/features/residue_detection.py's module
    # docstring for why this is conservative-by-design (UNKNOWN whenever
    # evidence is ambiguous, never a forced confident call). One of:
    # "living_vegetation" | "likely_residue" | "likely_bare_soil" | "unknown"
    # | None (no BARE_SOIL area to evaluate this observation).
    residue_classification: str | None = None
    residue_confidence_note: str | None = None


class FarmTechObservation(BaseModel):
    """Shadow-only FarmTech measurements for one 1 FPS observation — see
    cropmerge/pipeline/farmtech_shadow.py. Gated by CROP_MERGE_FARMTECH_SHADOW
    (default off). Computed and recorded for review, but MUST NEVER be used
    to alter an InspectionZone's type/priority/score/ranking — this campaign
    round only records evidence, it does not yet activate it. `mode` is the
    observability tier (cropmerge/features/observability.py) that gates
    which claims the imagery actually supports; spacing metrics are omitted
    entirely (no plant detector exists yet, see FARMTECH_RESEARCH_BASIS.md)."""

    mode: str  # "PLANT_RESOLVABLE" | "ROW_RESOLVABLE" | "CANOPY_ONLY"
    vegetation: FarmTechVegetation
    row_geometry: FarmTechRowGeometry | None = None
    structure: FarmTechStructure


class FrameQuality(BaseModel):
    frame_index: int
    timestamp_sec: float
    sharpness: float
    exposure_score: float
    mean_luminance: float
    near_black_fraction: float
    saturated_fraction: float
    usable: bool
    quality_weight: float
    warnings: list[str] = Field(default_factory=list)
    farm_tech: FarmTechObservation | None = None


class VideoSourceMeta(BaseModel):
    filename: str
    path: str | None = None
    duration_sec: float
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str | None = None
    orientation: int | None = None
    created_at: str | None = None
    gps: dict[str, float | None] | None = None


class AnalysisSummary(BaseModel):
    frames_sampled: int
    frames_usable: int
    sample_fps: float
    segmentation_backend: str
    dino_backend: str
    used_fallback: bool
    device: str
    stage_latency_sec: dict[str, float]
    total_runtime_sec: float


class ArtifactPaths(BaseModel):
    results_json: str
    annotated_video: str | None = None
    heatmap_png: str | None = None
    metrics_json: str | None = None
    frames_dir: str | None = None
    overlays_dir: str | None = None


def zone_to_camel_dict(z: InspectionZone) -> dict[str, Any]:
    """Emit TS-contract keys (camelCase) for a single InspectionZone.

    Shared by FieldTriageReport.to_camel_dict() and Live Drone event
    serialization (cropmerge/live/schemas.py) so the two modes never drift
    on the farmer-facing zone shape.
    """
    return {
        "id": z.id,
        "reviewPriority": z.review_priority.value,
        "anomalyScore": z.anomaly_score,
        "appearanceAnomalyScore": z.appearance_anomaly_score,
        "structuralAnomalyScore": z.structural_anomaly_score,
        "reviewScore": z.review_score,
        "persistenceScore": z.persistence_score,
        "primaryType": z.primary_type.value,
        "primarySignalLabel": z.primary_signal_label,
        "persistentObservations": z.persistent_observations,
        "totalObservations": z.total_observations,
        "firstSeenMs": z.first_seen_ms,
        "lastSeenMs": z.last_seen_ms,
        "firstSeenSec": z.first_seen_sec,
        "lastSeenSec": z.last_seen_sec,
        "framesSeen": z.frames_seen,
        "relativeLocation": z.relative_location.value,
        "centroidNorm": z.centroid_norm,
        "bboxNorm": z.bbox_norm,
        "evidence": {
            "cropCoverageDelta": z.evidence.crop_coverage_delta,
            "colorDifference": z.evidence.color_difference,
            "vegetationDifference": z.evidence.vegetation_difference,
            "textureDifference": z.evidence.texture_difference,
            "embeddingDifference": z.evidence.embedding_difference,
            "persistence": z.evidence.persistence,
            "appearanceAnomalyScore": z.evidence.appearance_anomaly_score,
            "structuralAnomalyScore": z.evidence.structural_anomaly_score,
            "rowContinuityBefore": z.evidence.row_continuity_before,
            "rowContinuityAfter": z.evidence.row_continuity_after,
            "gapExtentNormalized": z.evidence.gap_extent_normalized,
            "soilExposureDelta": z.evidence.soil_exposure_delta,
            "fragmentationScore": z.evidence.fragmentation_score,
            "registrationConfidence": z.evidence.registration_confidence,
        },
        "reasons": z.reasons,
        "recommendation": z.recommendation,
    }


def farm_tech_to_camel_dict(ft: FarmTechObservation | None) -> dict[str, Any] | None:
    """Emit TS-contract keys for one observation's shadow FarmTech block, or
    None when shadow computation is disabled/unavailable for this frame."""
    if ft is None:
        return None
    return {
        "mode": ft.mode,
        "vegetation": {
            "exgMean": ft.vegetation.exg_mean,
            "variMean": ft.vegetation.vari_mean,
        },
        "rowGeometry": (
            {
                "method": ft.row_geometry.method,
                "orientationDeg": ft.row_geometry.orientation_deg,
                "coherence": ft.row_geometry.coherence,
                "rowCount": ft.row_geometry.row_count,
                "supportPx": ft.row_geometry.support_px,
            }
            if ft.row_geometry is not None
            else None
        ),
        "structure": {
            "cropOccupancy": ft.structure.crop_occupancy,
            "soilFraction": ft.structure.soil_fraction,
            "fragmentation": ft.structure.fragmentation,
            "understoryVegetationFraction": ft.structure.understory_vegetation_fraction,
            "vegetatedSoilFractionOfField": ft.structure.vegetated_soil_fraction_of_field,
            "residueClassification": ft.structure.residue_classification,
            "residueConfidenceNote": ft.structure.residue_confidence_note,
        },
    }


class FieldAnalysisSummary(BaseModel):
    """Deterministic, template-generated synthesis of a completed run — every
    sentence here is built from already-computed report fields, never from an
    LLM. See cropmerge/pipeline/summary.py for the builder. The optional local
    LLM (cropmerge/enrich/) may polish `headline`/`key_findings` prose but
    must never invent evidence not present in this object.
    """

    scheduled_observations: int
    usable_observations: int
    limited_observations: int
    high_priority_count: int
    medium_priority_count: int
    low_priority_count: int
    highest_priority_zone_id: str | None = None
    headline: str
    key_findings: list[str]
    limitations: list[str]

    def to_camel_dict(self) -> dict[str, Any]:
        return {
            "scheduledObservations": self.scheduled_observations,
            "usableObservations": self.usable_observations,
            "limitedObservations": self.limited_observations,
            "highPriorityCount": self.high_priority_count,
            "mediumPriorityCount": self.medium_priority_count,
            "lowPriorityCount": self.low_priority_count,
            "highestPriorityZoneId": self.highest_priority_zone_id,
            "headline": self.headline,
            "keyFindings": self.key_findings,
            "limitations": self.limitations,
        }


class FieldTriageReport(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    created_at: str
    disclaimer: str
    source: VideoSourceMeta
    analysis: AnalysisSummary
    field: FieldSummary
    inspection_zones: list[InspectionZone]
    frame_quality: list[FrameQuality]
    class_coverage: dict[str, float]
    limitations: list[str]
    artifacts: ArtifactPaths
    georeferenced: bool = False
    map_label: str = "Image-relative field map — not georeferenced"
    summary: FieldAnalysisSummary | None = None
    source_sha256: str | None = None
    storage_policy_version: str = STORAGE_POLICY_VERSION
    farm_tech_version: str = FARMTECH_VERSION
    analysis_version: str = ANALYSIS_VERSION
    config_version: str | None = None

    def to_camel_dict(self) -> dict[str, Any]:
        """Emit TS-contract keys (camelCase) for Nuxt."""
        return {
            "schemaVersion": self.schema_version,
            "runId": self.run_id,
            "createdAt": self.created_at,
            "disclaimer": self.disclaimer,
            "sourceSha256": self.source_sha256,
            "storagePolicyVersion": self.storage_policy_version,
            "farmTechVersion": self.farm_tech_version,
            "analysisVersion": self.analysis_version,
            "configVersion": self.config_version,
            "source": {
                "filename": self.source.filename,
                "path": self.source.path,
                "durationSec": self.source.duration_sec,
                "width": self.source.width,
                "height": self.source.height,
                "fps": self.source.fps,
                "frameCount": self.source.frame_count,
                "codec": self.source.codec,
                "orientation": self.source.orientation,
                "createdAt": self.source.created_at,
                "gps": self.source.gps,
            },
            "analysis": {
                "framesSampled": self.analysis.frames_sampled,
                "framesUsable": self.analysis.frames_usable,
                "sampleFps": self.analysis.sample_fps,
                "segmentationBackend": self.analysis.segmentation_backend,
                "dinoBackend": self.analysis.dino_backend,
                "usedFallback": self.analysis.used_fallback,
                "device": self.analysis.device,
                "stageLatencySec": self.analysis.stage_latency_sec,
                "totalRuntimeSec": self.analysis.total_runtime_sec,
            },
            "field": {
                "detected": self.field.detected,
                "meanCropCoverage": self.field.mean_crop_coverage,
                "meanBareSoil": self.field.mean_bare_soil,
                "roadPathDetected": self.field.road_path_detected,
                "treeVegetationDetected": self.field.tree_vegetation_detected,
                "waterDetected": self.field.water_detected,
                "infrastructureDetected": self.field.infrastructure_detected,
                "meanFieldFraction": self.field.mean_field_fraction,
                "cropCoverage": (
                    {
                        "estimatedFraction": self.field.crop_coverage.estimated_fraction,
                        "analyzableFraction": self.field.crop_coverage.analyzable_fraction,
                        "segmentationConfidence": self.field.crop_coverage.segmentation_confidence,
                        "uncertainFraction": self.field.crop_coverage.uncertain_fraction,
                        "bareSoilFraction": self.field.crop_coverage.bare_soil_fraction,
                        "nonCropFraction": self.field.crop_coverage.non_crop_fraction,
                    }
                    if self.field.crop_coverage
                    else None
                ),
                "boundary": (
                    {
                        "label": self.field.boundary.label,
                        "source": self.field.boundary.source,
                        "confidence": self.field.boundary.confidence,
                        "derivation": self.field.boundary.derivation,
                        "tooltip": self.field.boundary.tooltip,
                    }
                    if self.field.boundary
                    else None
                ),
                "rowVisibility": self.field.row_visibility,
            },
            "inspectionZones": [zone_to_camel_dict(z) for z in self.inspection_zones],
            "frameQuality": [
                {
                    "frameIndex": q.frame_index,
                    "timestampSec": q.timestamp_sec,
                    "sharpness": q.sharpness,
                    "exposureScore": q.exposure_score,
                    "meanLuminance": q.mean_luminance,
                    "nearBlackFraction": q.near_black_fraction,
                    "saturatedFraction": q.saturated_fraction,
                    "usable": q.usable,
                    "qualityWeight": q.quality_weight,
                    "warnings": q.warnings,
                    "farmTech": farm_tech_to_camel_dict(q.farm_tech),
                }
                for q in self.frame_quality
            ],
            "classCoverage": self.class_coverage,
            "limitations": self.limitations,
            "artifacts": {
                "resultsJson": self.artifacts.results_json,
                "annotatedVideo": self.artifacts.annotated_video,
                "heatmapPng": self.artifacts.heatmap_png,
                "metricsJson": self.artifacts.metrics_json,
                "framesDir": self.artifacts.frames_dir,
                "overlaysDir": self.artifacts.overlays_dir,
            },
            "georeferenced": False,
            "mapLabel": self.map_label,
            "summary": self.summary.to_camel_dict() if self.summary else None,
        }
