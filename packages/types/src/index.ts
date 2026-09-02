/** Shared domain types for CropMerge Field Triage (TypeScript-first contract). */

export type ReviewPriority = 'low' | 'medium' | 'high'

export type RelativeLocation =
  | 'northwest'
  | 'north'
  | 'northeast'
  | 'west'
  | 'center'
  | 'east'
  | 'southwest'
  | 'south'
  | 'southeast'

export type SemanticClass =
  | 'FIELD'
  | 'CROP'
  | 'BARE_SOIL'
  | 'ROAD_PATH'
  | 'TREE_VEGETATION'
  | 'WATER'
  | 'INFRASTRUCTURE'
  | 'UNKNOWN'

export type AnalysisStatus =
  | 'queued'
  | 'preparing'
  | 'processing'
  | 'rendering'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type SegmentationBackend = 'sam3' | 'heuristic' | 'mock'
export type DinoBackend = 'dinov3' | 'heuristic' | 'mock'

export type InspectionZoneType =
  | 'stand_gap'
  | 'sparse_canopy'
  | 'exposed_soil'
  | 'color_variation'
  | 'texture_variation'
  | 'water_like_region'
  | 'row_discontinuity'
  | 'general_visual_variation'

export interface ZoneEvidence {
  cropCoverageDelta?: number
  colorDifference?: number
  vegetationDifference?: number
  textureDifference?: number
  embeddingDifference?: number
  persistence?: number
  appearanceAnomalyScore?: number
  structuralAnomalyScore?: number
  rowContinuityBefore?: number
  rowContinuityAfter?: number
  gapExtentNormalized?: number
  soilExposureDelta?: number
  fragmentationScore?: number
  registrationConfidence?: number
}

export interface InspectionZone {
  id: string
  reviewPriority: ReviewPriority
  anomalyScore: number
  appearanceAnomalyScore?: number
  structuralAnomalyScore?: number
  reviewScore?: number
  persistenceScore: number
  primaryType?: InspectionZoneType
  primarySignalLabel?: string
  persistentObservations?: number
  totalObservations?: number
  firstSeenMs: number
  lastSeenMs: number
  firstSeenSec: number
  lastSeenSec: number
  framesSeen: number
  relativeLocation: RelativeLocation
  centroidNorm: { x: number; y: number }
  bboxNorm: { x: number; y: number; w: number; h: number }
  evidence: ZoneEvidence
  reasons: string[]
  recommendation: string
  /** Optional local-LLM-rewritten farmer-facing note. Only present when
   * enrichment was enabled and the model produced output — never a
   * placeholder. Absent means: show `reasons` instead. */
  llmSummary?: string | null
}

export interface CropCoverageDetail {
  estimatedFraction: number
  analyzableFraction: number
  segmentationConfidence?: number | null
  uncertainFraction?: number
  bareSoilFraction?: number
  nonCropFraction?: number
}

export interface FieldBoundaryInfo {
  label?: string
  source?: string
  confidence?: string
  derivation?: string
  tooltip?: string
}

export type FarmTechObservabilityMode = 'PLANT_RESOLVABLE' | 'ROW_RESOLVABLE' | 'CANOPY_ONLY'

export interface FarmTechVegetation {
  exgMean: number
  variMean: number
}

export interface FarmTechRowGeometry {
  method: string
  orientationDeg?: number | null
  coherence: number
  rowCount: number
  supportPx: number
}

export interface FarmTechStructure {
  cropOccupancy: number
  soilFraction: number
  fragmentation: number
}

/** Shadow-only per-observation FarmTech evidence — recorded for review, not
 * yet authoritative. Never shown prominently in primary farmer UI copy. */
export interface FarmTechObservation {
  mode: FarmTechObservabilityMode
  vegetation: FarmTechVegetation
  rowGeometry?: FarmTechRowGeometry | null
  structure: FarmTechStructure
}

export interface FrameQuality {
  frameIndex: number
  timestampSec: number
  sharpness: number
  exposureScore: number
  meanLuminance: number
  nearBlackFraction: number
  saturatedFraction: number
  usable: boolean
  qualityWeight: number
  warnings: string[]
  farmTech?: FarmTechObservation | null
}

export interface VideoSourceMeta {
  filename: string
  path?: string
  durationSec: number
  width: number
  height: number
  fps: number
  frameCount: number
  codec?: string | null
  orientation?: number | null
  createdAt?: string | null
  gps?: {
    latitude?: number | null
    longitude?: number | null
    altitude?: number | null
  } | null
}

export interface FieldSummary {
  detected: boolean
  meanCropCoverage: number
  meanBareSoil: number
  roadPathDetected: boolean
  treeVegetationDetected: boolean
  waterDetected: boolean
  infrastructureDetected: boolean
  meanFieldFraction: number
  cropCoverage?: CropCoverageDetail | null
  boundary?: FieldBoundaryInfo | null
  rowVisibility?: string
}

export interface AnalysisSummary {
  framesSampled: number
  framesUsable: number
  sampleFps: number
  segmentationBackend: SegmentationBackend
  dinoBackend: DinoBackend
  usedFallback: boolean
  device: string
  stageLatencySec: Record<string, number>
  totalRuntimeSec: number
}

export interface ArtifactPaths {
  resultsJson: string
  annotatedVideo?: string | null
  heatmapPng?: string | null
  metricsJson?: string | null
  framesDir?: string | null
  overlaysDir?: string | null
}

/** Deterministic, template-generated synthesis of a completed run — every
 * sentence traces to a real number elsewhere in the report. Never LLM-authored
 * (the optional local-LLM layer may polish `headline`/zone `llmSummary` text
 * but must not invent evidence beyond what's already computed here). */
export interface FieldAnalysisSummary {
  scheduledObservations: number
  usableObservations: number
  limitedObservations: number
  highPriorityCount: number
  mediumPriorityCount: number
  lowPriorityCount: number
  highestPriorityZoneId?: string | null
  headline: string
  keyFindings: string[]
  limitations: string[]
}

export interface FieldTriageReport {
  schemaVersion: '1.0'
  runId: string
  createdAt: string
  disclaimer: string
  source: VideoSourceMeta
  analysis: AnalysisSummary
  field: FieldSummary
  inspectionZones: InspectionZone[]
  frameQuality: FrameQuality[]
  classCoverage: Partial<Record<SemanticClass, number>>
  limitations: string[]
  artifacts: ArtifactPaths
  georeferenced: false
  mapLabel: string
  summary?: FieldAnalysisSummary | null
  sourceSha256?: string | null
  storagePolicyVersion: string
  farmTechVersion: string
}

export interface AnalysisJob {
  id: string
  status: AnalysisStatus
  createdAt: string
  updatedAt: string
  filename: string
  progress: number
  stage?: string | null
  message?: string
  error?: string
  report?: FieldTriageReport | null
  artifacts?: ArtifactPaths | null
  /** Direct, short-lived URLs served by the vision host, never by Nuxt/Vercel. */
  artifactUrls?: Record<string, string>
}

export interface CreateAnalysisRequest {
  sampleFps?: number
  maxFrames?: number
  skipDino?: boolean
  segmentationBackend?: SegmentationBackend
  debug?: boolean
}

export interface VisionHealth {
  status: 'ok' | 'degraded'
  service?: string
  version: string
  device: string
  segmentationBackend: SegmentationBackend
  dinoBackend: DinoBackend
  samAvailable: boolean
  dinoAvailable: boolean
  torchAvailable: boolean
  opencvAvailable: boolean
  ffmpegAvailable: boolean
  gpuBusy?: boolean
  gpuAvailable?: boolean
  gpuMessage?: string | null
  activeJobId?: string | null
  queueDepth?: number
  maxQueueDepth?: number
}

// ---------------------------------------------------------------------------
// Live Drone mode (Docker Compose only — never reachable from the Vercel
// deploy of apps/web). See docs/LIVE_MODE_LIMITATIONS.md.
// ---------------------------------------------------------------------------

export type LiveSessionStatus =
  | 'disconnected'
  | 'waiting'
  | 'connecting'
  | 'live'
  | 'stream_lost'
  | 'reconnecting'
  | 'ended'

export type TelemetryAvailability = 'exact' | 'nearby' | 'unavailable'

export interface LiveSession {
  id: string
  createdAt: string
  endedAt?: string | null
  status: LiveSessionStatus
  streamPath: string
  sampleFps: number
  simulated: boolean
  whepUrl: string
}

/**
 * Fields are null, never fabricated, when the DJI Cloud API message did not
 * carry them. `source: 'unavailable'` means no MQTT telemetry has been
 * received for this session at all (e.g. simulated live input).
 */
export interface DroneTelemetry {
  timestampMs: number
  latitude: number | null
  longitude: number | null
  altitudeM: number | null
  headingDeg: number | null
  gimbalPitchDeg: number | null
  source: 'dji-cloud-api' | 'unavailable'
}

export interface TelemetryAssociation {
  availability: TelemetryAvailability
  telemetry?: DroneTelemetry | null
  toleranceMs: number
}

export interface LiveInspectionAreaEvent extends InspectionZone {
  liveFrameTimestampMs: number
  telemetryAssociation: TelemetryAssociation
}

export type LiveStreamEvent =
  | { type: 'frame_analyzed'; sessionId: string; frameTimestampMs: number; zones: LiveInspectionAreaEvent[] }
  | { type: 'state_change'; sessionId: string; status: LiveSessionStatus }
  | { type: 'telemetry'; sessionId: string; telemetry: DroneTelemetry }
  | { type: 'error'; sessionId: string; message: string }

export interface SavedInspectionArea {
  id: string
  sessionId: string
  zone: InspectionZone
  telemetryAssociation: TelemetryAssociation
  snapshotUrl?: string
  savedAt: string
  note?: string
}

export const DISCLAIMER =
  'CropMerge Field Triage analyzes visual patterns in RGB imagery. Flagged regions represent differences from surrounding field appearance and are intended for human review. They are not diagnoses of crop disease, nutrient status, irrigation failure, or plant health.'

export const DEFAULT_LIMITATIONS = [
  'RGB-only visual analysis',
  'No agronomic diagnosis inferred',
  'Image-relative field map (not georeferenced unless GPS metadata present)',
  'Exploratory visual anomaly scores are not calibrated probabilities',
] as const
