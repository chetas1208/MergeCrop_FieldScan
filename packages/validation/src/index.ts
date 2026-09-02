import { z } from 'zod'

export const reviewPrioritySchema = z.enum(['low', 'medium', 'high'])

export const relativeLocationSchema = z.enum([
  'northwest',
  'north',
  'northeast',
  'west',
  'center',
  'east',
  'southwest',
  'south',
  'southeast',
])

export const semanticClassSchema = z.enum([
  'FIELD',
  'CROP',
  'BARE_SOIL',
  'ROAD_PATH',
  'TREE_VEGETATION',
  'WATER',
  'INFRASTRUCTURE',
  'UNKNOWN',
])

export const analysisStatusSchema = z.enum([
  'queued',
  'preparing',
  'processing',
  'rendering',
  'completed',
  'failed',
  'cancelled',
])

export const zoneEvidenceSchema = z.object({
  cropCoverageDelta: z.number().optional(),
  colorDifference: z.number().optional(),
  vegetationDifference: z.number().optional(),
  textureDifference: z.number().optional(),
  embeddingDifference: z.number().optional(),
  persistence: z.number().optional(),
  appearanceAnomalyScore: z.number().optional(),
  structuralAnomalyScore: z.number().optional(),
  rowContinuityBefore: z.number().optional(),
  rowContinuityAfter: z.number().optional(),
  gapExtentNormalized: z.number().optional(),
  soilExposureDelta: z.number().optional(),
  fragmentationScore: z.number().optional(),
  registrationConfidence: z.number().optional(),
})

export const inspectionZoneSchema = z.object({
  id: z.string(),
  reviewPriority: reviewPrioritySchema,
  anomalyScore: z.number().min(0).max(1),
  appearanceAnomalyScore: z.number().min(0).max(1).optional(),
  structuralAnomalyScore: z.number().min(0).max(1).optional(),
  reviewScore: z.number().min(0).max(1).optional(),
  persistenceScore: z.number().min(0).max(1),
  primaryType: z.string().optional(),
  primarySignalLabel: z.string().optional(),
  persistentObservations: z.number().int().optional(),
  totalObservations: z.number().int().optional(),
  firstSeenMs: z.number(),
  lastSeenMs: z.number(),
  firstSeenSec: z.number(),
  lastSeenSec: z.number(),
  framesSeen: z.number().int().nonnegative(),
  relativeLocation: relativeLocationSchema,
  centroidNorm: z.object({ x: z.number(), y: z.number() }),
  bboxNorm: z.object({
    x: z.number(),
    y: z.number(),
    w: z.number(),
    h: z.number(),
  }),
  evidence: zoneEvidenceSchema,
  reasons: z.array(z.string()),
  recommendation: z.string(),
  llmSummary: z.string().nullish(),
})

export const cropCoverageDetailSchema = z.object({
  estimatedFraction: z.number(),
  analyzableFraction: z.number(),
  segmentationConfidence: z.number().nullable().optional(),
  uncertainFraction: z.number().optional(),
  bareSoilFraction: z.number().optional(),
  nonCropFraction: z.number().optional(),
})

export const fieldBoundaryInfoSchema = z.object({
  label: z.string().optional(),
  source: z.string().optional(),
  confidence: z.string().optional(),
  derivation: z.string().optional(),
  tooltip: z.string().optional(),
})

export const farmTechObservabilityModeSchema = z.enum([
  'PLANT_RESOLVABLE',
  'ROW_RESOLVABLE',
  'CANOPY_ONLY',
])

export const farmTechVegetationSchema = z.object({
  exgMean: z.number(),
  variMean: z.number(),
})

export const farmTechRowGeometrySchema = z.object({
  method: z.string(),
  orientationDeg: z.number().nullish(),
  coherence: z.number(),
  rowCount: z.number().int(),
  supportPx: z.number(),
})

export const farmTechStructureSchema = z.object({
  cropOccupancy: z.number(),
  soilFraction: z.number(),
  fragmentation: z.number(),
})

export const farmTechObservationSchema = z.object({
  mode: farmTechObservabilityModeSchema,
  vegetation: farmTechVegetationSchema,
  rowGeometry: farmTechRowGeometrySchema.nullish(),
  structure: farmTechStructureSchema,
})

export const frameQualitySchema = z.object({
  frameIndex: z.number().int(),
  timestampSec: z.number(),
  sharpness: z.number(),
  exposureScore: z.number(),
  meanLuminance: z.number(),
  nearBlackFraction: z.number(),
  saturatedFraction: z.number(),
  usable: z.boolean(),
  qualityWeight: z.number(),
  warnings: z.array(z.string()),
  farmTech: farmTechObservationSchema.nullish(),
})

export const videoSourceMetaSchema = z.object({
  filename: z.string(),
  path: z.string().optional(),
  durationSec: z.number(),
  width: z.number().int(),
  height: z.number().int(),
  fps: z.number(),
  frameCount: z.number().int(),
  codec: z.string().nullish(),
  orientation: z.number().nullish(),
  createdAt: z.string().nullish(),
  gps: z
    .object({
      latitude: z.number().nullish(),
      longitude: z.number().nullish(),
      altitude: z.number().nullish(),
    })
    .nullish(),
})

export const fieldSummarySchema = z.object({
  detected: z.boolean(),
  meanCropCoverage: z.number(),
  meanBareSoil: z.number(),
  roadPathDetected: z.boolean(),
  treeVegetationDetected: z.boolean(),
  waterDetected: z.boolean(),
  infrastructureDetected: z.boolean(),
  meanFieldFraction: z.number(),
  cropCoverage: cropCoverageDetailSchema.nullish(),
  boundary: fieldBoundaryInfoSchema.nullish(),
  rowVisibility: z.string().optional(),
})

export const analysisSummarySchema = z.object({
  framesSampled: z.number().int(),
  framesUsable: z.number().int(),
  sampleFps: z.number(),
  segmentationBackend: z.enum(['sam3', 'heuristic', 'mock']),
  dinoBackend: z.enum(['dinov3', 'heuristic', 'mock']),
  usedFallback: z.boolean(),
  device: z.string(),
  stageLatencySec: z.record(z.number()),
  totalRuntimeSec: z.number(),
})

export const artifactPathsSchema = z.object({
  resultsJson: z.string(),
  annotatedVideo: z.string().nullish(),
  heatmapPng: z.string().nullish(),
  metricsJson: z.string().nullish(),
  framesDir: z.string().nullish(),
  overlaysDir: z.string().nullish(),
})

export const fieldAnalysisSummarySchema = z.object({
  scheduledObservations: z.number().int(),
  usableObservations: z.number().int(),
  limitedObservations: z.number().int(),
  highPriorityCount: z.number().int(),
  mediumPriorityCount: z.number().int(),
  lowPriorityCount: z.number().int(),
  highestPriorityZoneId: z.string().nullish(),
  headline: z.string(),
  keyFindings: z.array(z.string()),
  limitations: z.array(z.string()),
})

export const fieldTriageReportSchema = z.object({
  schemaVersion: z.literal('1.0'),
  runId: z.string(),
  createdAt: z.string(),
  disclaimer: z.string(),
  source: videoSourceMetaSchema,
  analysis: analysisSummarySchema,
  field: fieldSummarySchema,
  inspectionZones: z.array(inspectionZoneSchema),
  frameQuality: z.array(frameQualitySchema),
  classCoverage: z.record(z.number()),
  limitations: z.array(z.string()),
  artifacts: artifactPathsSchema,
  georeferenced: z.literal(false),
  mapLabel: z.string(),
  summary: fieldAnalysisSummarySchema.nullish(),
})

export const createAnalysisRequestSchema = z.object({
  sampleFps: z.number().min(0.5).max(10).optional(),
  maxFrames: z.number().int().positive().optional(),
  skipDino: z.boolean().optional(),
  segmentationBackend: z.enum(['sam3', 'heuristic', 'mock']).optional(),
  debug: z.boolean().optional(),
})

export const visionHealthSchema = z.object({
  status: z.enum(['ok', 'degraded']),
  version: z.string(),
  device: z.string(),
  segmentationBackend: z.enum(['sam3', 'heuristic', 'mock']),
  dinoBackend: z.enum(['dinov3', 'heuristic', 'mock']),
  samAvailable: z.boolean(),
  dinoAvailable: z.boolean(),
  torchAvailable: z.boolean(),
  opencvAvailable: z.boolean(),
  ffmpegAvailable: z.boolean(),
})

export type CreateAnalysisRequestInput = z.infer<typeof createAnalysisRequestSchema>
export type FieldTriageReportInput = z.infer<typeof fieldTriageReportSchema>

// ---------------------------------------------------------------------------
// Live Drone mode
// ---------------------------------------------------------------------------

export const liveSessionStatusSchema = z.enum([
  'disconnected',
  'waiting',
  'connecting',
  'live',
  'stream_lost',
  'reconnecting',
  'ended',
])

export const telemetryAvailabilitySchema = z.enum(['exact', 'nearby', 'unavailable'])

export const liveSessionSchema = z.object({
  id: z.string(),
  createdAt: z.string(),
  endedAt: z.string().nullish(),
  status: liveSessionStatusSchema,
  streamPath: z.string(),
  sampleFps: z.number(),
  simulated: z.boolean(),
  whepUrl: z.string(),
})

export const droneTelemetrySchema = z.object({
  timestampMs: z.number(),
  latitude: z.number().nullable(),
  longitude: z.number().nullable(),
  altitudeM: z.number().nullable(),
  headingDeg: z.number().nullable(),
  gimbalPitchDeg: z.number().nullable(),
  source: z.enum(['dji-cloud-api', 'unavailable']),
})

export const telemetryAssociationSchema = z.object({
  availability: telemetryAvailabilitySchema,
  telemetry: droneTelemetrySchema.nullish(),
  toleranceMs: z.number(),
})

export const liveInspectionAreaEventSchema = inspectionZoneSchema.extend({
  liveFrameTimestampMs: z.number(),
  telemetryAssociation: telemetryAssociationSchema,
})

export const liveStreamEventSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('frame_analyzed'),
    sessionId: z.string(),
    frameTimestampMs: z.number(),
    zones: z.array(liveInspectionAreaEventSchema),
  }),
  z.object({
    type: z.literal('state_change'),
    sessionId: z.string(),
    status: liveSessionStatusSchema,
  }),
  z.object({
    type: z.literal('telemetry'),
    sessionId: z.string(),
    telemetry: droneTelemetrySchema,
  }),
  z.object({
    type: z.literal('error'),
    sessionId: z.string(),
    message: z.string(),
  }),
])

export const savedInspectionAreaSchema = z.object({
  id: z.string(),
  sessionId: z.string(),
  zone: inspectionZoneSchema,
  telemetryAssociation: telemetryAssociationSchema,
  snapshotUrl: z.string().optional(),
  savedAt: z.string(),
  note: z.string().optional(),
})

export type LiveStreamEventInput = z.infer<typeof liveStreamEventSchema>
