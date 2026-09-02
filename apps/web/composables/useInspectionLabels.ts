import type { FieldTriageReport, InspectionZone, RelativeLocation } from '@cropmerge/types'

/** Internal enum / primaryType → farmer-facing issue label */
export const INSPECTION_ISSUE_LABELS: Record<string, string> = {
  stand_gap: 'Possible Crop Gap',
  row_discontinuity: 'Possible Crop Gap',
  STAND_GAP: 'Possible Crop Gap',
  ROW_DISCONTINUITY: 'Possible Crop Gap',

  sparse_canopy: 'Sparse Crop Coverage',
  SPARSE_CANOPY: 'Sparse Crop Coverage',

  exposed_soil: 'Exposed Soil',
  EXPOSED_SOIL: 'Exposed Soil',

  color_variation: 'Visual Difference',
  texture_variation: 'Visual Difference',
  general_visual_variation: 'Visual Difference',
  water_like_region: 'Visual Difference',
  COLOR_VARIATION: 'Visual Difference',
  TEXTURE_VARIATION: 'Visual Difference',
  GENERAL_VISUAL_VARIATION: 'Visual Difference',
  WATER_LIKE_REGION: 'Visual Difference',
}

export const INSPECTION_ISSUE_TOOLTIPS: Record<string, string> = {
  'Possible Crop Gap':
    'A section where crop coverage appears to break or decrease within otherwise continuous crop structure.',
  'Sparse Crop Coverage':
    'This area shows less visible crop coverage than nearby parts of the field.',
  'Visual Difference':
    'This area looks visually different from nearby crop based on color, texture, or image features.',
  'Exposed Soil': 'More visible bare soil than in surrounding crop areas.',
}

export const FIELD_COPY = {
  boundaryLabel: 'Analysis Field Boundary',
  boundaryTooltip:
    'Estimated field area used for this analysis. It is derived from the imagery and is not a property or surveyed boundary.',
  cropCoverageLabel: 'Estimated Crop Coverage',
  cropCoverageTooltip:
    'Estimated portion of the analyzed field that appears to contain visible crop vegetation in the current imagery.',
  inspectionAreasLabel: 'Inspection Areas',
  inspectionAreaPrefix: 'Inspection Area',
  emptyInspectionAreasTitle: 'No significant inspection areas found',
  emptyInspectionAreasBody:
    'The analyzed footage did not show persistent visual differences above the current review threshold.',
  limitedAnalysisTitle: 'Limited Analysis',
  limitedAnalysisBody:
    'Some field areas could not be analyzed confidently due to image quality, visibility, or crop structure.',
} as const

const BACKEND_SIGNAL_MAP: Record<string, string> = {
  'Possible stand gap / crop discontinuity': 'Possible Crop Gap',
  'Possible row discontinuity': 'Possible Crop Gap',
  'Possible fragmented or sparse canopy': 'Sparse Crop Coverage',
  'Increased exposed soil': 'Exposed Soil',
  'Unusual crop appearance (color)': 'Visual Difference',
  'Unusual surface texture': 'Visual Difference',
  'General visual variation': 'Visual Difference',
  'Visual variation': 'Visual Difference',
}

export function inspectionAreaTitle(index: number): string {
  return `${FIELD_COPY.inspectionAreaPrefix} ${index + 1}`
}

export function inspectionIssueLabel(zone: InspectionZone): string {
  const t = zone.primaryType
  if (t && INSPECTION_ISSUE_LABELS[t]) return INSPECTION_ISSUE_LABELS[t]
  const sig = zone.primarySignalLabel?.trim()
  if (sig && BACKEND_SIGNAL_MAP[sig]) return BACKEND_SIGNAL_MAP[sig]
  if (sig) {
    for (const [key, label] of Object.entries(BACKEND_SIGNAL_MAP)) {
      if (sig.toLowerCase().includes(key.toLowerCase())) return label
    }
  }
  const struct = zone.structuralAnomalyScore ?? 0
  const appear = zone.appearanceAnomalyScore ?? zone.anomalyScore ?? 0
  if (struct >= 0.35 && struct > appear) return 'Possible Crop Gap'
  if (struct >= 0.25) return 'Sparse Crop Coverage'
  if (appear >= 0.3) return 'Visual Difference'
  return 'Visual Difference'
}

export function inspectionIssueTooltip(zone: InspectionZone): string {
  return INSPECTION_ISSUE_TOOLTIPS[inspectionIssueLabel(zone)] ?? INSPECTION_ISSUE_TOOLTIPS['Visual Difference']
}

export function locationLabel(location: RelativeLocation | string): string {
  return String(location)
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

export function reviewPriorityLabel(priority: string): string {
  return priority.charAt(0).toUpperCase() + priority.slice(1)
}

export function formatObservations(zone: InspectionZone): string {
  if (zone.persistentObservations != null && zone.totalObservations != null) {
    return `Seen in ${zone.persistentObservations} of ${zone.totalObservations} usable observations`
  }
  if (zone.framesSeen) return `Seen in ${zone.framesSeen} frames`
  return ''
}

/** Plain-language reasons for farmers; keeps honesty, strips ML jargon */
export function formatInspectionReason(reason: string): string {
  let r = reason
  const replacements: [RegExp, string][] = [
    [/DINO visual features differ from field baseline \(distance [\d.]+\)/gi, 'Looks different from nearby crop in the image'],
    [/embedding distance [\d.]+/gi, 'looks different from nearby crop'],
    [/RGB color profile diverges from the field median \(Lab distance [\d.]+[^)]*\)/gi, 'Color differs from surrounding crop'],
    [/Greenness proxy \(ExG\) is off the field baseline by [\d.]+[^—]*/gi, 'Greenness differs from the surrounding field'],
    [/Surface texture \/ edge pattern differs from neighboring crop \([\d.]+[^)]*\)/gi, 'Texture differs from nearby areas'],
    [/Multivariate fingerprint is an outlier[^.]*\./gi, 'Several image cues differ together in this area.'],
    [/Canopy looks thinner than the field norm \(([^)]+)\)/gi, 'Crop coverage is lower here ($1)'],
    [/Visible vegetation occupancy is (\d+%) below nearby rows/gi, 'Crop coverage is about $1 lower than nearby rows'],
    [/Crop-row continuity breaks in this region/gi, 'Crop structure appears interrupted in this area'],
    [/Rows\/vegetation continue into and out of this region/gi, 'Crop structure continues before and after this area'],
    [/Additional exposed soil is visible/gi, 'More soil is visible here'],
    [/Pattern persists across (\d+) of (\d+) usable observations \(persistence [\d.]+%\)/gi, 'Seen in $1 of $2 usable observations'],
    [/Not a one-frame glitch — pattern held for ~[\d.]+% of sampled timestamps[^.]*\./gi, 'The pattern persists across multiple frames.'],
    [/No strong row discontinuity detected — primarily appearance-based/gi, 'Difference is mainly visual, not a clear crop gap'],
    [/visual anomaly/gi, 'visual difference'],
    [/anomaly/gi, 'difference'],
    [/structural/gi, 'crop structure'],
    [/appearance-only/gi, 'visual'],
    [/NDVI/gi, 'vegetation index'],
  ]
  for (const [pat, rep] of replacements) {
    r = r.replace(pat, rep)
  }
  return r
}

export function formatInspectionReasons(zone: InspectionZone): string[] {
  return (zone.reasons ?? []).map(formatInspectionReason)
}

export function showLimitedAnalysis(report: FieldTriageReport): boolean {
  const lowRows = report.field.rowVisibility === 'LOW'
  const lowUsable =
    report.analysis.framesUsable / Math.max(report.analysis.framesSampled, 1) < 0.5
  const hasLimitation = (report.limitations ?? []).some(
    (l) =>
      /quality|visibility|registration|fallback|row visibility/i.test(l),
  )
  return lowRows || lowUsable || hasLimitation
}

export function estimatedCropCoverage(report: FieldTriageReport): number {
  return report.field.cropCoverage?.estimatedFraction ?? report.field.meanCropCoverage
}
