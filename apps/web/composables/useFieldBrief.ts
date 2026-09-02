import type { FieldTriageReport, InspectionZone } from '@cropmerge/types'
import type { Ref } from 'vue'
import {
  FIELD_COPY,
  estimatedCropCoverage,
  inspectionIssueLabel,
  locationLabel,
} from '~/composables/useInspectionLabels'

export function combinedReviewScore(zone: InspectionZone): number {
  if (zone.reviewScore != null) return zone.reviewScore
  const base = Math.max(
    zone.structuralAnomalyScore ?? 0,
    zone.appearanceAnomalyScore ?? zone.anomalyScore,
  )
  return 0.65 * base + 0.35 * zone.persistenceScore
}

export function zoneHeadline(zone: InspectionZone): string {
  return locationLabel(zone.relativeLocation)
}

export function useFieldBrief(report: Ref<FieldTriageReport | null | undefined>, zones: Ref<InspectionZone[]>) {
  const brief = computed(() => {
    const r = report.value
    const z = zones.value
    if (!r) return ''

    const duration = r.source.durationSec?.toFixed?.(0) ?? '?'
    const frames = r.analysis.framesSampled
    const crop = `${Math.round(estimatedCropCoverage(r) * 100)}%`

    if (!r.field.detected) {
      return `We reviewed ${frames} frames from a ${duration}s clip but could not estimate a clear ${FIELD_COPY.boundaryLabel.toLowerCase()}. The view may include mostly bare ground, trees, or an angle that hides the field. Try a higher pass or a tighter crop.`
    }

    if (!z.length) {
      return `${FIELD_COPY.emptyInspectionAreasTitle}. ${FIELD_COPY.emptyInspectionAreasBody} Estimated crop coverage in the analyzed area: ${crop}.`
    }

    const high = z.filter((x) => x.reviewPriority === 'high').length
    const top = [...z].sort((a, b) => combinedReviewScore(b) - combinedReviewScore(a))[0]
    const topLoc = locationLabel(top.relativeLocation)
    const topIssue = inspectionIssueLabel(top)

    let opener = `After ${frames} frames (${duration}s of flight), ${FIELD_COPY.cropCoverageLabel.toLowerCase()} is about ${crop}. `
    opener += `${z.length} ${FIELD_COPY.inspectionAreasLabel.toLowerCase()} may be worth a closer look`
    if (high) opener += ` — ${high} marked high priority`
    opener += `. The most notable is in the ${topLoc} area (${topIssue}). `
    opener += `These are image-based differences for review, not diagnoses of crop health, yield, or planting failure.`
    return opener
  })

  const loadingLines = [
    'Reading your flight footage…',
    'Estimating the field area in view…',
    'Checking crop coverage across sampled frames…',
    'Looking for areas that differ from the surrounding field…',
    'Preparing overlays and summary…',
  ]

  return { brief, loadingLines, zoneHeadline, combinedReviewScore }
}
