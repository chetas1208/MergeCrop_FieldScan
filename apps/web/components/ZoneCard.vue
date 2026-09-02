<script setup lang="ts">
import type { InspectionZone } from '@cropmerge/types'
import {
  formatInspectionReasons,
  formatObservations,
  inspectionAreaTitle,
  inspectionIssueLabel,
  inspectionIssueTooltip,
  locationLabel,
  reviewPriorityLabel,
} from '~/composables/useInspectionLabels'

defineProps<{
  zone: InspectionZone
  areaIndex: number
  selected?: boolean
}>()
defineEmits<{ select: [] }>()
</script>

<template>
  <button type="button" class="zone" :class="{ selected }" @click="$emit('select')">
    <div class="zone-head">
      <span class="zone-id">{{ inspectionAreaTitle(areaIndex) }}</span>
      <span class="priority" :class="zone.reviewPriority">
        {{ reviewPriorityLabel(zone.reviewPriority) }}
      </span>
    </div>
    <p class="zone-loc" style="font-weight: 600; margin-bottom: 0.25rem" :title="inspectionIssueTooltip(zone)">
      {{ inspectionIssueLabel(zone) }}
    </p>
    <p class="zone-loc">{{ locationLabel(zone.relativeLocation) }} field</p>
    <p class="zone-loc muted" style="font-size: 0.78rem; margin-top: 0.35rem">
      Review priority: <strong>{{ reviewPriorityLabel(zone.reviewPriority).toUpperCase() }}</strong>
    </p>
    <p v-if="zone.llmSummary" class="zone-summary">
      <span class="zone-summary-tag">Field note</span>{{ zone.llmSummary }}
    </p>
    <ul v-else-if="formatInspectionReasons(zone).length" class="zone-reasons-preview">
      <li v-for="(r, i) in formatInspectionReasons(zone).slice(0, 2)" :key="i">{{ r }}</li>
    </ul>
    <p v-if="formatObservations(zone)" class="muted" style="font-size: 0.75rem; margin: 0.35rem 0 0">
      {{ formatObservations(zone) }}
    </p>
    <details class="zone-tech" @click.stop>
      <summary>View technical details</summary>
      <dl class="tech-dl">
        <dt>Structural score</dt>
        <dd>{{ (zone.structuralAnomalyScore ?? 0).toFixed(2) }}</dd>
        <dt>Appearance score</dt>
        <dd>{{ (zone.appearanceAnomalyScore ?? zone.anomalyScore).toFixed(2) }}</dd>
        <dt>Persistence</dt>
        <dd>
          <template v-if="zone.persistentObservations != null && zone.totalObservations != null">
            {{ zone.persistentObservations }} / {{ zone.totalObservations }} observations
          </template>
          <template v-else>{{ zone.persistenceScore.toFixed(2) }}</template>
        </dd>
        <template v-if="zone.evidence?.cropCoverageDelta != null">
          <dt>Crop coverage delta</dt>
          <dd>{{ Math.round(zone.evidence.cropCoverageDelta * 100) }}%</dd>
        </template>
        <template v-if="zone.evidence?.soilExposureDelta != null">
          <dt>Soil exposure delta</dt>
          <dd>+{{ Math.round(zone.evidence.soilExposureDelta * 100) }}%</dd>
        </template>
        <dt>Internal ID</dt>
        <dd class="mono">{{ zone.id }}</dd>
      </dl>
    </details>
  </button>
</template>

<style scoped>
.zone-summary {
  margin: 0.5rem 0 0;
  font-size: 0.85rem;
  color: var(--text-soft);
  line-height: 1.5;
}
.zone-summary-tag {
  display: inline-block;
  margin-right: 0.45rem;
  padding: 0.05rem 0.4rem;
  border-radius: 999px;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--sky);
  border: 1px solid color-mix(in srgb, var(--sky) 40%, transparent);
  vertical-align: middle;
}
.zone-reasons-preview {
  margin: 0.5rem 0 0;
  padding-left: 1rem;
  font-size: 0.78rem;
  color: var(--muted, #8a8a8a);
}
.zone-tech {
  margin-top: 0.65rem;
  font-size: 0.75rem;
  text-align: left;
}
.zone-tech summary {
  cursor: pointer;
  color: var(--muted, #8a8a8a);
}
.tech-dl {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.2rem 0.65rem;
  margin: 0.45rem 0 0;
}
.tech-dl dt {
  color: var(--muted, #8a8a8a);
}
.tech-dl dd {
  margin: 0;
}
</style>
