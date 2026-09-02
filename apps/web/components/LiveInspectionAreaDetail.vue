<script setup lang="ts">
import type { LiveInspectionAreaEvent } from '@cropmerge/types'
import {
  formatInspectionReasons,
  inspectionIssueLabel,
  inspectionIssueTooltip,
  locationLabel,
  reviewPriorityLabel,
} from '~/composables/useInspectionLabels'

const props = defineProps<{ zone: LiveInspectionAreaEvent; onSave: (note: string) => Promise<unknown> }>()

const saving = ref(false)
const saved = ref(false)
const note = ref('')
const saveError = ref('')

async function handleSave() {
  saving.value = true
  saveError.value = ''
  try {
    await props.onSave(note.value)
    saved.value = true
  } catch (err) {
    saveError.value = err instanceof Error ? err.message : 'Failed to save this inspection area'
  } finally {
    saving.value = false
  }
}

watch(
  () => props.zone.id,
  () => {
    saved.value = false
    note.value = ''
    saveError.value = ''
  },
)

const availabilityLabel = computed(() => {
  const availability = props.zone.telemetryAssociation.availability
  if (availability === 'exact') return 'Exact telemetry match'
  if (availability === 'nearby') return 'Nearby telemetry match'
  return 'No telemetry available at this moment'
})
</script>

<template>
  <div class="card">
    <div class="card-head">
      <span class="section-label">Inspection Area</span>
      <span class="priority" :class="zone.reviewPriority">{{ reviewPriorityLabel(zone.reviewPriority) }}</span>
    </div>
    <p style="font-weight: 600; margin-bottom: 0.25rem" :title="inspectionIssueTooltip(zone)">
      {{ inspectionIssueLabel(zone) }}
    </p>
    <p class="muted" style="font-size: 0.85rem">{{ locationLabel(zone.relativeLocation) }} of frame</p>

    <ul v-if="formatInspectionReasons(zone).length" class="zone-reasons">
      <li v-for="(reason, i) in formatInspectionReasons(zone)" :key="i">{{ reason }}</li>
    </ul>

    <p class="muted" style="font-size: 0.8rem; margin-top: 0.5rem">{{ availabilityLabel }}</p>
    <p v-if="zone.telemetryAssociation.telemetry?.latitude != null" class="mono" style="font-size: 0.8rem">
      {{ zone.telemetryAssociation.telemetry.latitude.toFixed(5) }},
      {{ zone.telemetryAssociation.telemetry.longitude!.toFixed(5) }}
    </p>

    <div class="stack" style="margin-top: 0.75rem">
      <input
        v-model="note"
        type="text"
        placeholder="Optional note"
        class="text-input"
        :disabled="saved"
      >
      <button type="button" class="btn btn-primary" :disabled="saving || saved" @click="handleSave">
        {{ saved ? 'Saved for Follow-Up' : saving ? 'Saving…' : 'Save for Follow-Up' }}
      </button>
      <p v-if="saveError" class="field-error">{{ saveError }}</p>
    </div>
  </div>
</template>

<style scoped>
.zone-reasons {
  margin: 0.5rem 0 0;
  padding-left: 1rem;
  font-size: 0.85rem;
  color: var(--text-soft);
}
.text-input {
  width: 100%;
  padding: 0.55rem 0.7rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: var(--bg-elevated);
  color: var(--text);
  font-size: 0.88rem;
}
.text-input:focus-visible {
  outline: 2px solid var(--crop);
  outline-offset: 1px;
}
</style>
