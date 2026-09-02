<script setup lang="ts">
import type { DroneTelemetry } from '@cropmerge/types'

defineProps<{ telemetry: DroneTelemetry | null; simulated?: boolean }>()

function fmt(value: number | null | undefined, unit: string, digits = 1): string {
  return value == null ? '—' : `${value.toFixed(digits)}${unit}`
}
</script>

<template>
  <div class="card">
    <div class="card-head">
      <span class="section-label">Flight</span>
    </div>
    <div v-if="simulated" class="muted" style="font-size: 0.85rem">
      No telemetry — simulated live input has no DJI Cloud API connection.
    </div>
    <div v-else-if="!telemetry" class="muted" style="font-size: 0.85rem">
      No telemetry yet — waiting for the DJI Cloud API connection.
    </div>
    <div v-else class="stat-grid">
      <div class="stat">
        <span class="label">Altitude</span>
        <span class="value">{{ fmt(telemetry.altitudeM, 'm') }}</span>
      </div>
      <div class="stat">
        <span class="label">Heading</span>
        <span class="value">{{ fmt(telemetry.headingDeg, '°', 0) }}</span>
      </div>
      <div class="stat">
        <span class="label">GPS</span>
        <span class="value" style="font-size: 0.85rem">
          <template v-if="telemetry.latitude != null && telemetry.longitude != null">
            {{ telemetry.latitude.toFixed(5) }}, {{ telemetry.longitude.toFixed(5) }}
          </template>
          <template v-else>—</template>
        </span>
      </div>
      <div class="stat">
        <span class="label">Gimbal pitch</span>
        <span class="value">{{ fmt(telemetry.gimbalPitchDeg, '°', 0) }}</span>
      </div>
    </div>
  </div>
</template>
