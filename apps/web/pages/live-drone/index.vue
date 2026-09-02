<script setup lang="ts">
import type { LiveInspectionAreaEvent } from '@cropmerge/types'
import LiveStreamPlayer from '~/components/LiveStreamPlayer.vue'

const {
  liveDroneEnabled,
  status,
  session,
  telemetry,
  zones,
  errorMessage,
  videoStream,
  simulated,
  startSession,
  stopSession,
  toggleSimulatedInput,
  saveForFollowUp,
} = useLiveSession()

const playerRef = ref<InstanceType<typeof LiveStreamPlayer> | null>(null)
const selectedZone = ref<LiveInspectionAreaEvent | null>(null)
const simulating = ref(false)

const STATUS_LABELS: Record<string, string> = {
  disconnected: 'Disconnected',
  waiting: 'Waiting for stream',
  connecting: 'Connecting…',
  live: 'Live',
  stream_lost: 'Stream interrupted — waiting to reconnect',
  reconnecting: 'Reconnecting…',
  ended: 'Ended',
}
const statusLabel = computed(() => STATUS_LABELS[status.value] ?? status.value)

const statusClass = computed(() => {
  if (status.value === 'live') return 'ok'
  if (status.value === 'stream_lost' || status.value === 'disconnected') return 'bad'
  return 'warn'
})

async function handleStartSimulated() {
  simulating.value = true
  await toggleSimulatedInput(true).catch((err) => {
    errorMessage.value = err instanceof Error ? err.message : 'Failed to start the simulated feed'
  })
  await startSession({ simulated: true })
}

async function handleStop() {
  await stopSession()
  if (simulating.value) {
    await toggleSimulatedInput(false).catch(() => {})
    simulating.value = false
  }
  selectedZone.value = null
}

function selectZone(zone: LiveInspectionAreaEvent) {
  selectedZone.value = zone
}

async function handleSave(note: string) {
  if (!selectedZone.value) return
  await saveForFollowUp(selectedZone.value, note)
}

watch(zones, (next) => {
  if (selectedZone.value && !next.some((z) => z.id === selectedZone.value!.id)) {
    selectedZone.value = null
  }
})
</script>

<template>
  <div class="stack-lg">
    <div>
      <h1 class="page-title">Live Drone</h1>
      <p class="page-lead">
        Connect DJI Pilot 2's live feed (or a simulated one) to see field-scan Inspection Areas as they're flown.
        <NuxtLink to="/live-drone/setup">Connect a real drone &rarr;</NuxtLink>
      </p>
    </div>

    <div v-if="!liveDroneEnabled" class="disclaimer-box">
      <span class="ico">i</span>
      <div>
        Live Drone mode is not configured for this deployment. It requires MediaMTX + MQTT running
        locally via <code>docker compose up --build</code> — see AGENTS.md. It is never available on the
        hosted (Vercel) deployment.
      </div>
    </div>

    <template v-else>
      <div class="row-between">
        <span class="chip" :class="statusClass">
          <span class="dot" />
          {{ statusLabel }}
        </span>
        <div class="row">
          <button
            v-if="status === 'disconnected' || status === 'ended'"
            type="button"
            class="btn btn-primary"
            @click="handleStartSimulated"
          >
            Start Simulated Live Input
          </button>
          <button v-else type="button" class="btn btn-ghost" @click="handleStop">
            Stop
          </button>
        </div>
      </div>

      <p v-if="errorMessage" class="field-error">{{ errorMessage }}</p>

      <div class="grid-media">
        <LiveStreamPlayer ref="playerRef" :video-stream="videoStream" :simulated="simulated">
          <InspectionOverlay
            v-if="playerRef?.videoEl && playerRef?.containerEl"
            :video-el="playerRef.videoEl"
            :container-el="playerRef.containerEl"
            :zones="zones"
            :selected-zone-id="selectedZone?.id"
            @select="selectZone"
          />
        </LiveStreamPlayer>

        <div class="stack">
          <TelemetryPanel :telemetry="telemetry" :simulated="simulated" />
          <div class="card">
            <div class="card-head">
              <span class="section-label">Field Scan</span>
            </div>
            <p class="muted" style="font-size: 0.85rem">
              {{ zones.length }} Inspection Area{{ zones.length === 1 ? '' : 's' }} currently tracked
            </p>
          </div>
        </div>
      </div>

      <LiveInspectionAreaDetail v-if="selectedZone" :zone="selectedZone" :on-save="handleSave" />

      <div v-if="session" class="muted" style="font-size: 0.8rem">
        Session <span class="mono">{{ session.id }}</span> ·
        <NuxtLink :to="`/live-drone/sessions/${session.id}`">view session &amp; export</NuxtLink>
      </div>
    </template>
  </div>
</template>
