<script setup lang="ts">
const { request } = useVisionApi()

type NetworkInfo = {
  candidateAddresses: string[]
  mqttPort: number
  rtmpPort: number
  rtspPort: number
  streamPath: string
  note: string
}

const info = ref<NetworkInfo | null>(null)
const error = ref('')

onMounted(async () => {
  try {
    info.value = await request<NetworkInfo>('/vision/live/network-info')
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load network info'
  }
})
</script>

<template>
  <div class="stack-lg">
    <h1 class="page-title">Connect DJI Mavic 3M</h1>
    <p class="page-lead">
      This flow follows DJI's documented Pilot 2 Open Platforms setup. CropMerge does not drive the
      pairing itself — Pilot 2 remains the flight system and owns this connection.
    </p>

    <p v-if="error" class="field-error">{{ error }}</p>

    <div class="card">
      <div class="card-head">
        <span class="section-label">This laptop</span>
      </div>
      <p v-if="!info" class="muted">Loading…</p>
      <template v-else>
        <p class="muted" style="font-size: 0.85rem; margin-bottom: 0.5rem">{{ info.note }}</p>
        <ul class="stack">
          <li v-for="addr in info.candidateAddresses" :key="addr" class="mono">{{ addr }}</li>
        </ul>
        <p class="muted" style="font-size: 0.8rem; margin-top: 0.75rem">
          RTMP/RTSP stream path: <span class="mono">{{ info.streamPath }}</span> · MQTT port
          <span class="mono">{{ info.mqttPort }}</span>
        </p>
      </template>
    </div>

    <div class="card">
      <div class="card-head">
        <span class="section-label">Setup steps</span>
      </div>
      <ol class="stack">
        <li>Put this laptop and the RC Pro Enterprise on networks that can reach each other.</li>
        <li>On the RC Pro Enterprise, open DJI Pilot 2.</li>
        <li>Open Cloud Services → Open Platforms.</li>
        <li>Enter one of the addresses above as the CropMerge connection URL.</li>
        <li>Connect.</li>
        <li>Verify telemetry appears on the <NuxtLink to="/live-drone">Live Drone</NuxtLink> page.</li>
        <li>Start the live stream from Pilot 2.</li>
      </ol>
      <p class="muted" style="font-size: 0.8rem">
        This connection is not shown as established until telemetry or a video stream is actually
        received — see <span class="mono">docs/DJI_M3M_FIELD_TEST.md</span> for the full hardware
        verification checklist.
      </p>
    </div>
  </div>
</template>
