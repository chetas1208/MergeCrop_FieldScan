<script setup lang="ts">
const route = useRoute()
const sessionId = String(route.params.id)
const { request } = useVisionApi()

type SavedArea = {
  id: string
  savedAt: string
  note?: string | null
  telemetryAssociation: { availability: string }
  zone: { primaryType?: string; reviewPriority: string }
}

const savedAreas = ref<SavedArea[]>([])
const artifactUrls = ref<Record<string, string>>({})
const loading = ref(true)
const error = ref('')

async function load() {
  loading.value = true
  error.value = ''
  try {
    const areasResponse = await request<{ areas: SavedArea[] }>(`/vision/live/sessions/${sessionId}/saved-areas`)
    savedAreas.value = areasResponse.areas
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load saved areas'
  } finally {
    loading.value = false
  }
}

async function runExport() {
  try {
    const result = await request<{ artifactUrls: Record<string, string> }>(`/vision/live/sessions/${sessionId}/export`)
    artifactUrls.value = result.artifactUrls
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Export failed'
  }
}

onMounted(load)
</script>

<template>
  <div class="stack-lg">
    <h1 class="page-title">Live Session</h1>
    <p class="page-lead mono">{{ sessionId }}</p>

    <p v-if="error" class="field-error">{{ error }}</p>

    <div class="card">
      <div class="card-head">
        <span class="section-label">Saved Inspection Areas</span>
      </div>
      <p v-if="loading" class="muted">Loading…</p>
      <p v-else-if="!savedAreas.length" class="muted">No areas saved for follow-up yet in this session.</p>
      <ul v-else class="stack">
        <li v-for="area in savedAreas" :key="area.id" class="card-flush">
          <strong>{{ area.zone.primaryType || 'Inspection area' }}</strong>
          <span class="muted"> · {{ area.telemetryAssociation.availability }} telemetry</span>
          <span v-if="area.note" class="muted"> · {{ area.note }}</span>
        </li>
      </ul>
    </div>

    <div class="card">
      <div class="card-head">
        <span class="section-label">Export</span>
      </div>
      <button type="button" class="btn btn-primary" @click="runExport">Generate Export</button>
      <ul v-if="Object.keys(artifactUrls).length" class="stack" style="margin-top: 0.75rem">
        <li v-for="(url, name) in artifactUrls" :key="name">
          <a :href="url" target="_blank" rel="noopener">{{ name }}</a>
        </li>
      </ul>
    </div>
  </div>
</template>
