<script setup lang="ts">
import type { AnalysisJob, FieldTriageReport, InspectionZone, VisionHealth } from '@cropmerge/types'
import {
  FIELD_COPY,
  estimatedCropCoverage,
  formatInspectionReasons,
  formatObservations,
  inspectionAreaTitle,
  inspectionIssueLabel,
  inspectionIssueTooltip,
  locationLabel,
  reviewPriorityLabel,
  showLimitedAnalysis,
} from '~/composables/useInspectionLabels'

type Health = {
  ok?: boolean
  error?: string
  db?: { ok?: boolean; mode?: string; path?: string }
  vision?: VisionHealth
}

type RecentJob = {
  id: string
  status: string
  createdAt: string
  filename: string
  zoneCount?: number
  fieldDetected?: boolean | null
}

const file = ref<File | null>(null)
const loading = ref(false)
const error = ref('')
const job = ref<AnalysisJob | null>(null)
const health = ref<Health | null>(null)
const recent = ref<RecentJob[]>([])
const dragover = ref(false)
const mediaTab = ref<'video' | 'heatmap' | 'montage' | 'frames'>('video')
const selectedZoneId = ref<string | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)
const frameSectionEl = ref<HTMLElement | null>(null)
const mediaFrameEl = ref<HTMLElement | null>(null)
const annotatedVideoEl = ref<HTMLVideoElement | null>(null)
const uploadProgress = ref<number | null>(null)
const activeUploadId = ref<string | null>(null)
const abortController = ref<AbortController | null>(null)
const { request } = useVisionApi()
const {
  isDemo,
  demoCases,
  loadManifest,
  fetchDemoJob,
  simulateDemoRun,
  setMode,
  manifestError,
} = useDemoMode()
const selectedDemoId = ref<string | null>(null)
const demoResult = ref(false)

useViewportPlaybackPause(mediaFrameEl, {
  threshold: 0.3,
  isPlaying: () => Boolean(annotatedVideoEl.value && !annotatedVideoEl.value.paused),
  onPause: () => annotatedVideoEl.value?.pause(),
})

function scrollToFrames() {
  mediaTab.value = 'frames'
  nextTick(() => {
    frameSectionEl.value?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  })
}

const PIPELINE = [
  'Decode',
  'Quality',
  'Segment',
  'Compare',
  'Track',
  'Summarize',
]

onMounted(async () => {
  await Promise.all([refreshHealth(), refreshRecent(), loadManifest()])
})

function demoCaseForSample(sampleId: string) {
  if (sampleId === 'vid') return 'drone-field-video'
  if (sampleId === 'img') return 'soybean-field-image'
  return null
}

function selectDemoSample(sampleId: string) {
  selectedDemoId.value = demoCaseForSample(sampleId)
  error.value = ''
}

async function refreshHealth() {
  try {
    const vision = await request<VisionHealth>('/vision/health', {}, false)
    health.value = { ok: vision.status === 'ok' && vision.gpuAvailable !== false, vision }
  } catch {
    health.value = { ok: false, error: 'Health check failed' }
  }
}

async function refreshRecent() {
  try {
    const response = await request<{ analyses: AnalysisJob[] }>('/vision/analyses')
    recent.value = response.analyses.map((analysis) => ({
      id: analysis.id,
      status: analysis.status,
      createdAt: analysis.createdAt,
      filename: analysis.filename,
      zoneCount: analysis.report?.inspectionZones.length,
      fieldDetected: analysis.report?.field.detected,
    }))
  } catch {
    recent.value = []
  }
}

const report = computed(() => job.value?.report as FieldTriageReport | null | undefined)

const zones = computed(() => report.value?.inspectionZones ?? [])

const { brief: fieldBrief, loadingLines } = useFieldBrief(report, zones)

const loadingCaption = computed(() => {
  if (uploadProgress.value != null && uploadProgress.value < 1) {
    return `Uploading directly to the vision server · ${Math.round(uploadProgress.value * 100)}%`
  }
  if (job.value?.message) return job.value.message
  const stage = job.value?.stage || ''
  const idx = ['queued', 'preparing', 'processing', 'rendering', 'completed'].indexOf(stage)
  if (idx >= 0) return loadingLines[idx] ?? loadingLines[2]
  return loadingLines[2]
})

const selectedZone = computed<InspectionZone | null>(() => {
  if (!zones.value.length) return null
  const hit = zones.value.find((z) => z.id === selectedZoneId.value)
  return hit || zones.value[0]
})

const selectedAreaIndex = computed(() => {
  const z = selectedZone.value
  if (!z) return 0
  const idx = zones.value.findIndex((item) => item.id === z.id)
  return idx >= 0 ? idx : 0
})

watch(
  zones,
  (z) => {
    if (z.length && !z.some((x) => x.id === selectedZoneId.value)) {
      selectedZoneId.value = z[0].id
    }
  },
  { immediate: true },
)

const highestPriority = computed(() => {
  if (zones.value.some((z) => z.reviewPriority === 'high')) return 'high'
  if (zones.value.some((z) => z.reviewPriority === 'medium')) return 'medium'
  if (zones.value.length) return 'low'
  return null
})

const visionOnline = computed(() => health.value?.vision?.status === 'ok')
const canAnalyze = computed(() => {
  if (isDemo.value) return Boolean(selectedDemoId.value)
  return Boolean(file.value) && visionOnline.value
})
const gpuBusy = computed(() => Boolean(health.value?.vision?.gpuBusy))
const gpuStatusMessage = computed(() => {
  const vision = health.value?.vision
  if (vision?.gpuMessage) return vision.gpuMessage
  if (gpuBusy.value) {
    const depth = vision?.queueDepth ?? 0
    return depth > 0
      ? `GPU is busy — ${depth} analysis job(s) waiting in queue.`
      : 'GPU is busy running another analysis.'
  }
  return ''
})

function pct(n: number | undefined) {
  if (n == null || Number.isNaN(n)) return '—'
  return `${Math.round(n * 100)}%`
}

function artifactUrl(name: string) {
  return job.value?.artifactUrls?.[name] || ''
}

function formatBytes(n: number) {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

function formatWhen(iso: string) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return iso
  }
}

const VIDEO_EXTS = ['.mp4', '.mov', '.m4v']
const IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
const ALLOWED_EXTS = [...VIDEO_EXTS, ...IMAGE_EXTS]

function extOf(name: string) {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i).toLowerCase() : ''
}

function isAllowedFile(f: File) {
  return ALLOWED_EXTS.includes(extOf(f.name)) || f.type.startsWith('video/') || f.type.startsWith('image/')
}

const isImageUpload = computed(() => {
  if (!file.value) return false
  const ext = extOf(file.value.name)
  return IMAGE_EXTS.includes(ext) || file.value.type.startsWith('image/')
})

function setFile(f: File | null) {
  if (f && !isAllowedFile(f)) {
    error.value = 'Use video (.mp4 .mov .m4v) or image (.jpg .png .webp .bmp).'
    return
  }
  file.value = f
  error.value = ''
}

function onFileInput(e: Event) {
  const input = e.target as HTMLInputElement
  setFile(input.files?.[0] ?? null)
}

function onDrop(e: DragEvent) {
  dragover.value = false
  const f = e.dataTransfer?.files?.[0]
  if (f) setFile(f)
}

function openFilePicker() {
  fileInput.value?.click()
}

function onDropZoneClick(e: MouseEvent) {
  const t = e.target as HTMLElement | null
  if (t?.closest('button, a, input, label')) return
  openFilePicker()
}

const SAMPLE_INPUTS = [
  {
    id: 'vid',
    label: 'Real drone field video',
    detail: 'Estonia · 10s · 1280×720',
    url: '/samples/real_field_drone.mp4',
    filename: 'real_field_drone.mp4',
    mime: 'video/mp4',
  },
  {
    id: 'img',
    label: 'Real soybean field image',
    detail: 'South Dakota · RGB still',
    url: '/samples/real_soybean_field.jpg',
    filename: 'real_soybean_field.jpg',
    mime: 'image/jpeg',
  },
] as const

const loadingSample = ref<string | null>(null)

async function useSample(sample: (typeof SAMPLE_INPUTS)[number]) {
  if (isDemo.value) {
    const demoId = demoCaseForSample(sample.id)
    if (demoId) {
      selectedDemoId.value = demoId
      error.value = ''
      return
    }
  }
  loadingSample.value = sample.id
  error.value = ''
  try {
    const res = await fetch(sample.url)
    if (!res.ok) throw new Error(`Sample download failed (${res.status})`)
    const blob = await res.blob()
    setFile(new File([blob], sample.filename, { type: sample.mime }))
    selectedDemoId.value = null
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loadingSample.value = null
  }
}

async function runDemoAnalysis(demoId: string) {
  loading.value = true
  error.value = ''
  job.value = null
  demoResult.value = true
  uploadProgress.value = null
  activeUploadId.value = null
  abortController.value = new AbortController()
  const demoCase = demoCases.value.find((item) => item.id === demoId)
  mediaTab.value = demoCase?.kind === 'image' ? 'heatmap' : 'video'
  try {
    await simulateDemoRun((stage, progress, message) => {
      job.value = {
        id: demoCase?.runId || demoId,
        status: stage === 'completed' ? 'completed' : 'processing',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        filename: demoCase?.inputFilename || demoId,
        progress,
        stage,
        message,
        report: job.value?.report ?? null,
        artifactUrls: job.value?.artifactUrls,
      }
    }, abortController.value.signal)
    job.value = await fetchDemoJob(demoId)
    selectedDemoId.value = demoId
  } catch (e: unknown) {
    if ((e as { name?: string }).name !== 'AbortError') {
      error.value = e instanceof Error ? e.message : String(e)
    }
    demoResult.value = false
  } finally {
    loading.value = false
    abortController.value = null
  }
}

async function openDemoCase(demoId: string) {
  selectedDemoId.value = demoId
  await runDemoAnalysis(demoId)
}

type UploadResponse = {
  uploadId: string
  status: string
  size: number
}

type UploadInitResponse = {
  uploadId: string
  chunkSize: number
  totalChunks: number
}

async function uploadFile(input: File, signal: AbortSignal): Promise<string> {
  const configuredThreshold = Number(useRuntimeConfig().public.visionSmallUploadThresholdBytes)
  if (input.size <= configuredThreshold) {
    uploadProgress.value = 0.05
    const body = new FormData()
    body.append('file', input)
    const uploaded = await request<UploadResponse>('/vision/uploads', {
      method: 'POST',
      body,
      signal,
    })
    uploadProgress.value = 1
    return uploaded.uploadId
  }

  const initialized = await request<UploadInitResponse>('/vision/uploads/init', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ filename: input.name, size: input.size, contentType: input.type || undefined }),
    signal,
  })
  activeUploadId.value = initialized.uploadId

  for (let index = 0; index < initialized.totalChunks; index += 1) {
    const start = index * initialized.chunkSize
    const chunk = input.slice(start, Math.min(start + initialized.chunkSize, input.size))
    await request<void>(`/vision/uploads/${initialized.uploadId}/chunks/${index}`, {
      method: 'PUT',
      headers: { 'content-type': 'application/octet-stream' },
      body: chunk,
      signal,
    })
    uploadProgress.value = (index + 1) / initialized.totalChunks
  }

  const completed = await request<UploadResponse>(`/vision/uploads/${initialized.uploadId}/complete`, {
    method: 'POST',
    signal,
  })
  return completed.uploadId
}

async function waitForAnalysis(id: string, signal: AbortSignal) {
  while (!signal.aborted) {
    const current = await request<AnalysisJob>(`/vision/analyses/${id}`, { signal })
    job.value = current
    if (current.status === 'completed') return current
    if (current.status === 'failed' || current.status === 'cancelled') {
      throw new Error(current.error || current.message || 'Analysis did not complete')
    }
    if (current.status === 'queued' && current.message) {
      error.value = ''
    }
    await new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(resolve, 1500)
      signal.addEventListener(
        'abort',
        () => {
          window.clearTimeout(timer)
          reject(new DOMException('Request aborted', 'AbortError'))
        },
        { once: true },
      )
    })
  }
  throw new DOMException('Request aborted', 'AbortError')
}

async function analyze() {
  if (isDemo.value) {
    if (!selectedDemoId.value) {
      error.value = 'Choose a demo flight below.'
      return
    }
    await runDemoAnalysis(selectedDemoId.value)
    return
  }
  if (!file.value) {
    error.value = 'Choose a drone video or field image first.'
    return
  }
  if (!visionOnline.value) {
    error.value = health.value?.vision?.gpuMessage
      || 'Vision engine is offline. Switch to Demo mode or start the server on port 8001.'
    return
  }
  if (gpuBusy.value && (health.value?.vision?.queueDepth ?? 0) >= (health.value?.vision?.maxQueueDepth ?? 3)) {
    error.value = gpuStatusMessage.value || 'GPU queue is full. Try again in a few minutes.'
    return
  }
  loading.value = true
  error.value = ''
  job.value = null
  demoResult.value = false
  uploadProgress.value = 0
  activeUploadId.value = null
  abortController.value = new AbortController()
  mediaTab.value = isImageUpload.value ? 'heatmap' : 'video'
  try {
    const uploadId = await uploadFile(file.value, abortController.value.signal)
    activeUploadId.value = null
    job.value = await request<AnalysisJob>('/vision/analyses', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        uploadId,
        sampleFps: 2,
        // Images are repeated across a few synthetic timestamps so temporal
        // persistence (min ~3 frames) can still form inspection zones.
        maxFrames: isImageUpload.value ? 3 : 24,
        skipDino: false,
        segmentationBackend: health.value?.vision?.segmentationBackend || 'heuristic',
        dinoBackend: health.value?.vision?.dinoBackend || 'heuristic',
      }),
      signal: abortController.value.signal,
    })
    await waitForAnalysis(job.value.id, abortController.value.signal)
    await refreshRecent()
  } catch (e: unknown) {
    if ((e as { name?: string }).name !== 'AbortError') {
      error.value = e instanceof Error ? e.message : String(e)
    }
  } finally {
    loading.value = false
    uploadProgress.value = null
    activeUploadId.value = null
    abortController.value = null
  }
}

async function openRecent(id: string) {
  error.value = ''
  loading.value = true
  demoResult.value = false
  try {
    if (isDemo.value) {
      const match = demoCases.value.find((item) => item.runId === id)
      if (match) {
        await runDemoAnalysis(match.id)
        return
      }
    }
    job.value = await request<AnalysisJob>(`/vision/analyses/${id}`)
    mediaTab.value = 'video'
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

async function cancelUpload() {
  abortController.value?.abort()
  const uploadId = activeUploadId.value
  if (uploadId) {
    try {
      await request<void>(`/vision/uploads/${uploadId}`, { method: 'DELETE' })
    } catch {
      // The canceled browser request may have already removed the partial upload.
    }
  }
}

function reset() {
  void cancelUpload()
  job.value = null
  file.value = null
  selectedZoneId.value = null
  selectedDemoId.value = null
  demoResult.value = false
  error.value = ''
  if (fileInput.value) fileInput.value.value = ''
}

const demoRecent = computed(() =>
  demoCases.value.map((item) => ({
    id: item.runId,
    demoId: item.id,
    status: 'completed',
    createdAt: 'Demo',
    filename: item.inputFilename,
    subtitle: item.subtitle,
    zoneCount: item.zoneCount,
    fieldDetected: true,
  })),
)

const mediaSrc = computed(() => {
  if (mediaTab.value === 'heatmap') return artifactUrl('heatmap.png')
  if (mediaTab.value === 'montage') return artifactUrl('segmentation_montage.jpg')
  if (mediaTab.value === 'frames') return ''
  return artifactUrl('annotated_video.mp4')
})
</script>

<template>
  <div class="stack stack-lg">
    <!-- Empty / upload state -->
    <template v-if="!report && !loading">
      <div class="row row-between">
        <div>
          <p class="section-label" style="margin-bottom: 0.35rem">Flight review</p>
          <h1 class="page-title">Review your field from drone footage</h1>
          <p class="page-lead">
            Upload RGB field video or images. CropMerge estimates crop coverage, outlines the analyzed
            field area, and highlights inspection areas that may deserve a closer look — not a crop-health
            diagnosis.
          </p>
        </div>
      </div>

      <div class="disclaimer-box">
        <div class="ico" aria-hidden="true">!</div>
        <div>
          Flagged areas are <strong>differences worth reviewing</strong> in the imagery — not disease,
          nutrient status, irrigation failure, plant count, or crop health.
        </div>
      </div>

      <div v-if="isDemo" class="demo-banner">
        <div class="ico" aria-hidden="true">D</div>
        <div>
          <strong>Demo mode</strong> — pre-computed analyses stored in this repo on GitHub.
          No GPU or vision server required. Pick a flight below and click
          <strong>View demo analysis</strong>.
          <span v-if="manifestError" class="muted" style="display: block; margin-top: 0.35rem">
            {{ manifestError }}
          </span>
        </div>
      </div>

      <div v-else-if="!visionOnline" class="demo-banner">
        <div class="ico" aria-hidden="true">!</div>
        <div>
          Vision engine is offline. Switch to <strong>Demo</strong> to explore stored field reviews,
          or start the server on port 8001 for live analysis.
          <button type="button" class="btn btn-ghost btn-sm" style="margin-top: 0.55rem" @click="setMode('demo')">
            Switch to Demo
          </button>
        </div>
      </div>

      <div class="grid-hero">
        <div
          class="upload-drop"
          :class="{ dragover, 'has-file': !!file }"
          role="button"
          tabindex="0"
          aria-label="Upload field video or image"
          @click="onDropZoneClick"
          @keydown.enter.prevent="openFilePicker"
          @keydown.space.prevent="openFilePicker"
          @dragenter.prevent="dragover = true"
          @dragover.prevent="dragover = true"
          @dragleave.prevent="dragover = false"
          @drop.prevent="onDrop"
        >
          <input
            ref="fileInput"
            type="file"
            class="sr-only"
            accept=".mp4,.mov,.m4v,.jpg,.jpeg,.png,.webp,.bmp,video/*,image/*"
            @change="onFileInput"
            @click.stop
          />
          <div class="upload-icon" aria-hidden="true">↑</div>
          <h2 class="upload-title">Drop field video or image here</h2>
          <p class="upload-meta">
            Video .mp4 · .mov · .m4v · Image .jpg · .png · .webp · 2 FPS sample
          </p>

          <div v-if="file && !isDemo" class="file-chip" @click.stop>
            <span>{{ file.name }}</span>
            <span class="muted">{{ formatBytes(file.size) }}</span>
            <span v-if="isImageUpload" class="muted">image</span>
          </div>
          <div v-else-if="isDemo && selectedDemoId" class="file-chip" @click.stop>
            <span>{{ demoCases.find((d) => d.id === selectedDemoId)?.title }}</span>
            <span class="muted">stored demo</span>
          </div>

          <div class="hero-actions" @click.stop>
            <button
              type="button"
              class="btn btn-primary"
              :disabled="!canAnalyze || loading"
              @click="analyze"
            >
              {{ isDemo ? 'View demo analysis' : 'Analyze field' }}
            </button>
            <button v-if="!isDemo" type="button" class="btn btn-ghost" @click="openFilePicker">
              Browse files
            </button>
          </div>

          <div class="sample-row" @click.stop>
            <span class="muted" style="font-size: 0.78rem">
              {{ isDemo ? 'Demo flights:' : 'Try a real sample:' }}
            </span>
            <button
              v-for="s in SAMPLE_INPUTS"
              :key="s.id"
              type="button"
              class="btn btn-ghost btn-sm"
              :class="{ 'btn-primary': isDemo && selectedDemoId === demoCaseForSample(s.id) }"
              :disabled="!!loadingSample || loading"
              @click="isDemo ? selectDemoSample(s.id) : useSample(s)"
            >
              {{ loadingSample === s.id ? 'Loading…' : s.label }}
            </button>
            <a
              class="muted mono"
              :href="isDemo ? '/demo/SOURCES.md' : '/samples/SOURCES.md'"
              target="_blank"
              rel="noopener"
              style="font-size: 0.72rem"
            >
              sources
            </a>
          </div>

          <div class="pipeline">
            <span v-for="(step, i) in PIPELINE" :key="step" class="pipe-step">
              <b>{{ String(i + 1).padStart(2, '0') }}</b>
              {{ step }}
            </span>
          </div>
        </div>

        <div class="stack">
          <div class="card">
            <h2 class="section-label">System</h2>
            <div class="stack" style="gap: 0.55rem">
              <div class="row row-between">
                <span class="muted">Vision engine</span>
                <span class="chip" :class="visionOnline ? 'ok' : 'bad'">
                  <span class="dot" />
                  {{ visionOnline ? 'Ready' : 'Offline' }}
                </span>
              </div>
              <div class="row row-between">
                <span class="muted">GPU queue</span>
                <span class="chip" :class="gpuBusy ? 'warn' : 'ok'">
                  <span class="dot" />
                  {{
                    gpuBusy
                      ? `${health?.vision?.queueDepth ?? 0} waiting`
                      : 'Available'
                  }}
                </span>
              </div>
              <details>
                <summary class="muted" style="cursor: pointer; font-size: 0.82rem">Technical system info</summary>
                <div class="stack" style="gap: 0.45rem; margin-top: 0.5rem">
                  <div class="row row-between">
                    <span class="muted">Segmentation</span>
                    <span class="mono">{{ health?.vision?.segmentationBackend || '—' }}</span>
                  </div>
                  <div class="row row-between">
                    <span class="muted">Feature model</span>
                    <span class="mono">{{ health?.vision?.dinoBackend || '—' }}</span>
                  </div>
                  <div class="row row-between">
                    <span class="muted">Device</span>
                    <span class="mono">{{ health?.vision?.device || '—' }}</span>
                  </div>
                </div>
              </details>
              <div class="row row-between">
                <span class="muted">Database</span>
                <span class="mono">{{ health?.db?.mode || 'sqlite' }}</span>
              </div>
            </div>
            <p v-if="gpuStatusMessage" class="warn-banner" style="margin-top: 1rem; margin-bottom: 0">
              {{ gpuStatusMessage }}
            </p>
            <p v-if="!visionOnline && !isDemo" class="error-banner" style="margin-top: 1rem; margin-bottom: 0">
              Start vision:
              <code class="mono">cd apps/vision && uvicorn api.main:app --port 8001</code>
            </p>
            <p v-else-if="isDemo" class="muted" style="margin-top: 1rem; margin-bottom: 0; font-size: 0.85rem">
              Demo uses static artifacts from <code class="mono">/demo/</code> on GitHub — no live GPU.
            </p>
          </div>

          <div class="card">
            <div class="card-head">
              <h2 class="section-label" style="margin: 0">{{ isDemo ? 'Demo flights' : 'Recent analyses' }}</h2>
              <button v-if="!isDemo" type="button" class="btn btn-ghost btn-sm" @click="refreshRecent">
                Refresh
              </button>
            </div>
            <p v-if="isDemo && !demoCases.length" class="muted" style="margin: 0">
              Demo manifest not loaded.
            </p>
            <p v-else-if="!isDemo && !recent.length" class="muted" style="margin: 0">
              No saved jobs yet. Run an analysis to populate the local database.
            </p>
            <div v-else-if="isDemo" class="recent-list">
              <button
                v-for="r in demoRecent.slice(0, 6)"
                :key="r.id"
                type="button"
                class="recent-item"
                @click="openDemoCase(r.demoId)"
              >
                <div style="min-width: 0">
                  <div class="name">{{ r.filename }}</div>
                  <div class="meta">
                    {{ r.subtitle || 'Demo' }}
                    · {{ r.zoneCount }} areas
                  </div>
                </div>
                <span class="chip">Open</span>
              </button>
            </div>
            <div v-else class="recent-list">
              <button
                v-for="r in recent.slice(0, 6)"
                :key="r.id"
                type="button"
                class="recent-item"
                @click="openRecent(r.id)"
              >
                <div style="min-width: 0">
                  <div class="name">{{ r.filename }}</div>
                  <div class="meta">
                    {{ formatWhen(r.createdAt) }}
                    · {{ r.status }}
                    <template v-if="r.zoneCount != null"> · {{ r.zoneCount }} areas</template>
                  </div>
                </div>
                <span class="chip">Open</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div v-if="error" class="error-banner">{{ error }}</div>
    </template>

    <!-- Loading -->
    <div v-else-if="loading" class="card loading-panel">
      <div class="spinner" aria-hidden="true" />
      <h2 class="upload-title" style="margin-bottom: 0.35rem">
        {{ isDemo ? 'Loading demo analysis' : 'Analyzing field flight' }}
      </h2>
      <p class="muted" style="margin: 0">
        {{ loadingCaption }}
      </p>
      <div class="progress-track" aria-hidden="true"><i /></div>
      <div class="pipeline" style="justify-content: center">
        <span v-for="(step, i) in PIPELINE" :key="step" class="pipe-step">
          <b>{{ String(i + 1).padStart(2, '0') }}</b>
          {{ step }}
        </span>
      </div>
      <button
        v-if="activeUploadId"
        type="button"
        class="btn btn-ghost btn-sm"
        style="margin-top: 1rem"
        @click="cancelUpload"
      >
        Cancel upload
      </button>
    </div>

    <!-- Results -->
    <template v-else-if="report">
      <div class="row row-between">
        <div style="min-width: 0">
          <p class="section-label" style="margin-bottom: 0.35rem">
            {{ demoResult ? 'Demo analysis' : 'Analysis complete' }}
          </p>
          <h1 class="page-title" style="font-size: clamp(1.4rem, 2.5vw, 1.85rem)">
            {{ report.source.filename }}
          </h1>
          <p class="muted" style="margin: 0.25rem 0 0">
            {{ report.source.width }}×{{ report.source.height }}
            · {{ report.analysis.framesSampled }} frames @ {{ report.analysis.sampleFps }} FPS
            · {{ report.analysis.totalRuntimeSec.toFixed(1) }}s
            · <span class="mono">{{ report.runId }}</span>
          </p>
        </div>
        <div class="row">
          <a class="btn btn-ghost btn-sm" :href="artifactUrl('results.json')" target="_blank" rel="noopener">
            results.json
          </a>
          <button type="button" class="btn btn-ghost" @click="reset">New analysis</button>
        </div>
      </div>

      <div class="disclaimer-box">
        <div class="ico" aria-hidden="true">!</div>
        <div>{{ report.disclaimer }}</div>
      </div>

      <div v-if="showLimitedAnalysis(report)" class="disclaimer-box" style="border-color: #b8860b">
        <div class="ico" aria-hidden="true">i</div>
        <div>
          <strong>{{ FIELD_COPY.limitedAnalysisTitle }}</strong>
          <p style="margin: 0.35rem 0 0">{{ FIELD_COPY.limitedAnalysisBody }}</p>
        </div>
      </div>

      <div v-if="fieldBrief" class="card flight-brief">
        <h2 class="section-label">Summary</h2>
        <p class="flight-brief-text">{{ fieldBrief }}</p>
      </div>

      <!-- Overview stats -->
      <div class="card">
        <h2 class="section-label">Field overview</h2>
        <div class="stat-grid">
          <div
            class="stat"
            :title="report.field.boundary?.tooltip ?? FIELD_COPY.boundaryTooltip"
          >
            <span class="label">{{ FIELD_COPY.boundaryLabel }}</span>
            <span class="value">{{ report.field.boundary?.confidence ?? 'Medium' }}</span>
            <span class="sub">Vision-estimated analyzed area</span>
          </div>
          <div class="stat" :title="FIELD_COPY.cropCoverageTooltip">
            <span class="label">{{ FIELD_COPY.cropCoverageLabel }}</span>
            <span class="value">{{ pct(estimatedCropCoverage(report)) }}</span>
            <span class="sub">Image-based estimate</span>
          </div>
          <div class="stat">
            <span class="label">{{ FIELD_COPY.inspectionAreasLabel }}</span>
            <span class="value">{{ zones.length }} found</span>
            <span class="sub">Areas worth reviewing</span>
          </div>
          <div class="stat">
            <span class="label">Highest review priority</span>
            <span class="value">
              <span v-if="highestPriority" class="priority" :class="highestPriority">
                {{ reviewPriorityLabel(highestPriority) }}
              </span>
              <span v-else>—</span>
            </span>
            <span class="sub">Most important first</span>
          </div>
          <div class="stat" v-if="report.field.rowVisibility">
            <span class="label">Row visibility</span>
            <span class="value">{{ report.field.rowVisibility }}</span>
            <span class="sub">
              {{
                report.field.rowVisibility === 'LOW'
                  ? 'Gap detail may be limited'
                  : 'Crop rows visible in imagery'
              }}
            </span>
          </div>
        </div>
        <FieldAnalysisLegend />
        <details class="overview-tech" style="margin-top: 0.85rem">
          <summary class="muted" style="cursor: pointer; font-size: 0.82rem">View technical details</summary>
          <div class="stat-grid" style="margin-top: 0.65rem">
            <div class="stat">
              <span class="label">Field area analyzed</span>
              <span class="value">{{
                pct(report.field.cropCoverage?.analyzableFraction ?? report.field.meanFieldFraction)
              }}</span>
            </div>
            <div class="stat">
              <span class="label">Exposed soil (estimate)</span>
              <span class="value">{{
                pct(report.field.cropCoverage?.bareSoilFraction ?? report.field.meanBareSoil)
              }}</span>
            </div>
            <div class="stat">
              <span class="label">Usable frames</span>
              <span class="value">{{ report.analysis.framesUsable }}/{{ report.analysis.framesSampled }}</span>
            </div>
          </div>
        </details>
        <div class="row" style="margin-top: 0.85rem">
          <span class="chip" :class="report.field.roadPathDetected ? 'ok' : ''">
            Road/path {{ report.field.roadPathDetected ? 'yes' : 'no' }}
          </span>
          <span class="chip" :class="report.field.treeVegetationDetected ? 'ok' : ''">
            Trees {{ report.field.treeVegetationDetected ? 'yes' : 'no' }}
          </span>
          <span class="chip" :class="report.field.waterDetected ? 'ok' : ''">
            Water {{ report.field.waterDetected ? 'yes' : 'no' }}
          </span>
        </div>
      </div>

      <!-- Media -->
      <div class="grid-media">
        <div class="card">
          <div class="card-head">
            <h2 class="section-label" style="margin: 0">Field media</h2>
            <div class="tabs" role="tablist">
              <button
                type="button"
                class="tab"
                :class="{ active: mediaTab === 'video' }"
                role="tab"
                @click="mediaTab = 'video'"
              >
                Annotated media
              </button>
              <button
                type="button"
                class="tab"
                :class="{ active: mediaTab === 'frames' }"
                role="tab"
                @click="scrollToFrames"
              >
                All frames
                <span v-if="report.analysis.framesSampled" class="tab-count">
                  {{ report.analysis.framesSampled }}
                </span>
              </button>
              <button
                type="button"
                class="tab"
                :class="{ active: mediaTab === 'heatmap' }"
                role="tab"
                @click="mediaTab = 'heatmap'"
              >
                Variation map
              </button>
              <button
                type="button"
                class="tab"
                :class="{ active: mediaTab === 'montage' }"
                role="tab"
                @click="mediaTab = 'montage'"
              >
                Montage
              </button>
            </div>
          </div>

          <div ref="mediaFrameEl" class="media-frame">
            <video
              v-if="mediaTab === 'video' && mediaSrc"
              ref="annotatedVideoEl"
              :key="mediaSrc"
              controls
              playsinline
              :src="mediaSrc"
            />
            <img
              v-else-if="mediaSrc"
              :key="mediaSrc"
              :src="mediaSrc"
              :alt="mediaTab === 'heatmap' ? 'Visual field variation' : 'Segmentation montage'"
            />
            <div v-else-if="mediaTab === 'frames'" class="frames-tab-hint">
              <p class="muted" style="margin: 0 0 0.75rem">
                {{ report.analysis.framesSampled }} sampled frames with overlays and quality scores.
              </p>
              <button type="button" class="btn btn-primary btn-sm" @click="scrollToFrames">
                Open frame review
              </button>
            </div>
            <p v-else class="muted">Artifact not available</p>
          </div>
          <p class="media-caption">
            <template v-if="mediaTab === 'heatmap'">
              Where the field looks most different from itself in the imagery — not GPS, not a health map
            </template>
            <template v-else-if="mediaTab === 'montage'">
              Snapshot quilt from the flight — good for spotting row patterns at a glance
            </template>
            <template v-else-if="mediaTab === 'frames'">
              Full per-frame strip is below — every sample the engine scored
            </template>
            <template v-else>
              {{ report.mapLabel }}
            </template>
          </p>
        </div>

        <div class="card zone-detail" v-if="selectedZone">
          <div class="row row-between" style="margin-bottom: 0.75rem">
            <div>
              <h2 class="section-label" style="margin-bottom: 0.35rem">
                {{ inspectionAreaTitle(selectedAreaIndex) }}
              </h2>
              <p
                class="zone-loc"
                style="margin-bottom: 0.35rem; font-weight: 600; font-size: 1.05rem"
                :title="inspectionIssueTooltip(selectedZone)"
              >
                {{ inspectionIssueLabel(selectedZone) }}
              </p>
              <div class="row">
                <span class="priority" :class="selectedZone.reviewPriority">
                  Review priority: {{ reviewPriorityLabel(selectedZone.reviewPriority).toUpperCase() }}
                </span>
              </div>
            </div>
            <LocationCompass
              :location="selectedZone.relativeLocation"
              :priority="selectedZone.reviewPriority"
            />
          </div>

          <p class="zone-loc" style="margin-bottom: 0.85rem">
            {{ locationLabel(selectedZone.relativeLocation) }} field
            <template v-if="formatObservations(selectedZone)">
              · {{ formatObservations(selectedZone) }}
            </template>
          </p>

          <h3 class="section-label" style="margin-top: 0.5rem">Why it was flagged</h3>
          <ul>
            <li v-for="(r, i) in formatInspectionReasons(selectedZone)" :key="i">{{ r }}</li>
          </ul>

          <div class="reco">
            <strong>Recommended action</strong>
            {{ selectedZone.recommendation }}
          </div>

          <details class="overview-tech" style="margin-top: 1rem">
            <summary class="muted" style="cursor: pointer">View technical details</summary>
            <dl class="tech-dl" style="margin-top: 0.65rem">
              <dt>Structural score</dt>
              <dd>{{ (selectedZone.structuralAnomalyScore ?? 0).toFixed(2) }}</dd>
              <dt>Appearance score</dt>
              <dd>{{ (selectedZone.appearanceAnomalyScore ?? selectedZone.anomalyScore).toFixed(2) }}</dd>
              <dt>Persistence</dt>
              <dd>
                <template
                  v-if="selectedZone.persistentObservations != null && selectedZone.totalObservations != null"
                >
                  {{ selectedZone.persistentObservations }} / {{ selectedZone.totalObservations }} observations
                </template>
                <template v-else>{{ selectedZone.persistenceScore.toFixed(2) }}</template>
              </dd>
              <template v-if="selectedZone.evidence?.cropCoverageDelta != null">
                <dt>Crop coverage delta</dt>
                <dd>{{ Math.round(selectedZone.evidence.cropCoverageDelta * 100) }}%</dd>
              </template>
              <template v-if="selectedZone.evidence?.soilExposureDelta != null">
                <dt>Soil exposure delta</dt>
                <dd>+{{ Math.round(selectedZone.evidence.soilExposureDelta * 100) }}%</dd>
              </template>
              <dt>Internal ID</dt>
              <dd class="mono">{{ selectedZone.id }}</dd>
            </dl>
          </details>
        </div>

        <div v-else class="card">
          <h2 class="section-label">{{ FIELD_COPY.inspectionAreasLabel }}</h2>
          <p class="muted" style="margin: 0">
            <strong>{{ FIELD_COPY.emptyInspectionAreasTitle }}</strong><br />
            {{ FIELD_COPY.emptyInspectionAreasBody }}
          </p>
        </div>
      </div>

      <!-- Per-frame review -->
      <div ref="frameSectionEl">
        <FrameReview
          v-if="report.runId"
          :run-id="report.runId"
          :frames="report.frameQuality || []"
          :artifact-urls="job?.artifactUrls || {}"
          :frames-sampled="report.analysis.framesSampled"
          :focus-timestamp-sec="selectedZone?.firstSeenSec ?? null"
        />
      </div>

      <!-- Inspection area list -->
      <div class="card" v-if="zones.length">
        <div class="card-head">
          <h2 class="section-label" style="margin: 0">
            {{ FIELD_COPY.inspectionAreasLabel }}
            <span class="muted" style="letter-spacing: 0; text-transform: none; font-weight: 500">
              · {{ zones.length }} found
            </span>
          </h2>
        </div>
        <div class="zone-list" style="max-height: none; display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 0.65rem">
          <ZoneCard
            v-for="(z, idx) in zones"
            :key="z.id"
            :zone="z"
            :area-index="idx"
            :selected="z.id === selectedZone?.id"
            @select="selectedZoneId = z.id"
          />
        </div>
      </div>

      <div class="limitations">
        <details>
          <summary>Limitations & method notes</summary>
          <ul>
            <li v-for="(l, i) in report.limitations" :key="i">{{ l }}</li>
          </ul>
          <p class="muted mono" style="margin: 0.75rem 0 0">
            Backend {{ report.analysis.segmentationBackend }} / {{ report.analysis.dinoBackend }}
            · device {{ report.analysis.device }}
            ·
            <a :href="artifactUrl('metrics.json')" target="_blank" rel="noopener">metrics.json</a>
          </p>
        </details>
      </div>
    </template>
  </div>
</template>
