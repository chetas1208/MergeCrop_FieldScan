<script setup lang="ts">
import type { FrameQuality } from '@cropmerge/types'

const props = defineProps<{
  runId: string
  frames: FrameQuality[]
  /** Short-lived, direct vision-host URLs keyed by artifact-relative path. */
  artifactUrls: Record<string, string>
  framesSampled?: number
  /** Prefer jumping to the frame nearest this timestamp (seconds). */
  focusTimestampSec?: number | null
}>()

const selectedIndex = ref(0)
const viewMode = ref<'overlay' | 'raw' | 'split'>('overlay')
const overlayKind = ref<'segmentation' | 'continuity'>('segmentation')
const playing = ref(false)
let playTimer: ReturnType<typeof setInterval> | null = null

const rootEl = ref<HTMLElement | null>(null)
const stripEl = ref<HTMLElement | null>(null)

const frameList = computed(() => {
  const fromQuality = props.frames ?? []
  if (fromQuality.length) {
    return [...fromQuality].sort((a, b) => a.frameIndex - b.frameIndex)
  }
  // Fallback when quality list missing but we know sample count
  const n = props.framesSampled ?? 0
  return Array.from({ length: n }, (_, i) => ({
    frameIndex: i,
    timestampSec: i * 0.5,
    sharpness: 0,
    exposureScore: 0,
    meanLuminance: 0,
    nearBlackFraction: 0,
    saturatedFraction: 0,
    usable: true,
    qualityWeight: 1,
    warnings: [] as string[],
  }))
})

const count = computed(() => frameList.value.length)

const selected = computed(() => frameList.value[selectedIndex.value] ?? null)

function pad(i: number) {
  return String(i).padStart(4, '0')
}

function rawUrl(i: number) {
  return props.artifactUrls[`frames/frame_${pad(i)}.jpg`] || ''
}

function overlayUrl(i: number) {
  const sub = overlayKind.value === 'continuity' ? 'continuity' : 'overlays'
  const prefix = overlayKind.value === 'continuity' ? 'continuity' : 'overlay'
  return props.artifactUrls[`${sub}/${prefix}_${pad(i)}.jpg`] || props.artifactUrls[`overlays/overlay_${pad(i)}.jpg`] || ''
}

function select(i: number) {
  if (!count.value) return
  selectedIndex.value = Math.max(0, Math.min(count.value - 1, i))
  scrollThumbIntoView()
}

function prev() {
  select(selectedIndex.value - 1)
}

function next() {
  select(selectedIndex.value + 1)
}

function scrollThumbIntoView() {
  if (playing.value) return
  nextTick(() => {
    const root = stripEl.value
    if (!root) return
    const thumb = root.querySelector<HTMLElement>(`[data-frame-idx="${selectedIndex.value}"]`)
    if (!thumb) return
    scrollFilmstripThumb(root, thumb)
  })
}

function stopPlay() {
  playing.value = false
  if (playTimer) {
    clearInterval(playTimer)
    playTimer = null
  }
}

function togglePlay() {
  if (playing.value) {
    stopPlay()
    return
  }
  if (!count.value) return
  stopPlay()
  playing.value = true
  playTimer = setInterval(() => {
    if (selectedIndex.value >= count.value - 1) {
      select(0)
    } else {
      next()
    }
  }, 450)
}

function onKey(e: KeyboardEvent) {
  if (!isInsideFrameReview(e.target)) return
  const t = e.target as HTMLElement | null
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
  if (e.key === 'ArrowLeft') {
    e.preventDefault()
    stopPlay()
    prev()
  } else if (e.key === 'ArrowRight') {
    e.preventDefault()
    stopPlay()
    next()
  } else if (e.key === ' ' || e.key === 'Spacebar') {
    e.preventDefault()
    togglePlay()
  } else if (e.key === '1') {
    viewMode.value = 'raw'
  } else if (e.key === '2') {
    viewMode.value = 'overlay'
  } else if (e.key === '3') {
    viewMode.value = 'split'
  }
}

watch(
  () => props.runId,
  () => {
    stopPlay()
    selectedIndex.value = 0
  },
)

watch(count, (n) => {
  if (selectedIndex.value >= n) selectedIndex.value = Math.max(0, n - 1)
})

watch(
  () => [props.focusTimestampSec, count.value] as const,
  ([ts]) => {
    if (ts == null || !frameList.value.length) return
    let best = 0
    let bestDist = Infinity
    frameList.value.forEach((fr, i) => {
      const d = Math.abs(fr.timestampSec - ts)
      if (d < bestDist) {
        bestDist = d
        best = i
      }
    })
    stopPlay()
    select(best)
  },
)

onMounted(() => {
  window.addEventListener('keydown', onKey)
})

useViewportPlaybackPause(rootEl, {
  threshold: 0.3,
  isPlaying: () => playing.value,
  onPause: stopPlay,
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey)
  stopPlay()
})

function pct(n: number | undefined) {
  if (n == null || Number.isNaN(n)) return '—'
  return `${Math.round(n * 100)}%`
}
</script>

<template>
  <div ref="rootEl" class="card frame-review" tabindex="-1">
    <div class="card-head">
      <div>
        <h2 class="section-label" style="margin: 0">
          Sampled frames
          <span class="muted" style="letter-spacing: 0; text-transform: none; font-weight: 500">
            · {{ count }}
          </span>
        </h2>
        <p class="muted" style="margin: 0.35rem 0 0; font-size: 0.82rem">
          Every sampled frame — click a thumbnail, use ← →, or play to step through.
        </p>
      </div>
      <div class="frame-toolbar">
        <div class="tabs" role="tablist" aria-label="Overlay view" v-if="viewMode !== 'raw'">
          <span class="muted" style="font-size: 0.72rem; align-self: center; margin-right: 0.25rem">Overlay</span>
          <button
            type="button"
            class="tab"
            :class="{ active: overlayKind === 'segmentation' }"
            @click="overlayKind = 'segmentation'"
          >
            Field segmentation
          </button>
          <button
            type="button"
            class="tab"
            :class="{ active: overlayKind === 'continuity' }"
            title="Highlights possible gaps and fragmented crop areas"
            @click="overlayKind = 'continuity'"
          >
            Crop continuity
          </button>
        </div>
        <div class="tabs" role="tablist" aria-label="Frame view mode">
          <button
            type="button"
            class="tab"
            :class="{ active: viewMode === 'raw' }"
            @click="viewMode = 'raw'"
          >
            Raw
          </button>
          <button
            type="button"
            class="tab"
            :class="{ active: viewMode === 'overlay' }"
            @click="viewMode = 'overlay'"
          >
            Overlay
          </button>
          <button
            type="button"
            class="tab"
            :class="{ active: viewMode === 'split' }"
            @click="viewMode = 'split'"
          >
            Split
          </button>
        </div>
        <div class="frame-nav">
          <button type="button" class="btn btn-ghost btn-sm" :disabled="selectedIndex <= 0" @click="stopPlay(); prev()">
            ← Prev
          </button>
          <button
            type="button"
            class="btn btn-ghost btn-sm"
            :disabled="!count"
            :aria-label="playing ? 'Pause replay' : 'Play replay'"
            @click="togglePlay"
          >
            {{ playing ? 'Pause' : 'Play' }}
          </button>
          <button
            type="button"
            class="btn btn-ghost btn-sm"
            :disabled="selectedIndex >= count - 1"
            @click="stopPlay(); next()"
          >
            Next →
          </button>
        </div>
      </div>
    </div>

    <div v-if="!count" class="muted" style="padding: 1rem 0">No sampled frames available for this run.</div>

    <template v-else>
      <div class="frame-stage" :class="{ split: viewMode === 'split' }">
        <template v-if="viewMode === 'split' && selected">
          <figure class="frame-pane">
            <img :src="rawUrl(selected.frameIndex)" :alt="`Raw frame ${selected.frameIndex}`" />
            <figcaption>Raw</figcaption>
          </figure>
          <figure class="frame-pane">
            <img :src="overlayUrl(selected.frameIndex)" :alt="`Overlay frame ${selected.frameIndex}`" />
            <figcaption>Overlay</figcaption>
          </figure>
        </template>
        <figure v-else-if="selected" class="frame-pane single">
          <img
            :key="`${viewMode}-${selected.frameIndex}`"
            :src="viewMode === 'raw' ? rawUrl(selected.frameIndex) : overlayUrl(selected.frameIndex)"
            :alt="`Frame ${selected.frameIndex}`"
          />
        </figure>
      </div>

      <div v-if="selected" class="frame-meta">
        <div class="frame-meta-main">
          <span class="mono frame-id">#{{ pad(selected.frameIndex) }}</span>
          <span class="chip" :class="selected.usable ? 'ok' : 'bad'">
            <span class="dot" />
            {{ selected.usable ? 'Usable' : 'Skipped' }}
          </span>
          <span class="muted mono">{{ selected.timestampSec.toFixed(2) }}s</span>
          <span class="muted">{{ selectedIndex + 1 }} / {{ count }}</span>
        </div>
        <details class="frame-tech" style="margin-top: 0.65rem">
          <summary class="muted" style="cursor: pointer; font-size: 0.78rem">View frame quality details</summary>
          <div class="frame-quality-grid" style="margin-top: 0.5rem">
          <div class="fq">
            <label>Sharpness</label>
            <strong>{{ pct(selected.sharpness) }}</strong>
            <ScoreBar :value="selected.sharpness" />
          </div>
          <div class="fq">
            <label>Exposure</label>
            <strong>{{ pct(selected.exposureScore) }}</strong>
            <ScoreBar :value="selected.exposureScore" />
          </div>
          <div class="fq">
            <label>Weight</label>
            <strong>{{ pct(selected.qualityWeight) }}</strong>
            <ScoreBar :value="selected.qualityWeight" />
          </div>
          <div class="fq">
            <label>Luminance</label>
            <strong>{{ Math.round(selected.meanLuminance) }}</strong>
            <span class="muted" style="font-size: 0.72rem">mean 0–255</span>
          </div>
          </div>
        </details>
        <ul v-if="selected.warnings?.length" class="frame-warnings">
          <li v-for="(w, i) in selected.warnings" :key="i">{{ w }}</li>
        </ul>
      </div>

      <div ref="stripEl" class="frame-strip" role="listbox" aria-label="Frame filmstrip">
        <button
          v-for="(fr, i) in frameList"
          :key="fr.frameIndex"
          type="button"
          class="frame-thumb"
          :class="{ active: i === selectedIndex, unusable: !fr.usable }"
          :data-frame-idx="i"
          role="option"
          :aria-selected="i === selectedIndex"
          :title="`Frame ${fr.frameIndex} @ ${fr.timestampSec.toFixed(2)}s`"
          @click="stopPlay(); select(i)"
        >
          <img
            :src="overlayUrl(fr.frameIndex)"
            :alt="`Frame ${fr.frameIndex}`"
            loading="lazy"
            @error="($event.target as HTMLImageElement).src = rawUrl(fr.frameIndex)"
          />
          <span class="thumb-label">
            <b>{{ fr.frameIndex }}</b>
            <i>{{ fr.timestampSec.toFixed(1) }}s</i>
          </span>
          <span v-if="!fr.usable" class="thumb-badge">skip</span>
        </button>
      </div>
    </template>
  </div>
</template>
