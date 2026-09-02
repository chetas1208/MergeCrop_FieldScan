<script setup lang="ts">
import type { LiveInspectionAreaEvent } from '@cropmerge/types'
import { computeLetterboxRect, mapNormBoxToDisplay } from '~/composables/live/coordinateMapping'
import { inspectionIssueLabel } from '~/composables/useInspectionLabels'

const props = defineProps<{
  videoEl: HTMLVideoElement | null
  containerEl: HTMLElement | null
  zones: LiveInspectionAreaEvent[]
  selectedZoneId?: string | null
}>()
const emit = defineEmits<{ select: [zone: LiveInspectionAreaEvent] }>()

const canvasEl = ref<HTMLCanvasElement | null>(null)
let rafHandle: number | null = null

// Canvas strokeStyle/fillStyle can't resolve var(--x) — read the app's real
// design tokens once so overlay boxes use the same semantic colors as the
// .priority chips elsewhere, instead of an unrelated ad-hoc palette.
let tokens = { crop: '#9fcf5a', high: '#e86a4e', medium: '#e0a94a', low: '#b8c45a', text: '#eef2e6' }

function readTokens() {
  const style = getComputedStyle(document.documentElement)
  const read = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback
  tokens = {
    crop: read('--crop', tokens.crop),
    high: read('--high', tokens.high),
    medium: read('--medium', tokens.medium),
    low: read('--low', tokens.low),
    text: read('--text', tokens.text),
  }
}

function priorityColor(zone: LiveInspectionAreaEvent): string {
  if (zone.reviewPriority === 'high') return tokens.high
  if (zone.reviewPriority === 'medium') return tokens.medium
  return tokens.low
}

function draw() {
  const canvas = canvasEl.value
  const video = props.videoEl
  const container = props.containerEl
  if (!canvas || !video || !container) return

  const rect = container.getBoundingClientRect()
  canvas.width = rect.width
  canvas.height = rect.height
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const letterbox = computeLetterboxRect(video.videoWidth, video.videoHeight, rect.width, rect.height)

  for (const zone of props.zones) {
    const box = mapNormBoxToDisplay(zone.bboxNorm, letterbox)
    const selected = zone.id === props.selectedZoneId
    ctx.strokeStyle = selected ? tokens.crop : priorityColor(zone)
    ctx.lineWidth = selected ? 3 : 2
    ctx.strokeRect(box.x, box.y, box.w, box.h)

    const label = inspectionIssueLabel(zone)
    ctx.font = '600 12px "DM Sans", system-ui, sans-serif'
    const textWidth = ctx.measureText(label).width
    ctx.fillStyle = 'rgba(12, 14, 10, 0.75)'
    ctx.fillRect(box.x, Math.max(0, box.y - 18), textWidth + 10, 18)
    ctx.fillStyle = tokens.text
    ctx.fillText(label, box.x + 5, Math.max(12, box.y - 5))
  }

  rafHandle = requestAnimationFrame(draw)
}

function handleClick(event: MouseEvent) {
  const canvas = canvasEl.value
  const video = props.videoEl
  if (!canvas || !video) return
  const rect = canvas.getBoundingClientRect()
  const clickX = event.clientX - rect.left
  const clickY = event.clientY - rect.top
  const letterbox = computeLetterboxRect(video.videoWidth, video.videoHeight, rect.width, rect.height)

  for (const zone of props.zones) {
    const box = mapNormBoxToDisplay(zone.bboxNorm, letterbox)
    if (clickX >= box.x && clickX <= box.x + box.w && clickY >= box.y && clickY <= box.y + box.h) {
      emit('select', zone)
      return
    }
  }
}

onMounted(() => {
  readTokens()
  rafHandle = requestAnimationFrame(draw)
})

onBeforeUnmount(() => {
  if (rafHandle) cancelAnimationFrame(rafHandle)
})
</script>

<template>
  <canvas ref="canvasEl" class="inspection-overlay" @click="handleClick" />
</template>

<style scoped>
.inspection-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 1;
  cursor: pointer;
}
</style>
