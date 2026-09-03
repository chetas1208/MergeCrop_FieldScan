<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  imageUrl: string
  bbox: { x: number; y: number; w: number; h: number }
  alt?: string
  /** Extra normalized padding added around the raw bbox on each side, so
   * the crop shows a little surrounding context instead of framing the
   * anomaly edge-to-edge (real micro-regions can be a few percent of the
   * frame — padding keeps them recognizable, not a single blown-up pixel
   * block). */
  paddingFraction?: number
}>()

const paddedBbox = computed(() => {
  const pad = props.paddingFraction ?? 0.35
  const w = Math.max(props.bbox.w, 0.015)
  const h = Math.max(props.bbox.h, 0.015)
  const padW = w * pad
  const padH = h * pad
  const x = Math.max(0, props.bbox.x - padW)
  const y = Math.max(0, props.bbox.y - padH)
  const w2 = Math.min(1 - x, w + padW * 2)
  const h2 = Math.min(1 - y, h + padH * 2)
  return { x, y, w: w2, h: h2 }
})

const cropStyle = computed(() => ({
  '--bbox-x': paddedBbox.value.x,
  '--bbox-y': paddedBbox.value.y,
  '--bbox-w': paddedBbox.value.w,
  '--bbox-h': paddedBbox.value.h,
}))

const markerStyle = computed(() => {
  const b = props.bbox
  const p = paddedBbox.value
  // Marker position/size relative to the padded crop window, in percent.
  const left = ((b.x - p.x) / p.w) * 100
  const top = ((b.y - p.y) / p.h) * 100
  const width = (b.w / p.w) * 100
  const height = (b.h / p.h) * 100
  return {
    left: `${left}%`,
    top: `${top}%`,
    width: `${width}%`,
    height: `${height}%`,
  }
})
</script>

<template>
  <div class="micro-crop" :style="cropStyle">
    <img v-if="imageUrl" :src="imageUrl" :alt="alt || 'Zoomed view of the flagged region'" />
    <div v-else class="micro-crop-empty muted">No image available</div>
    <div v-if="imageUrl" class="micro-crop-marker" :style="markerStyle" aria-hidden="true" />
  </div>
</template>

<style scoped>
.micro-crop {
  position: relative;
  width: 100%;
  aspect-ratio: 1 / 1;
  overflow: hidden;
  border-radius: 0.6rem;
  background: #0b0f14;
  border: 1px solid rgba(0, 0, 0, 0.12);
}
.micro-crop img {
  position: absolute;
  width: calc(100% / var(--bbox-w));
  height: calc(100% / var(--bbox-h));
  left: calc(-1 * var(--bbox-x) * (100% / var(--bbox-w)));
  top: calc(-1 * var(--bbox-y) * (100% / var(--bbox-h)));
  max-width: none;
  max-height: none;
}
.micro-crop-marker {
  position: absolute;
  border: 2px solid #f97316;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35);
  border-radius: 0.15rem;
  pointer-events: none;
}
.micro-crop-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-size: 0.78rem;
  text-align: center;
  padding: 0.5rem;
}
</style>
