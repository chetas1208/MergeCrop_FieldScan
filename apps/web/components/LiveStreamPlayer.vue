<script setup lang="ts">
const props = defineProps<{
  videoStream: MediaStream | null
  simulated?: boolean
}>()

const videoEl = ref<HTMLVideoElement | null>(null)
const containerEl = ref<HTMLDivElement | null>(null)

watch(
  () => props.videoStream,
  (stream) => {
    if (videoEl.value) videoEl.value.srcObject = stream
  },
  { immediate: true },
)

defineExpose({ videoEl, containerEl })
</script>

<template>
  <div ref="containerEl" class="live-player">
    <video ref="videoEl" autoplay muted playsinline class="live-video" />
    <span v-if="simulated" class="chip warn simulated-badge">
      <span class="dot" />
      SIMULATED LIVE INPUT
    </span>
    <slot />
  </div>
</template>

<style scoped>
.live-player {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: var(--bg-deep);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}
.live-video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.simulated-badge {
  position: absolute;
  top: 0.75rem;
  left: 0.75rem;
  z-index: 2;
}
</style>
