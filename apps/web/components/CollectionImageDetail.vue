<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useCollections } from '~/composables/useCollections'
import type { CollectionImageDetail } from '~/composables/useCollections'

const props = defineProps<{
  collectionId: string
  imageId: string
}>()

const { getCollectionImage } = useCollections()

const detail = ref<CollectionImageDetail | null>(null)
const loading = ref(false)
const error = ref('')
const activeArtifact = ref<string | null>(null)

const OVERLAY_LABELS: Record<string, string> = {
  'heatmap.png': 'Anomaly heatmap',
  'segmentation_montage.jpg': 'Segmentation',
}

const artifactOptions = computed(() => {
  if (!detail.value) return []
  return Object.keys(detail.value.artifactUrls)
    .filter((name) => name.endsWith('.png') || name.endsWith('.jpg'))
    .map((name) => ({ name, label: OVERLAY_LABELS[name] || name }))
})

async function load() {
  loading.value = true
  error.value = ''
  detail.value = null
  activeArtifact.value = null
  try {
    detail.value = await getCollectionImage(props.collectionId, props.imageId)
    activeArtifact.value = artifactOptions.value[0]?.name ?? null
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

watch(() => [props.collectionId, props.imageId], load, { immediate: true })
</script>

<template>
  <div class="card card-flush image-detail">
    <div v-if="loading" class="muted" style="padding: 1rem">Loading image detail…</div>
    <div v-else-if="error" class="field-error" style="padding: 1rem">{{ error }}</div>
    <template v-else-if="detail">
      <div class="row-between" style="padding: 0.7rem 0.9rem 0">
        <span class="mono muted" style="font-size: 0.75rem">Run {{ detail.runId }}</span>
        <div v-if="artifactOptions.length > 1" class="row">
          <button
            v-for="opt in artifactOptions"
            :key="opt.name"
            type="button"
            class="btn btn-ghost btn-sm"
            :class="{ 'btn-primary': activeArtifact === opt.name }"
            @click="activeArtifact = opt.name"
          >
            {{ opt.label }}
          </button>
        </div>
      </div>
      <div v-if="activeArtifact" class="image-detail-frame">
        <img :src="detail.artifactUrls[activeArtifact]" :alt="activeArtifact" />
      </div>
      <div v-else class="muted" style="padding: 1rem">No overlay images available for this image.</div>
      <div v-if="detail.field" class="row" style="padding: 0.6rem 0.9rem 0.9rem; gap: 1rem">
        <span class="muted" style="font-size: 0.8rem">
          Field detected: {{ detail.field.detected ? 'yes' : 'no' }}
        </span>
        <span v-if="detail.field.meanCropCoverage != null" class="muted" style="font-size: 0.8rem">
          Crop coverage: {{ Math.round(detail.field.meanCropCoverage * 100) }}%
        </span>
      </div>
    </template>
  </div>
</template>

<style scoped>
.image-detail {
  margin-top: 0.8rem;
}
.image-detail-frame {
  width: 100%;
  max-height: 420px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #0b0f14;
}
.image-detail-frame img {
  width: 100%;
  height: auto;
  display: block;
}
</style>
