<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import CollectionImageDetail from '~/components/CollectionImageDetail.vue'
import type { CollectionAnalysisResult, CollectionImageResult } from '~/composables/useCollections'

const props = defineProps<{
  collectionId: string
  analysis: CollectionAnalysisResult
}>()

const PAGE_SIZE = 12
const page = ref(0)
const selectedImageId = ref<string | null>(null)

const totalPages = computed(() => Math.max(1, Math.ceil(props.analysis.imageResults.length / PAGE_SIZE)))
const pagedResults = computed(() => {
  const start = page.value * PAGE_SIZE
  return props.analysis.imageResults.slice(start, start + PAGE_SIZE)
})

function selectImage(result: CollectionImageResult) {
  if (result.status !== 'analyzed') return
  selectedImageId.value = selectedImageId.value === result.imageId ? null : result.imageId
}

function fmtPct(v: number | null): string {
  return v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`
}

function fmtIqr(iqr: [number, number] | null): string {
  return iqr ? `${Math.round(iqr[0] * 100)}–${Math.round(iqr[1] * 100)}%` : '—'
}

watch(
  () => props.analysis,
  () => {
    page.value = 0
    selectedImageId.value = null
  },
)
</script>

<template>
  <div class="card stack-lg collection-results">
    <div class="card-head">
      <h3>Collection results</h3>
      <span class="muted">
        {{ analysis.summary.analyzedCount }} analyzed
        <template v-if="analysis.summary.failedCount">· {{ analysis.summary.failedCount }} failed</template>
        of {{ analysis.summary.totalValidImages }} images
      </span>
    </div>

    <div v-if="analysis.groupStatistics.length" class="stack">
      <div class="section-label">Group comparison</div>
      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th>Group</th>
              <th>Images</th>
              <th>Crop coverage (median, IQR)</th>
              <th>Bare soil (median, IQR)</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="g in analysis.groupStatistics" :key="g.group">
              <td>{{ g.group }}</td>
              <td>{{ g.imageCount }}</td>
              <td>{{ fmtPct(g.cropCoverageMedian) }} <span class="muted">({{ fmtIqr(g.cropCoverageIqr) }})</span></td>
              <td>{{ fmtPct(g.bareSoilMedian) }} <span class="muted">({{ fmtIqr(g.bareSoilIqr) }})</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="stack">
      <div class="row-between">
        <div class="section-label">Images</div>
        <div v-if="totalPages > 1" class="row">
          <button type="button" class="btn btn-ghost btn-sm" :disabled="page === 0" @click="page -= 1">
            Prev
          </button>
          <span class="muted" style="font-size: 0.8rem">Page {{ page + 1 }} of {{ totalPages }}</span>
          <button
            type="button"
            class="btn btn-ghost btn-sm"
            :disabled="page >= totalPages - 1"
            @click="page += 1"
          >
            Next
          </button>
        </div>
      </div>

      <div class="collection-grid">
        <button
          v-for="r in pagedResults"
          :key="r.imageId"
          type="button"
          class="collection-cell"
          :class="{ failed: r.status === 'failed', selected: selectedImageId === r.imageId }"
          :disabled="r.status !== 'analyzed'"
          @click="selectImage(r)"
        >
          <span class="mono" style="font-size: 0.75rem">{{ r.imageId.slice(0, 8) }}</span>
          <span v-if="r.relativeGroup" class="muted" style="font-size: 0.72rem">{{ r.relativeGroup }}</span>
          <span v-if="r.status === 'failed'" class="chip collection-chip-failed">Failed</span>
          <template v-else>
            <span style="font-size: 0.78rem">Crop cover {{ fmtPct(r.cropCoverage) }}</span>
            <span class="muted" style="font-size: 0.72rem">Soil {{ fmtPct(r.bareSoilFraction) }}</span>
          </template>
        </button>
      </div>
    </div>

    <CollectionImageDetail
      v-if="selectedImageId"
      :key="selectedImageId"
      :collection-id="collectionId"
      :image-id="selectedImageId"
    />
  </div>
</template>

<style scoped>
.collection-results {
  margin-top: 1.5rem;
}
.table-scroll {
  overflow-x: auto;
}
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
.data-table th,
.data-table td {
  text-align: left;
  padding: 0.4rem 0.6rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
}
.collection-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.6rem;
}
.collection-cell {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  align-items: flex-start;
  padding: 0.6rem 0.7rem;
  border-radius: 0.6rem;
  border: 1px solid rgba(0, 0, 0, 0.12);
  background: transparent;
  cursor: pointer;
  text-align: left;
  font: inherit;
  color: inherit;
}
.collection-cell:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}
.collection-cell.selected {
  border-color: currentColor;
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25);
}
.collection-cell.failed {
  background: rgba(220, 38, 38, 0.08);
}
.collection-chip-failed {
  background: rgba(220, 38, 38, 0.15);
  color: #991b1b;
}
</style>
