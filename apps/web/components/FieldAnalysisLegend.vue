<script setup lang="ts">
import { FIELD_COPY } from '~/composables/useInspectionLabels'
</script>

<template>
  <div class="field-legend">
    <h3 class="section-label" style="margin: 0 0 0.65rem">Field analysis legend</h3>
    <ul class="field-legend-list">
      <li>
        <span class="legend-swatch legend-boundary" aria-hidden="true" />
        <span>
          <strong>Yellow line</strong> — {{ FIELD_COPY.boundaryLabel }}
        </span>
      </li>
      <li>
        <span class="legend-swatch legend-crop" aria-hidden="true" />
        <span><strong>Green</strong> — Crop</span>
      </li>
      <li>
        <span class="legend-swatch legend-soil" aria-hidden="true" />
        <span><strong>Brown</strong> — Exposed Soil</span>
      </li>
      <li>
        <span class="legend-swatch legend-road" aria-hidden="true" />
        <span><strong>Gray</strong> — Road / Path</span>
      </li>
      <li>
        <span class="legend-swatch legend-trees" aria-hidden="true" />
        <span><strong>Dark green</strong> — Trees / Non-crop vegetation</span>
      </li>
      <li>
        <span class="legend-swatch legend-highlight" aria-hidden="true" />
        <span><strong>Highlighted box</strong> — Inspection Area</span>
      </li>
    </ul>
    <p class="muted" style="font-size: 0.78rem; margin: 0.75rem 0 0">
      Issue types:
      Possible Crop Gap · Sparse Crop Coverage · Visual Difference · Exposed Soil
    </p>
  </div>
</template>

<style scoped>
.field-legend-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.45rem;
  font-size: 0.85rem;
}
.field-legend-list li {
  display: flex;
  align-items: center;
  gap: 0.55rem;
}
.legend-swatch {
  width: 1.1rem;
  height: 1.1rem;
  border-radius: 3px;
  flex-shrink: 0;
  border: 1px solid rgba(255, 255, 255, 0.12);
}
.legend-boundary {
  background: transparent;
  border: 2px solid #ffd54f;
}
.legend-crop {
  background: #28b428;
}
.legend-soil {
  background: #a06432;
}
.legend-road {
  background: #6a6a6a;
}
.legend-trees {
  background: #1a5014;
}
.legend-highlight {
  background: rgba(255, 80, 80, 0.35);
  border: 2px solid #ff5050;
}
</style>
