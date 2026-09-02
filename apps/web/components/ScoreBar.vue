<script setup lang="ts">
const props = defineProps<{
  value: number
  tone?: 'high' | 'medium' | 'low' | 'default'
}>()
const width = computed(() => `${Math.round(Math.min(1, Math.max(0, props.value)) * 100)}%`)
</script>

<template>
  <div class="score-bar" :class="tone && tone !== 'default' ? tone : undefined" :title="value.toFixed(2)">
    <i :style="{ width }" />
  </div>
</template>
