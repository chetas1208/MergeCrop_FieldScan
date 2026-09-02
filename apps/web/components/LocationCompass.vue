<script setup lang="ts">
import type { RelativeLocation, ReviewPriority } from '@cropmerge/types'

const props = defineProps<{
  location?: RelativeLocation | string | null
  priority?: ReviewPriority | string
}>()

const order: RelativeLocation[] = [
  'northwest',
  'north',
  'northeast',
  'west',
  'center',
  'east',
  'southwest',
  'south',
  'southeast',
]
</script>

<template>
  <div class="compass" :title="location || 'field location'" aria-hidden="true">
    <div
      v-for="cell in order"
      :key="cell"
      class="compass-cell"
      :class="{ on: cell === location, [priority || '']: cell === location }"
    />
  </div>
</template>
