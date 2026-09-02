<script setup lang="ts">
import type { AppMode } from '~/composables/useDemoMode'

const health = ref<{
  ok?: boolean
  db?: { ok?: boolean; mode?: string }
  vision?: { status?: string; device?: string; segmentationBackend?: string }
} | null>(null)
const { request } = useVisionApi()
const { mode, isDemo, setMode, loadManifest } = useDemoMode()

onMounted(async () => {
  await loadManifest()
  try {
    const vision = await request<NonNullable<typeof health.value>['vision']>('/vision/health', {}, false)
    health.value = { ok: vision?.status === 'ok', vision }
  } catch {
    health.value = { ok: false }
  }
})

const visionOk = computed(() => isDemo.value || health.value?.vision?.status === 'ok')
const dbOk = computed(() => health.value?.db?.ok !== false)

function chooseMode(next: AppMode) {
  setMode(next)
}
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <NuxtLink to="/" class="brand" style="text-decoration: none; color: inherit">
        <div class="brand-mark" aria-hidden="true" />
        <div class="brand-text">
          <span class="logo">CropMerge</span>
          <span class="product">Field Triage</span>
        </div>
      </NuxtLink>

      <div class="topbar-right">
        <NuxtLink to="/live-drone" class="btn btn-ghost btn-sm" style="text-decoration: none">Live Drone</NuxtLink>
        <div class="mode-toggle" role="group" aria-label="Analysis mode">
          <button
            type="button"
            class="mode-btn"
            :class="{ active: !isDemo }"
            @click="chooseMode('live')"
          >
            Live
          </button>
          <button
            type="button"
            class="mode-btn"
            :class="{ active: isDemo }"
            @click="chooseMode('demo')"
          >
            Demo
          </button>
        </div>
        <span class="chip" :class="isDemo ? 'warn' : visionOk ? 'ok' : 'bad'" title="Vision engine">
          <span class="dot" />
          {{ isDemo ? 'Demo data' : `Vision ${health?.vision?.status === 'ok' ? 'online' : 'offline'}` }}
        </span>
        <span class="chip" :class="dbOk ? 'ok' : 'warn'" title="Product database">
          <span class="dot" />
          DB {{ health?.db?.mode || 'sqlite' }}
        </span>
        <!-- Device/backend name (cuda/sam2/etc.) is diagnostic detail, not
             primary-interface copy — see /api/health for the raw values. -->
      </div>
    </header>

    <main class="main">
      <NuxtPage />
    </main>

    <footer class="footer">
      <p>
        Exploratory RGB analysis for human review — not disease, nutrient, irrigation, or yield diagnosis.
        Image-relative maps are not georeferenced unless GPS is present.
      </p>
    </footer>
  </div>
</template>
