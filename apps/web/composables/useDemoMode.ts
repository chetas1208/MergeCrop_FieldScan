import type { AnalysisJob } from '@cropmerge/types'

export type AppMode = 'live' | 'demo'

export interface DemoCase {
  id: string
  title: string
  subtitle: string
  kind: 'video' | 'image'
  inputUrl: string
  inputFilename: string
  inputMime: string
  jobUrl: string
  zoneCount: number
  cropCoverage?: number
  runId: string
}

export interface DemoManifest {
  version: number
  description: string
  cases: DemoCase[]
}

const DEMO_STAGES = ['queued', 'preparing', 'processing', 'rendering', 'completed'] as const

export function useDemoMode() {
  const mode = useState<AppMode>('app-mode', () => 'live')
  const manifest = useState<DemoManifest | null>('demo-manifest', () => null)
  const manifestError = useState<string | null>('demo-manifest-error', () => null)

  onMounted(() => {
    if (!import.meta.client) return
    const saved = localStorage.getItem('cropmerge-mode')
    if (saved === 'demo' || saved === 'live') mode.value = saved
  })

  watch(mode, (next) => {
    if (import.meta.client) localStorage.setItem('cropmerge-mode', next)
  })

  const isDemo = computed(() => mode.value === 'demo')
  const demoCases = computed(() => manifest.value?.cases ?? [])

  async function loadManifest(force = false): Promise<DemoManifest | null> {
    if (!force && manifest.value) return manifest.value
    try {
      const res = await fetch('/demo/manifest.json')
      if (!res.ok) throw new Error(`Demo manifest unavailable (${res.status})`)
      const body = (await res.json()) as DemoManifest
      manifest.value = body
      manifestError.value = null
      return body
    } catch (e: unknown) {
      manifestError.value = e instanceof Error ? e.message : String(e)
      return null
    }
  }

  async function fetchDemoJob(caseId: string): Promise<AnalysisJob> {
    const list = (await loadManifest())?.cases ?? []
    const match = list.find((item) => item.id === caseId)
    if (!match) throw new Error(`Demo case "${caseId}" was not found`)
    const res = await fetch(match.jobUrl)
    if (!res.ok) throw new Error(`Demo results unavailable (${res.status})`)
    return (await res.json()) as AnalysisJob
  }

  async function simulateDemoRun(
    onTick: (stage: (typeof DEMO_STAGES)[number], progress: number, message: string) => void,
    signal?: AbortSignal,
  ) {
    const messages: Record<(typeof DEMO_STAGES)[number], string> = {
      queued: 'Queued on demo pipeline',
      preparing: 'Loading stored flight from GitHub',
      processing: 'Replaying segmentation and inspection scoring',
      rendering: 'Preparing heatmap, montage, and frame review',
      completed: 'Demo analysis ready',
    }
    for (let i = 0; i < DEMO_STAGES.length; i += 1) {
      if (signal?.aborted) throw new DOMException('Request aborted', 'AbortError')
      const stage = DEMO_STAGES[i]
      onTick(stage, (i + 1) / DEMO_STAGES.length, messages[stage])
      await new Promise<void>((resolve, reject) => {
        const delay = 500 + i * 180
        const timer = window.setTimeout(resolve, delay)
        signal?.addEventListener(
          'abort',
          () => {
            window.clearTimeout(timer)
            reject(new DOMException('Request aborted', 'AbortError'))
          },
          { once: true },
        )
      })
    }
  }

  function setMode(next: AppMode) {
    mode.value = next
  }

  return {
    mode,
    isDemo,
    manifest,
    manifestError,
    demoCases,
    loadManifest,
    fetchDemoJob,
    simulateDemoRun,
    setMode,
  }
}
