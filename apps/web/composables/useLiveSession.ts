import type {
  DroneTelemetry,
  LiveInspectionAreaEvent,
  LiveSession,
  LiveSessionStatus,
  LiveStreamEvent,
} from '@cropmerge/types'
import { negotiateWhep, resolveWhepResourceUrl, toWebSocketUrl } from '~/composables/live/whepClient'

const MAX_RECONNECT_DELAY_MS = 15_000

export function useLiveSession() {
  const config = useRuntimeConfig()
  const { request, baseUrl, getSession } = useVisionApi()

  const whepBase = String(config.public.mediamtxWhepUrl || '').replace(/\/$/, '')
  const liveDroneEnabled = computed(() => whepBase.length > 0)

  const status = ref<LiveSessionStatus>('disconnected')
  const session = ref<LiveSession | null>(null)
  const telemetry = ref<DroneTelemetry | null>(null)
  const zones = ref<LiveInspectionAreaEvent[]>([])
  const errorMessage = ref('')
  const videoStream = ref<MediaStream | null>(null)
  const simulated = ref(false)

  let pc: RTCPeerConnection | null = null
  let ws: WebSocket | null = null
  let whepResourceUrl: string | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let reconnectAttempts = 0
  let stopping = false

  async function waitForIceGathering(peer: RTCPeerConnection): Promise<void> {
    if (peer.iceGatheringState === 'complete') return
    await new Promise<void>((resolve) => {
      const check = () => {
        if (peer.iceGatheringState === 'complete') {
          peer.removeEventListener('icegatheringstatechange', check)
          resolve()
        }
      }
      peer.addEventListener('icegatheringstatechange', check)
    })
  }

  async function connectWhep(streamPath: string): Promise<void> {
    pc = new RTCPeerConnection()
    pc.addTransceiver('video', { direction: 'recvonly' })
    pc.addTransceiver('audio', { direction: 'recvonly' })
    pc.ontrack = (event) => {
      videoStream.value = event.streams[0] ?? null
    }
    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await waitForIceGathering(pc)

    const endpoint = `${whepBase}/${streamPath}/whep`
    const { sdp, location } = await negotiateWhep(endpoint, pc.localDescription!.sdp)
    whepResourceUrl = resolveWhepResourceUrl(endpoint, location)
    await pc.setRemoteDescription({ type: 'answer', sdp })
  }

  function handleEvent(data: LiveStreamEvent) {
    if (data.type === 'frame_analyzed') zones.value = data.zones
    else if (data.type === 'state_change') status.value = data.status
    else if (data.type === 'telemetry') telemetry.value = data.telemetry
    else if (data.type === 'error') errorMessage.value = data.message
  }

  async function connectEvents(sessionId: string): Promise<void> {
    const tokenInfo = await getSession()
    const url = `${toWebSocketUrl(baseUrl)}/vision/live/sessions/${sessionId}/events?token=${encodeURIComponent(tokenInfo.token)}`
    ws = new WebSocket(url)
    ws.onmessage = (event) => {
      try {
        handleEvent(JSON.parse(event.data as string) as LiveStreamEvent)
      } catch {
        // Malformed event — ignore rather than crash the live view.
      }
    }
    ws.onclose = () => {
      if (!stopping) scheduleReconnect(sessionId)
    }
  }

  function scheduleReconnect(sessionId: string) {
    if (reconnectTimer) return
    status.value = 'reconnecting'
    reconnectAttempts += 1
    const delay = Math.min(1000 * 2 ** reconnectAttempts, MAX_RECONNECT_DELAY_MS)
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      connectEvents(sessionId).catch(() => scheduleReconnect(sessionId))
    }, delay)
  }

  async function startSession(opts: { simulated?: boolean } = {}): Promise<void> {
    if (!liveDroneEnabled.value) {
      errorMessage.value = 'Live Drone mode is not configured for this deployment.'
      return
    }
    errorMessage.value = ''
    stopping = false
    reconnectAttempts = 0
    status.value = 'connecting'
    simulated.value = !!opts.simulated
    try {
      const params = new URLSearchParams({ simulated: String(!!opts.simulated) })
      const created = await request<LiveSession>(`/vision/live/sessions?${params}`, { method: 'POST' })
      session.value = created
      await request(`/vision/live/sessions/${created.id}/start-worker`, { method: 'POST' })
      await connectEvents(created.id)
      await connectWhep(created.streamPath)
    } catch (err) {
      status.value = 'disconnected'
      errorMessage.value = err instanceof Error ? err.message : 'Failed to start the live session'
    }
  }

  async function toggleSimulatedInput(start: boolean): Promise<void> {
    await request(`/vision/live/simulate/${start ? 'start' : 'stop'}`, { method: 'POST' })
  }

  async function stopSession(): Promise<void> {
    stopping = true
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    const activeId = session.value?.id
    ws?.close()
    ws = null
    pc?.close()
    pc = null
    if (whepResourceUrl) {
      fetch(whepResourceUrl, { method: 'DELETE' }).catch(() => {})
      whepResourceUrl = null
    }
    videoStream.value = null
    zones.value = []
    telemetry.value = null
    status.value = 'ended'
    if (activeId) {
      await request(`/vision/live/sessions/${activeId}/stop`, { method: 'POST' }).catch(() => {})
    }
  }

  async function saveForFollowUp(zone: LiveInspectionAreaEvent, note?: string) {
    if (!session.value) throw new Error('No active live session')
    return request(`/vision/live/sessions/${session.value.id}/save-area`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zone, liveFrameTimestampMs: zone.liveFrameTimestampMs, note }),
    })
  }

  onBeforeUnmount(() => {
    stopSession().catch(() => {})
  })

  return {
    liveDroneEnabled,
    status,
    session,
    telemetry,
    zones,
    errorMessage,
    videoStream,
    simulated,
    startSession,
    stopSession,
    toggleSimulatedInput,
    saveForFollowUp,
  }
}
