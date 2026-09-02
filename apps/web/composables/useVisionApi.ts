type VisionSession = {
  token: string
  expiresAt: string
}

function responseMessage(status: number, body: unknown): string {
  if (body && typeof body === 'object') {
    const detail = (body as { detail?: unknown; statusMessage?: unknown }).detail
      ?? (body as { statusMessage?: unknown }).statusMessage
    if (typeof detail === 'string') return detail
  }
  if (status === 503) {
    return 'Vision server is busy or unavailable. Check the GPU server and try again shortly.'
  }
  if (status === 401) return 'Vision session expired. Refresh the page and try again.'
  if (status === 403) return 'This origin is not allowed to reach the vision server.'
  return `Vision request failed (${status})`
}

export function useVisionApi() {
  const config = useRuntimeConfig()
  const session = useState<VisionSession | null>('vision-session', () => null)
  const baseUrl = String(config.public.visionApiUrl).replace(/\/$/, '')
  let pendingSession: Promise<VisionSession> | null = null

  async function getSession(force = false): Promise<VisionSession> {
    if (!force && session.value && new Date(session.value.expiresAt).getTime() > Date.now() + 30_000) {
      return session.value
    }
    if (!force && pendingSession) return pendingSession

    pendingSession = (async () => {
      const response = await fetch('/api/vision/session', { method: 'POST', credentials: 'same-origin' })
      const body = await response.json().catch(() => null)
      if (!response.ok || !body?.token || !body?.expiresAt) {
        throw new Error(responseMessage(response.status, body))
      }
      const next = { token: String(body.token), expiresAt: String(body.expiresAt) }
      session.value = next
      return next
    })()

    try {
      return await pendingSession
    } finally {
      pendingSession = null
    }
  }

  async function request<T>(
    path: string,
    init: RequestInit = {},
    authenticated = true,
    retried = false,
  ): Promise<T> {
    const headers = new Headers(init.headers)
    if (authenticated) headers.set('Authorization', `Bearer ${(await getSession()).token}`)
    const response = await fetch(`${baseUrl}${path}`, {
      ...init,
      headers,
      credentials: 'omit',
    })

    if (authenticated && response.status === 401 && !retried) {
      await getSession(true)
      return request<T>(path, init, authenticated, true)
    }

    if (!response.ok) {
      const body = await response.json().catch(() => null)
      throw new Error(responseMessage(response.status, body))
    }

    if (response.status === 204) return undefined as T
    return (await response.json()) as T
  }

  return { baseUrl, request, getSession }
}
