export interface WhepAnswer {
  sdp: string
  location: string | null
}

/** POSTs a WHEP SDP offer and returns the answer + resource Location.
 * Pure request/response shaping, injectable fetch — no RTCPeerConnection
 * here, so this is unit-testable without a browser WebRTC stack. */
export async function negotiateWhep(
  whepEndpointUrl: string,
  offerSdp: string,
  fetchImpl: typeof fetch = fetch,
): Promise<WhepAnswer> {
  const response = await fetchImpl(whepEndpointUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/sdp' },
    body: offerSdp,
  })
  if (!response.ok) {
    throw new Error(`WHEP negotiation failed (${response.status})`)
  }
  const sdp = await response.text()
  const location = response.headers.get('Location')
  return { sdp, location }
}

/** Resolves the WHEP resource URL (used for a later DELETE to tear down
 * the session) against the endpoint URL, since Location may be relative. */
export function resolveWhepResourceUrl(whepEndpointUrl: string, location: string | null): string | null {
  if (!location) return null
  try {
    return new URL(location, whepEndpointUrl).toString()
  } catch {
    return null
  }
}

/** ws(s):// form of an http(s):// base URL, for the live events socket. */
export function toWebSocketUrl(httpBaseUrl: string): string {
  return httpBaseUrl.replace(/^http/, 'ws')
}
