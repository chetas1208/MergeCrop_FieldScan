import { describe, expect, it, vi } from 'vitest'
import { negotiateWhep, resolveWhepResourceUrl, toWebSocketUrl } from './whepClient'

describe('negotiateWhep', () => {
  it('posts the SDP offer and returns the answer + Location header', async () => {
    const fetchImpl = vi.fn(async () =>
      new Response('v=0\r\n...answer...', {
        status: 201,
        headers: { Location: '/whep/resource/abc123' },
      }),
    )
    const result = await negotiateWhep('http://mediamtx/cropmerge/live/whep', 'v=0\r\n...offer...', fetchImpl)
    expect(result.sdp).toContain('answer')
    expect(result.location).toBe('/whep/resource/abc123')
    expect(fetchImpl).toHaveBeenCalledWith(
      'http://mediamtx/cropmerge/live/whep',
      expect.objectContaining({ method: 'POST', body: 'v=0\r\n...offer...' }),
    )
  })

  it('throws with the status code when the server rejects the offer', async () => {
    const fetchImpl = vi.fn(async () => new Response('bad request', { status: 400 }))
    await expect(negotiateWhep('http://mediamtx/x/whep', 'offer', fetchImpl)).rejects.toThrow('400')
  })
})

describe('resolveWhepResourceUrl', () => {
  it('resolves a relative Location against the endpoint URL', () => {
    const resolved = resolveWhepResourceUrl('http://mediamtx:8889/cropmerge/live/whep', '/whep/resource/abc123')
    expect(resolved).toBe('http://mediamtx:8889/whep/resource/abc123')
  })

  it('returns null when there is no Location header', () => {
    expect(resolveWhepResourceUrl('http://mediamtx:8889/x/whep', null)).toBeNull()
  })
})

describe('toWebSocketUrl', () => {
  it('converts http to ws', () => {
    expect(toWebSocketUrl('http://127.0.0.1:8001')).toBe('ws://127.0.0.1:8001')
  })

  it('converts https to wss', () => {
    expect(toWebSocketUrl('https://vision.example.com')).toBe('wss://vision.example.com')
  })
})
