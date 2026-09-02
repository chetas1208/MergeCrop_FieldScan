import { describe, expect, it } from 'vitest'
import { computeLetterboxRect, mapNormBoxToDisplay } from './coordinateMapping'

describe('computeLetterboxRect', () => {
  it('pillarboxes when the video is narrower than the container', () => {
    // video 4:3 inside a 16:9 container -> letterboxed on the sides
    const rect = computeLetterboxRect(400, 300, 800, 300)
    expect(rect.h).toBe(300)
    expect(rect.w).toBeCloseTo(400)
    expect(rect.x).toBeCloseTo(200)
    expect(rect.y).toBe(0)
  })

  it('letterboxes when the video is wider than the container', () => {
    // video 16:9 inside a taller container -> bars on top/bottom
    const rect = computeLetterboxRect(1600, 900, 800, 800)
    expect(rect.w).toBe(800)
    expect(rect.h).toBeCloseTo(450)
    expect(rect.y).toBeCloseTo(175)
    expect(rect.x).toBe(0)
  })

  it('fills the container exactly when aspect ratios match', () => {
    const rect = computeLetterboxRect(1920, 1080, 960, 540)
    expect(rect).toEqual({ x: 0, y: 0, w: 960, h: 540 })
  })

  it('falls back to the raw container when video dimensions are unknown', () => {
    const rect = computeLetterboxRect(0, 0, 640, 360)
    expect(rect).toEqual({ x: 0, y: 0, w: 640, h: 360 })
  })
})

describe('mapNormBoxToDisplay', () => {
  it('maps a normalized box into the letterboxed pixel space', () => {
    const letterbox = { x: 100, y: 0, w: 400, h: 300 }
    const box = mapNormBoxToDisplay({ x: 0.5, y: 0.5, w: 0.2, h: 0.2 }, letterbox)
    expect(box).toEqual({ x: 100 + 0.5 * 400, y: 0.5 * 300, w: 0.2 * 400, h: 0.2 * 300 })
  })

  it('maps the full-frame box onto the full letterbox rect', () => {
    const letterbox = { x: 10, y: 20, w: 200, h: 100 }
    const box = mapNormBoxToDisplay({ x: 0, y: 0, w: 1, h: 1 }, letterbox)
    expect(box).toEqual(letterbox)
  })
})
