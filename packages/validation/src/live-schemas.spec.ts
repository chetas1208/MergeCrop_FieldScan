import { describe, expect, it } from 'vitest'
import {
  analysisStatusSchema,
  droneTelemetrySchema,
  liveInspectionAreaEventSchema,
  liveSessionSchema,
  liveStreamEventSchema,
  savedInspectionAreaSchema,
  telemetryAssociationSchema,
} from './index'

const baseZone = {
  id: 'z1',
  reviewPriority: 'high' as const,
  anomalyScore: 0.8,
  persistenceScore: 0.9,
  firstSeenMs: 0,
  lastSeenMs: 1000,
  firstSeenSec: 0,
  lastSeenSec: 1,
  framesSeen: 5,
  relativeLocation: 'center' as const,
  centroidNorm: { x: 0.5, y: 0.5 },
  bboxNorm: { x: 0.4, y: 0.4, w: 0.2, h: 0.2 },
  evidence: {},
  reasons: ['test'],
  recommendation: 'Review',
}

describe('analysisStatusSchema', () => {
  it('accepts the full 7-value status enum (no longer drifted from the TS type)', () => {
    for (const status of ['queued', 'preparing', 'processing', 'rendering', 'completed', 'failed', 'cancelled']) {
      expect(analysisStatusSchema.parse(status)).toBe(status)
    }
  })
})

describe('liveSessionSchema', () => {
  it('parses a valid live session', () => {
    const session = {
      id: 'abc123456789',
      createdAt: '2026-01-01T00:00:00Z',
      status: 'disconnected',
      streamPath: 'cropmerge/live',
      sampleFps: 2,
      simulated: true,
      whepUrl: 'http://127.0.0.1:8889',
    }
    expect(liveSessionSchema.parse(session)).toMatchObject(session)
  })
})

describe('droneTelemetrySchema / telemetryAssociationSchema', () => {
  it('allows null fields — telemetry must never be fabricated', () => {
    const telemetry = {
      timestampMs: 1000,
      latitude: null,
      longitude: null,
      altitudeM: null,
      headingDeg: null,
      gimbalPitchDeg: null,
      source: 'unavailable' as const,
    }
    expect(droneTelemetrySchema.parse(telemetry)).toEqual(telemetry)

    const association = { availability: 'unavailable' as const, telemetry: null, toleranceMs: 2000 }
    expect(telemetryAssociationSchema.parse(association)).toEqual(association)
  })
})

describe('liveInspectionAreaEventSchema', () => {
  it('extends inspectionZoneSchema with the two live-only fields', () => {
    const event = {
      ...baseZone,
      liveFrameTimestampMs: 5000,
      telemetryAssociation: { availability: 'exact', telemetry: null, toleranceMs: 2000 },
    }
    expect(liveInspectionAreaEventSchema.parse(event)).toMatchObject(event)
  })
})

describe('liveStreamEventSchema', () => {
  it('discriminates all four event types', () => {
    expect(liveStreamEventSchema.parse({ type: 'state_change', sessionId: 's1', status: 'live' })).toBeTruthy()
    expect(liveStreamEventSchema.parse({ type: 'error', sessionId: 's1', message: 'boom' })).toBeTruthy()
    expect(
      liveStreamEventSchema.parse({
        type: 'telemetry',
        sessionId: 's1',
        telemetry: {
          timestampMs: 1,
          latitude: null,
          longitude: null,
          altitudeM: null,
          headingDeg: null,
          gimbalPitchDeg: null,
          source: 'unavailable',
        },
      }),
    ).toBeTruthy()
    expect(
      liveStreamEventSchema.parse({
        type: 'frame_analyzed',
        sessionId: 's1',
        frameTimestampMs: 1000,
        zones: [
          {
            ...baseZone,
            liveFrameTimestampMs: 1000,
            telemetryAssociation: { availability: 'unavailable', telemetry: null, toleranceMs: 2000 },
          },
        ],
      }),
    ).toBeTruthy()
  })

  it('rejects an unknown event type', () => {
    expect(() => liveStreamEventSchema.parse({ type: 'bogus', sessionId: 's1' })).toThrow()
  })
})

describe('savedInspectionAreaSchema', () => {
  it('parses a saved area with unavailable telemetry and no fabricated coordinates', () => {
    const saved = {
      id: 's1-abc',
      sessionId: 's1',
      zone: baseZone,
      telemetryAssociation: { availability: 'unavailable', telemetry: null, toleranceMs: 2000 },
      savedAt: '2026-01-01T00:00:00Z',
    }
    expect(savedInspectionAreaSchema.parse(saved)).toMatchObject(saved)
  })
})
