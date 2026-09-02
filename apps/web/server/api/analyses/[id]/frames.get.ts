import { existsSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { loadJob } from '../../../utils/db'
import { requireLegacyLocalProxy } from '../../../utils/legacy-proxy'

function listJpgs(dir: string | null | undefined): string[] {
  if (!dir || !existsSync(dir)) return []
  try {
    return readdirSync(dir)
      .filter((f) => /\.jpe?g$/i.test(f))
      .sort()
  } catch {
    return []
  }
}

export default defineEventHandler((event) => {
  requireLegacyLocalProxy()
  const id = getRouterParam(event, 'id')
  if (!id) throw createError({ statusCode: 400 })
  const job = loadJob(id)
  if (!job) throw createError({ statusCode: 404 })

  const runId = job.report?.runId
  const frameQuality = job.report?.frameQuality ?? []
  const framesDir = job.report?.artifacts?.framesDir ?? null
  const overlaysDir = job.report?.artifacts?.overlaysDir ?? null

  const frameFiles = listJpgs(framesDir)
  const overlayFiles = listJpgs(overlaysDir)

  const frames = frameQuality.length
    ? frameQuality.map((fq) => {
        const idx = String(fq.frameIndex).padStart(4, '0')
        const rawName = `frame_${idx}.jpg`
        const overlayName = `overlay_${idx}.jpg`
        return {
          ...fq,
          rawUrl: runId ? `/api/artifacts/${runId}/frames/${rawName}` : null,
          overlayUrl: runId ? `/api/artifacts/${runId}/overlays/${overlayName}` : null,
        }
      })
    : frameFiles.map((name, i) => {
        const m = name.match(/(\d+)/)
        const frameIndex = m ? Number(m[1]) : i
        const idx = String(frameIndex).padStart(4, '0')
        return {
          frameIndex,
          timestampSec: frameIndex * 0.5,
          usable: true,
          rawUrl: runId ? `/api/artifacts/${runId}/frames/frame_${idx}.jpg` : null,
          overlayUrl: runId ? `/api/artifacts/${runId}/overlays/overlay_${idx}.jpg` : null,
        }
      })

  return {
    runId: runId ?? null,
    frameQuality,
    frames,
    frameFiles,
    overlayFiles,
    framesDir,
    overlaysDir,
    count: frames.length,
  }
})
