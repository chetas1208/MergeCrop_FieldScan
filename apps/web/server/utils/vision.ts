import { readFile } from 'node:fs/promises'
import type { FieldTriageReport, VisionHealth } from '@cropmerge/types'

export async function visionHealth(baseUrl: string): Promise<VisionHealth> {
  const res = await fetch(`${baseUrl.replace(/\/$/, '')}/vision/health`)
  if (!res.ok) throw new Error(`Vision health failed: ${res.status}`)
  return (await res.json()) as VisionHealth
}

export async function visionAnalyze(
  baseUrl: string,
  filePath: string,
  filename: string,
  opts: {
    sampleFps?: number
    maxFrames?: number
    skipDino?: boolean
    segmentationBackend?: string
  } = {},
): Promise<FieldTriageReport> {
  const buf = await readFile(filePath)
  const form = new FormData()
  form.append('file', new Blob([buf]), filename)
  form.append('sample_fps', String(opts.sampleFps ?? 2))
  form.append('max_frames', String(opts.maxFrames ?? 120))
  form.append('skip_dino', String(opts.skipDino ?? false))
  form.append('segmentation_backend', opts.segmentationBackend ?? 'heuristic')
  form.append('dino_backend', 'heuristic')

  const res = await fetch(`${baseUrl.replace(/\/$/, '')}/vision/analyze`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Vision analyze failed (${res.status}): ${text}`)
  }
  return (await res.json()) as FieldTriageReport
}
