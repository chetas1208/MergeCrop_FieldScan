import { randomUUID } from 'node:crypto'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import type { AnalysisJob } from '@cropmerge/types'
import { saveJob } from '../../utils/db'
import { requireLegacyLocalProxy } from '../../utils/legacy-proxy'
import { visionAnalyze } from '../../utils/vision'

export default defineEventHandler(async (event) => {
  requireLegacyLocalProxy()
  const config = useRuntimeConfig()
  const form = await readMultipartFormData(event)
  if (!form) {
    throw createError({ statusCode: 400, statusMessage: 'Expected multipart form' })
  }

  const filePart = form.find((p) => p.name === 'file' && p.data)
  if (!filePart?.data) {
    throw createError({ statusCode: 400, statusMessage: 'Missing file' })
  }

  const filename = filePart.filename || 'upload.mp4'
  const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'))
  const videoExts = ['.mp4', '.mov', '.m4v']
  const imageExts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
  if (![...videoExts, ...imageExts].includes(ext)) {
    throw createError({
      statusCode: 400,
      statusMessage: 'Supported: .mp4 .mov .m4v video, or .jpg .png .webp .bmp image',
    })
  }
  const isImage = imageExts.includes(ext)

  const get = (name: string) => form.find((p) => p.name === name)?.data?.toString('utf-8')
  const sampleFps = Number(get('sampleFps') || get('sample_fps') || 2)
  const defaultMax = isImage ? 3 : 60
  const maxFrames = Number(get('maxFrames') || get('max_frames') || defaultMax)
  const skipDino = (get('skipDino') || get('skip_dino') || 'true') === 'true'
  const segmentationBackend = get('segmentationBackend') || get('segmentation_backend') || 'heuristic'

  const id = randomUUID().replace(/-/g, '').slice(0, 12)
  const uploads = config.uploadsDir as string
  mkdirSync(uploads, { recursive: true })
  const savePath = join(uploads, `${id}${ext}`)
  writeFileSync(savePath, filePart.data)

  const now = new Date().toISOString()
  const job: AnalysisJob = {
    id,
    status: 'processing',
    createdAt: now,
    updatedAt: now,
    filename,
    progress: 0.1,
    message: 'Calling vision engine…',
  }
  saveJob(job)

  try {
    const report = await visionAnalyze(config.visionServiceUrl as string, savePath, filename, {
      sampleFps,
      maxFrames,
      skipDino,
      segmentationBackend,
    })
    job.status = 'completed'
    job.progress = 1
    job.message = 'Analysis complete'
    job.updatedAt = new Date().toISOString()
    job.report = report
    job.artifacts = report.artifacts
    saveJob(job)
    return job
  } catch (e) {
    job.status = 'failed'
    job.progress = 0
    job.error = e instanceof Error ? e.message : String(e)
    job.message = 'Analysis failed'
    job.updatedAt = new Date().toISOString()
    saveJob(job)
    throw createError({ statusCode: 502, statusMessage: job.error })
  }
})
