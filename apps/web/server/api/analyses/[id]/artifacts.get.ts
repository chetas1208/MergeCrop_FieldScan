import { loadJob } from '../../../utils/db'
import { requireLegacyLocalProxy } from '../../../utils/legacy-proxy'

export default defineEventHandler((event) => {
  requireLegacyLocalProxy()
  const id = getRouterParam(event, 'id')
  if (!id) throw createError({ statusCode: 400 })
  const job = loadJob(id)
  if (!job) throw createError({ statusCode: 404 })
  const a = job.artifacts || job.report?.artifacts
  if (!a) return { artifacts: null }
  // Map to Nuxt-served proxy paths when runId known
  const runId = job.report?.runId
  return {
    artifacts: a,
    urls: runId
      ? {
          resultsJson: `/api/artifacts/${runId}/results.json`,
          heatmapPng: `/api/artifacts/${runId}/heatmap.png`,
          annotatedVideo: `/api/artifacts/${runId}/annotated_video.mp4`,
          metricsJson: `/api/artifacts/${runId}/metrics.json`,
        }
      : null,
  }
})
