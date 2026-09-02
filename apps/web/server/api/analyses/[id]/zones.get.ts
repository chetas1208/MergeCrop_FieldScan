import { loadJob } from '../../../utils/db'
import { requireLegacyLocalProxy } from '../../../utils/legacy-proxy'

export default defineEventHandler((event) => {
  requireLegacyLocalProxy()
  const id = getRouterParam(event, 'id')
  if (!id) throw createError({ statusCode: 400 })
  const job = loadJob(id)
  if (!job) throw createError({ statusCode: 404 })
  return job.report?.inspectionZones ?? []
})
