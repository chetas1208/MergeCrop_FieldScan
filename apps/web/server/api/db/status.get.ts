import { dbHealth, listJobs } from '../../utils/db'
import { requireLegacyLocalProxy } from '../../utils/legacy-proxy'

export default defineEventHandler(() => {
  requireLegacyLocalProxy()
  const health = dbHealth()
  const jobs = health.ok ? listJobs(5) : []
  return {
    ...health,
    recentJobs: jobs.map((j) => ({
      id: j.id,
      status: j.status,
      filename: j.filename,
      createdAt: j.createdAt,
    })),
  }
})
