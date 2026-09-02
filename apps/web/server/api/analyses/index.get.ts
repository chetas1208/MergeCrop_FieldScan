import { listJobs } from '../../utils/db'
import { requireLegacyLocalProxy } from '../../utils/legacy-proxy'

export default defineEventHandler(() => {
  requireLegacyLocalProxy()
  return listJobs().map((j) => ({
    id: j.id,
    status: j.status,
    createdAt: j.createdAt,
    updatedAt: j.updatedAt,
    filename: j.filename,
    progress: j.progress,
    message: j.message,
    error: j.error,
    zoneCount: j.report?.inspectionZones?.length ?? 0,
    fieldDetected: j.report?.field?.detected ?? null,
  }))
})
