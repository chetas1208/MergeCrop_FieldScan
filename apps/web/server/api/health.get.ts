import { dbHealth } from '../utils/db'
import { requireLegacyLocalProxy } from '../utils/legacy-proxy'
import { visionHealth } from '../utils/vision'

export default defineEventHandler(async () => {
  requireLegacyLocalProxy()
  const config = useRuntimeConfig()
  const db = dbHealth()
  try {
    const vision = await visionHealth(config.visionServiceUrl as string)
    return {
      ok: db.ok && vision.status === 'ok',
      web: 'ok',
      db,
      vision,
    }
  } catch (e) {
    return {
      ok: false,
      web: 'ok',
      db,
      vision: null,
      error: e instanceof Error ? e.message : String(e),
    }
  }
})
