import { createReadStream, existsSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { sendStream } from 'h3'
import { requireLegacyLocalProxy } from '../../../utils/legacy-proxy'

export default defineEventHandler((event) => {
  requireLegacyLocalProxy()
  const runId = getRouterParam(event, 'runId')
  const nameParam = getRouterParam(event, 'name')
  // Nitro catch-all may be string or string[]
  let name = Array.isArray(nameParam) ? nameParam.join('/') : (nameParam || '')
  name = name.replace(/^\/+/, '')
  if (!runId || !name || runId.includes('..') || name.includes('..')) {
    throw createError({ statusCode: 400, statusMessage: 'Invalid path' })
  }
  const config = useRuntimeConfig()
  const root = resolve(config.outputsDir as string)
  const file = resolve(join(root, runId, name))
  const rootPrefix = root.endsWith('/') ? root : root + '/'
  if (!file.startsWith(rootPrefix) && file !== root) {
    throw createError({ statusCode: 400, statusMessage: 'Invalid path' })
  }
  if (!existsSync(file) || !statSync(file).isFile()) {
    throw createError({ statusCode: 404, statusMessage: `Not found: ${name}` })
  }
  const st = statSync(file)
  const lower = name.toLowerCase()
  if (lower.endsWith('.json')) setHeader(event, 'content-type', 'application/json')
  else if (lower.endsWith('.png')) setHeader(event, 'content-type', 'image/png')
  else if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) setHeader(event, 'content-type', 'image/jpeg')
  else if (lower.endsWith('.mp4')) setHeader(event, 'content-type', 'video/mp4')
  setHeader(event, 'content-length', st.size)
  setHeader(event, 'cache-control', 'public, max-age=3600')
  if (event.method === 'HEAD') {
    return null
  }
  return sendStream(event, createReadStream(file))
})
