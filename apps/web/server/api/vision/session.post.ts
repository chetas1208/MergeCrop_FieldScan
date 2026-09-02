import { randomUUID } from 'node:crypto'
import { SignJWT } from 'jose'

function configuredOrigins(value: string): Set<string> {
  return new Set(
    value
      .split(',')
      .map((origin) => origin.trim().replace(/\/$/, ''))
      .filter(Boolean),
  )
}

export default defineEventHandler(async (event) => {
  const config = useRuntimeConfig()
  const secret = String(config.visionSharedSecret || '')
  if (Buffer.byteLength(secret, 'utf8') < 32) {
    throw createError({
      statusCode: 503,
      statusMessage: 'Vision session signing is not configured',
    })
  }

  // This endpoint mints a bearer token, so it is intentionally browser-only.
  // The Nuxt origin is always allowed; extra trusted origins are explicit.
  const origin = getHeader(event, 'origin')?.replace(/\/$/, '')
  const currentOrigin = getRequestURL(event).origin.replace(/\/$/, '')
  const allowedOrigins = configuredOrigins(String(config.visionSessionOrigins || ''))
  if (!origin || (origin !== currentOrigin && !allowedOrigins.has(origin))) {
    throw createError({ statusCode: 403, statusMessage: 'Untrusted session origin' })
  }

  const now = new Date()
  const expiresAt = new Date(now.getTime() + 15 * 60 * 1000)
  const token = await new SignJWT({ purpose: 'vision-session' })
    .setProtectedHeader({ alg: 'HS256', typ: 'JWT' })
    .setIssuer('cropmerge-web')
    .setJti(randomUUID())
    .setIssuedAt(Math.floor(now.getTime() / 1000))
    .setExpirationTime(Math.floor(expiresAt.getTime() / 1000))
    .sign(new TextEncoder().encode(secret))

  setHeader(event, 'cache-control', 'no-store')
  return { token, expiresAt: expiresAt.toISOString() }
})
