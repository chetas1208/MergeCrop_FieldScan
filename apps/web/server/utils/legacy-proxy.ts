/**
 * The old Nuxt upload/artifact routes are kept for explicitly opted-in local
 * compatibility only. Defaulting this off is what prevents a Vercel function
 * from ever becoming a video or artifact proxy.
 */
export function requireLegacyLocalProxy() {
  if (process.env.NUXT_ENABLE_LEGACY_LOCAL_PROXY !== 'true') {
    throw createError({
      statusCode: 410,
      statusMessage: 'Use the direct vision API; Nuxt does not proxy media in this deployment.',
    })
  }

  const base = String(useRuntimeConfig().visionServiceUrl || '')
  let url: URL
  try {
    url = new URL(base)
  } catch {
    throw createError({ statusCode: 500, statusMessage: 'Invalid local vision service URL' })
  }
  if (!['localhost', '127.0.0.1', '::1'].includes(url.hostname)) {
    throw createError({
      statusCode: 403,
      statusMessage: 'Legacy media proxy may only target a local vision service.',
    })
  }
}
