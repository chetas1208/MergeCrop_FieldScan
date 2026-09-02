import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../..', import.meta.url))

export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: false },
  css: ['~/assets/css/main.css'],
  runtimeConfig: {
    // Kept private: only the Nuxt session endpoint reads this value.
    visionSharedSecret: process.env.NUXT_VISION_SHARED_SECRET || '',
    visionSessionOrigins: process.env.NUXT_VISION_SESSION_ORIGINS || '',
    // Legacy server-side proxy settings. They are deliberately not used by the
    // deployed UI; large media must go browser -> vision service directly.
    visionServiceUrl: process.env.VISION_SERVICE_URL || 'http://127.0.0.1:8001',
    outputsDir: process.env.OUTPUTS_DIR || `${root}/outputs`,
    uploadsDir: process.env.UPLOADS_DIR || `${root}/data/uploads`,
    databaseDir: process.env.DATABASE_DIR || `${root}/data/db`,
    databaseUrl:
      process.env.DATABASE_URL ||
      process.env.NUXT_DATABASE_URL ||
      `sqlite:///${root}/data/db/cropmerge.sqlite`,
    public: {
      appName: process.env.NUXT_PUBLIC_APP_NAME || 'CropMerge Field Triage',
      visionApiUrl: process.env.NUXT_PUBLIC_VISION_API_URL || 'http://127.0.0.1:8001',
      visionSmallUploadThresholdBytes: Number(
        process.env.NUXT_PUBLIC_VISION_SMALL_UPLOAD_THRESHOLD_BYTES || 24 * 1024 * 1024,
      ),
      // Empty by default = Live Drone mode disabled. Docker Compose only —
      // never set on the Vercel deployment (no persistent WHEP/MQTT there).
      mediamtxWhepUrl: process.env.NUXT_PUBLIC_MEDIAMTX_WHEP_URL || '',
    },
  },
  typescript: {
    strict: true,
    typeCheck: false,
  },
  app: {
    head: {
      title: 'CropMerge Field Triage',
      meta: [
        {
          name: 'description',
          content:
            'RGB drone video → field review → inspection areas for farmer review',
        },
      ],
    },
  },
})
