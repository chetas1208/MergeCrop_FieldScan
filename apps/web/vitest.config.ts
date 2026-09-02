import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['composables/**/*.spec.ts'],
    environment: 'node',
  },
})
