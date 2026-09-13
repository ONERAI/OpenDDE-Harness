import { defineConfig } from 'vitest/config'

export default defineConfig({
  resolve: {
    // Same stub the bundle uses: the compaction extension's local-summary
    // import would otherwise need the whole pi coding agent installed.
    alias: {
      '@earendil-works/pi-coding-agent': new URL('src/model-service/pi-coding-agent-stub.ts', import.meta.url).pathname
    }
  },
  test: {
    include: ['src/__tests__/**/*.test.ts'],
    environment: 'node',
    // The extension is TypeScript source, so vite has to transform it rather
    // than hand it to node as a dependency.
    server: { deps: { inline: ['pi-codex-compact'] } }
  }
})
