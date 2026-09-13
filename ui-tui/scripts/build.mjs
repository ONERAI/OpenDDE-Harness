#!/usr/bin/env node
// Bundles src/entry.ts into a single self-contained dist/entry.js.
// No runtime node_modules needed — the Python launcher runs it as
// `node dist/entry.js` from an installed wheel.
import { build } from 'esbuild'
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
// Two self-contained bundles: the TUI, and the pi-ai model service the Python
// side spawns (src/model-service/main.ts) — same build, same launch rule.
const entries = [
  ['src/entry.ts', 'dist/entry.js'],
  ['src/model-service/main.ts', 'dist/model-service.js']
]

// The compaction extension (`pi-codex-compact`) statically imports the pi
// coding agent for its local text-summary fallback, the one part of it we do
// not use. Left alone, that single import would pull the whole agent into the
// bundle, so it resolves to a stub that throws if anything ever calls it.
const alias = {
  '@earendil-works/pi-coding-agent': resolve(root, 'src/model-service/pi-coding-agent-stub.ts')
}

for (const [source, target] of entries) {
  const out = resolve(root, target)

  await build({
    entryPoints: [resolve(root, source)],
    alias,
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node22',
    outfile: out,
    // Some transitive deps use CommonJS `require(...)` at runtime (pi-tui probes
    // for optional native modifier helpers this way). ESM bundles don't get a
    // `require` binding automatically, so inject one.
    banner: {
      js: "import { createRequire as __cr } from 'node:module'; const require = __cr(import.meta.url);"
    },
    logLevel: 'info'
  })

  // Nix's patchShebangs phase mangles `/usr/bin/env -S node --foo`; the launcher
  // always invokes this file as `node dist/entry.js`, so a shebang is only a
  // liability. Strip whatever esbuild carried over.
  const body = readFileSync(out, 'utf8')
  if (body.startsWith('#!')) {
    writeFileSync(out, body.slice(body.indexOf('\n') + 1))
  }

  console.log(`built ${out}`)
}
