// Build-time stand-in for `@earendil-works/pi-coding-agent`.
//
// `pi-codex-compact/src/compaction.ts` imports these three functions for its
// local text-summary fallback -- the one part of the extension we do not use:
// summarizing is the Python loop's job, and this service only asks OpenAI for
// the opaque replacement history. The import is static, so a bundle that never
// calls them would still have to contain the whole pi coding agent. The build
// and the test runner alias the module here instead.
//
// Nothing in our code path reaches these. If something ever does, it says so
// rather than silently returning a wrong summary.

function unavailable(name: string): never {
  throw new Error(
    `pi-codex-compact called ${name}() from @earendil-works/pi-coding-agent, which the model service does not bundle: ` +
      'only its remote-compaction path is used.'
  )
}

export function compact(): never {
  unavailable('compact')
}

export function convertToLlm(): never {
  unavailable('convertToLlm')
}

export function serializeConversation(): never {
  unavailable('serializeConversation')
}
