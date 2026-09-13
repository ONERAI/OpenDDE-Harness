// Types for the two pi packages the compaction extension type-imports.
//
// `pi-codex-compact` is a pi extension: its sources type-import
// `@earendil-works/pi-coding-agent` and `@earendil-works/pi-agent-core`, the
// two halves of the pi *agent*. We use the extension as a library and ship no
// agent, so installing them would mean ~26 MB of dev dependency (and a wasm
// image renderer) for names alone. These declarations stand in instead.
//
// Only what `pi-codex-compact/src/compaction.ts` actually references is
// declared. The three pi-coding-agent *values* below belong to its local
// text-summary path, which we never call; `scripts/build.mjs` and
// `vitest.config.ts` alias the module to `./pi-coding-agent-stub.ts` so the
// bundle does not need the real package either.

declare module '@earendil-works/pi-agent-core' {
  import type { Message } from '@earendil-works/pi-ai'

  /**
   * pi-agent-core's message is pi-ai's plus agent bookkeeping. The extension
   * reads only the pi-ai half (role, content blocks, signatures, toolCallId),
   * and our caller passes pi-ai messages straight through, so this is the
   * shape that keeps both sides honest.
   */
  export type AgentMessage = Message
  /** Passed through to `compact()` untouched; never inspected. */
  export type ThinkingLevel = string
}

declare module '@earendil-works/pi-coding-agent' {
  export interface ToolInfo {
    description: string
    name: string
    parameters: unknown
  }

  export interface CompactionResult {
    details?: unknown
    firstKeptEntryId: string
    summary: string
    tokensBefore: number
  }

  /** Only `preparation` is referenced, and only as an opaque handle. */
  export interface SessionBeforeCompactEvent {
    preparation: unknown
  }

  export function compact(...args: unknown[]): Promise<CompactionResult>
  export function convertToLlm(messages: unknown): unknown
  export function serializeConversation(conversation: unknown): string
}
