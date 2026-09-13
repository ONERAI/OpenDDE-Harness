// Turning a typed line into a command run.
//
// A local command wins; anything else goes to the gateway's `slash.exec`, whose
// output is printed into the transcript as a system block. There is no pager:
// the terminal's own scrollback is the transcript's scrollback.

import type { CommandRegistry } from './registry.js'
import type { CommandContext } from './types.js'

import { errorMessage, runOnGateway } from './passthrough.js'
import { looksLikeCommand, parseCommand } from './types.js'

/**
 * Commands that were removed, and where what they did now lives.
 *
 * Muscle memory outlives a release: someone who typed `/yolo` for a month
 * types it again, and sending that to the gateway answers "unknown command"
 * about a thing this app used to do. One line each, naming the key or the
 * place, and no more than that — these are not aliases and they do not run
 * anything.
 *
 * DELETE THIS TABLE one release after the commands went. It is a note for
 * people who knew the old set, not a permanent second name for everything.
 *
 * A Map, not an object literal: an object answers for every name on
 * `Object.prototype` too, so `/constructor` found a function here and sent it
 * to a renderer that expects a string. What a user types is never a safe key
 * for a plain object.
 */
const RETIRED = new Map<string, string>([
  ['branch', '/branch is now /fork'],
  ['busy', 'Enter queues while a turn runs; Ctrl+K sends the first queued message now'],
  ['context', 'the context window comes from the model; set it in `ddeharness provider set`'],
  ['details', 'Ctrl+T shows or hides thinking, Ctrl+O expands or collapses tool output'],
  ['history', 'the conversation is the transcript — scroll it'],
  ['reload-mcp', '/reload-mcp is now /mcp:reload'],
  ['setup', 'run `ddeharness onboard` in a terminal'],
  ['task', '/task <id> is now /tasks <id>, with /task:logs and /task:follow'],
  ['token-usage', 'the footer counters are a setting: `ddeharness config set tui.show_token_usage`'],
  ['yolo', '/yolo is Shift+Tab']
])

/**
 * CLI groups that are not commands: `provider` alone is a group head, and the
 * dispatcher refuses it. They left the popup with the rest of the
 * administration surface, so say where they went rather than let the gateway
 * answer "unknown command" about something that exists.
 *
 * Only the bare head. `/provider list` is a real command, still dispatchable,
 * and typing it in full still runs it — it is the popup it left, not the app.
 */
const GROUP_HEAD = /^(provider|skill|compute|protein-design)$/

/**
 * Run `text` as a command. Returns false when it is not one, so the caller can
 * send it as a prompt instead.
 */
export async function dispatch(registry: CommandRegistry, ctx: CommandContext, text: string): Promise<boolean> {
  if (!looksLikeCommand(text)) {
    return false
  }

  const { arg, name } = parseCommand(text)
  const command = registry.find(name)

  // One boundary around every branch, not only around a command that exists.
  // A name nobody registered is the one most likely to be strange, and until
  // this covered them a throw on that path escaped into the app.
  try {
    if (command) {
      await command.run(ctx, arg)

      return true
    }

    const retired = RETIRED.get(name)

    if (retired) {
      ctx.transcript.print(retired)

      return true
    }

    if (!arg.trim() && GROUP_HEAD.test(name)) {
      ctx.transcript.print(`${name} settings live in \`ddeharness ${name}\` — /model covers the live one`)

      return true
    }

    await runOnGateway(ctx, text.trim().slice(1), name ? name[0]!.toUpperCase() + name.slice(1) : 'Command')
  } catch (err) {
    ctx.transcript.print(`/${name}: ${errorMessage(err)}`, 'error')
  }

  return true
}
