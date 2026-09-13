// Running a command on the gateway instead of here.
//
// `slash.exec` hands the text after the slash to the CLI dispatcher, which is
// where every command this UI does not implement itself ends up — including the
// ones below, which exist locally only so they appear in the completion popup
// with a description.

import type { Command, CommandContext } from './types.js'

import { ConfigFieldReadonlyError } from '../rpc/index.js'

/** The short one-liner / long block threshold the old TUI used. */
const BLOCK_CHARS = 180
const BLOCK_LINES = 2

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

/**
 * What to say when a `config.set` fails.
 *
 * The gateway's write whitelist is the `agent.*` and `tui.*` preference keys
 * plus `model`
 * (`opendde_harness/tui_rpc/methods/config.py`); everything else comes back as
 * -32010. Say that plainly rather than showing the raw error.
 */
export function configErrorMessage(err: unknown, key: string): string {
  if (err instanceof ConfigFieldReadonlyError) {
    return `the gateway does not accept a ${key} change yet`
  }

  return `error: ${errorMessage(err)}`
}

/** The session id, or null with a note in the transcript. */
export function requireSession(ctx: CommandContext, what: string): null | string {
  const id = ctx.session.id()

  if (!id) {
    ctx.transcript.print(`no active session — ${what} needs one`, 'warn')
  }

  return id
}

/**
 * Run `command` (no leading slash) on the gateway and print what comes back.
 * Long output becomes a titled block; a line or two stays a plain note.
 */
export async function runOnGateway(ctx: CommandContext, command: string, title: string): Promise<void> {
  try {
    const result = await ctx.gateway.slashExec(command, ctx.session.id())
    const body = result.output?.trim() || `/${command.split(/\s+/)[0]}: no output`
    const text = result.warning ? `warning: ${result.warning}\n${body}` : body

    if (text.length > BLOCK_CHARS || text.split('\n').filter(Boolean).length > BLOCK_LINES) {
      ctx.transcript.printBlock(text, title)
    } else {
      ctx.transcript.print(text)
    }
  } catch (err) {
    ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
  }
}

/** A command whose whole implementation is "ask the gateway". */
export function passthrough(
  name: string,
  description: string,
  options: { argumentHint?: string; title?: string } = {}
): Command {
  return {
    description,
    name,
    ...(options.argumentHint ? { argumentHint: options.argumentHint } : {}),
    run: (ctx, arg) =>
      runOnGateway(ctx, arg ? `${name} ${arg}` : name, options.title ?? name[0]!.toUpperCase() + name.slice(1))
  }
}
