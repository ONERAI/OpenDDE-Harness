// `/memory` — the AGENTS.md and ODH.md files this session is following.
//
// The list is asked for rather than read from the session's facts: files are
// re-read every turn, so one written a minute ago is already in force, and a
// view that showed the boot-time answer would say otherwise.
//
// `on` and `off` last as long as the session. Someone switching a file off is
// saying "not for this conversation", not changing a setting, so nothing is
// written to disk and the next launch starts with everything on.

import type { ProjectInstructionFile } from '../rpc/index.js'
import type { Command } from './types.js'

import { errorMessage } from './passthrough.js'
import { wordItems } from './types.js'

const VERBS = ['off', 'on'] as const

/** Bytes as someone would say them, since the point is "is this file big". */
function size(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`
  }

  const kb = bytes / 1024

  return kb < 10 ? `${kb.toFixed(1)} KB` : `${Math.round(kb)} KB`
}

/** What became of one file, in the fewest words that stay true.
 *
 *  "changed" is the one worth reading twice: these files are followed as the
 *  user's own instructions, so content that moved after this session started
 *  is either something they did, or something they should know about. */
function state(file: ProjectInstructionFile): string {
  const drift = file.changed ? ' · changed since this session started' : ''

  if (!file.enabled) {
    return `off${drift}`
  }

  if (file.skipped) {
    return `dropped, over the 128 KB total${drift}`
  }

  return (file.truncated ? 'on, first 32 KB only' : 'on') + drift
}

function table(files: ProjectInstructionFile[]): string {
  const width = Math.max(...files.map(file => file.display.length))
  const sizes = Math.max(...files.map(file => size(file.size).length))

  return files
    .map(file => `${file.display.padEnd(width)}  ${size(file.size).padStart(sizes)}  ${state(file)}`)
    .join('\n')
}

/** Said when the search found nothing, which is easy to read as a failure of
 *  the search. It is not: there is nothing to read, and here is where to put
 *  it. Neither line claims a repository, because outside a work tree only the
 *  launch directory was searched. */
function nothingFound(cwd: string): string {
  return [
    'No AGENTS.md or ODH.md found. This session is following no project',
    'instructions at all.',
    '',
    'Write one and the next message picks it up, with nothing to restart:',
    `  ${cwd}/AGENTS.md`,
    '      rules for the work in this directory',
    '  ~/.opendde_harness/AGENTS.md',
    '      rules for every project'
  ].join('\n')
}

export const memoryCommands: Command[] = [
  {
    argumentHint: '[on|off <path>]',
    description: 'the AGENTS.md / ODH.md files this session follows',
    getArgumentCompletions: prefix => (prefix.includes(' ') ? null : wordItems(VERBS, prefix)),
    name: 'memory',
    run: async (ctx, arg) => {
      const [verb = '', ...rest] = arg.trim().split(/\s+/)
      const action = VERBS.includes(verb.toLowerCase() as (typeof VERBS)[number]) ? verb.toLowerCase() : ''

      if (verb && !action) {
        return ctx.transcript.print(`/memory takes on or off, not ${verb}`, 'warn')
      }

      try {
        // Named, because on and off belong to this conversation and to no
        // other. Without it the gateway would answer about the defaults and
        // switch a file off for every session it serves.
        const session = ctx.session.id()
        const result = await ctx.gateway.sessionInstructions({
          ...(session ? { session_id: session } : {}),
          ...(action ? { action: action as 'off' | 'on', path: rest.join(' ') } : {})
        })

        if (result.error) {
          return ctx.transcript.print(result.error, 'warn')
        }

        if (result.changed) {
          ctx.transcript.print(`${result.changed} is ${action} for this session`)
        }

        if (!result.files.length) {
          return ctx.transcript.printBlock(nothingFound(result.cwd), 'Project instructions')
        }

        ctx.transcript.printBlock(
          `${table(result.files)}\n\nRead top to bottom, so the last file has the last word.\n` +
            '/memory off <path> drops one for this session; /memory on <path> puts it back.',
          'Project instructions'
        )
      } catch (err) {
        ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
      }
    }
  }
]
