// Commands about this UI and this conversation: help, exit, status, what the
// transcript shows, and the clipboard.

import type { Keybinding } from '@earendil-works/pi-tui'

import type { Command } from './types.js'

import { writeClipboardText, writeOsc52Clipboard } from '../lib/clipboard.js'
import { errorMessage, requireSession, runOnGateway } from './passthrough.js'

/** Per-message preview length in /history. */
const HISTORY_PREVIEW_CHARS = 400

const TOKEN_USAGE_WORDS = ['on', 'off', 'status'] as const

/** The keys /help lists, in the order it lists them. */
export const HOTKEY_ORDER: Keybinding[] = [
  'app.interrupt',
  'app.reset',
  'app.exit',
  'app.thinking.toggle',
  'app.tools.expand',
  'app.yolo.toggle',
  'app.queue.submit',
  'app.queue.edit.older',
  'app.queue.drop',
  'app.paste',
  'app.editor.external',
  'app.redraw'
]

export function coreCommands(): Command[] {
  return [
    {
      description: 'list the commands and hotkeys',
      name: 'help',
      // The panel is built by the app, which knows the registry, the catalog
      // and the live keybindings; it is appended to the transcript rather than
      // taking the editor slot.
      run: ctx => ctx.showHelp()
    },

    {
      description: 'exit OpenDDE Harness',
      name: 'quit',
      run: ctx => ctx.quit()
    },

    {
      description: 'show live session info',
      name: 'status',
      run: async ctx => {
        const id = requireSession(ctx, '/status')

        if (!id) {
          return
        }

        try {
          const result = await ctx.gateway.sessionStatus(id)

          ctx.transcript.printBlock(result.output?.trim() || '(no status)', 'Status')
        } catch (err) {
          ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
        }
      }
    },

    {
      argumentHint: '[number]',
      description: 'copy a reply to the clipboard',
      name: 'copy',
      run: async (ctx, arg) => {
        const replies = ctx.transcript.history().filter(entry => entry.role === 'assistant')

        if (arg && !/^\d+$/.test(arg.trim())) {
          return ctx.transcript.print('usage: /copy [number]', 'warn')
        }

        const target = arg ? replies[Math.min(Number(arg.trim()), replies.length) - 1] : replies.at(-1)

        if (!target) {
          return ctx.transcript.print('nothing to copy — start a conversation first')
        }

        try {
          if (await writeClipboardText(target.text)) {
            return ctx.transcript.print(`copied ${target.text.length} characters`)
          }

          writeOsc52Clipboard(target.text)
          ctx.transcript.print('sent an OSC 52 copy sequence (your terminal has to support it)')
        } catch (err) {
          ctx.transcript.print(`copy failed: ${errorMessage(err)}`, 'error')
        }
      }
    },

    {
      description: 'undo the last exchange',
      name: 'undo',
      run: async ctx => {
        const id = requireSession(ctx, '/undo')

        if (!id) {
          return
        }

        try {
          const result = await ctx.gateway.sessionUndo(id)

          if ((result.removed ?? 0) <= 0) {
            return ctx.transcript.print('nothing to undo')
          }

          // Removed first, then reported: dropping an exchange takes
          // everything shown after it, which would include this note.
          const dropped = ctx.transcript.dropLastExchange()

          // The window is emptier than the last turn reported, and only the
          // gateway can say by how much -- or that nothing can. Null is that
          // answer, and it has to pass through: dropping it left the footer
          // showing the removed exchange's percentage as though it were still
          // measured. Absent is the only value that means "nothing changed".
          if (result.context_used !== undefined) {
            ctx.session.patchInfo({ contextTokens: result.context_used })
          }

          ctx.transcript.print(`undid ${result.removed} messages`)

          // The exchange the gateway removed was its last *saved* one, which
          // is not always the last one on screen. Say so rather than leave the
          // transcript quietly disagreeing with the session.
          if (!dropped) {
            ctx.transcript.print('the removed exchange was not one shown here', 'warn')
          }
        } catch (err) {
          ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
        }
      }
    },

    {
      description: 'send the last prompt again',
      name: 'retry',
      run: async ctx => {
        // The prompt, not the last user entry still on screen: after `/undo`
        // the prompt to resend is the one that was just taken off screen, and
        // reading the transcript would pick the exchange before it and then
        // undo that one too.
        const candidate = ctx.transcript.retryCandidate()

        if (!candidate) {
          return ctx.transcript.print('nothing to retry')
        }

        const id = ctx.session.id()

        // Nothing of it is stored — it was undone, or its turn failed before
        // it saved anything — so it only has to go out again.
        if (!id || !candidate.onServer) {
          return ctx.session.send(candidate.text)
        }

        try {
          const result = await ctx.gateway.sessionUndo(id)

          if ((result.removed ?? 0) <= 0) {
            return ctx.transcript.print('nothing to retry')
          }

          ctx.transcript.dropLastExchange()

          if (result.context_used !== undefined) {
            ctx.session.patchInfo({ contextTokens: result.context_used })
          }

          ctx.session.send(candidate.text)
        } catch (err) {
          ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
        }
      }
    },

    {
      argumentHint: '[--fix]',
      description: 'check configuration and compute resources',
      name: 'doctor',
      run: (ctx, arg) => runOnGateway(ctx, arg ? `doctor ${arg}` : 'doctor', 'Doctor')
    },

    {
      argumentHint: '[stop]',
      description: 'open the tracing dashboard (LLM, tool and memory spans); `stop` shuts it down',
      name: 'tracing',
      run: (ctx, arg) => runOnGateway(ctx, arg ? `tracing ${arg}` : 'tracing', 'Tracing')
    }
  ]
}
