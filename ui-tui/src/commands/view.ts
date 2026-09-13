// The two commands about how this process draws itself: which screen it uses,
// and how loud a collapsed tool row is.
//
// The UI draws on the alternate screen by default, where it owns scrolling,
// search and selection. `/fullscreen off` hands the screen back to the
// terminal, whose own scrollback then holds the conversation. It writes nothing
// to the gateway or to configuration: it changes this process and this run, and
// `OPENDDE_HARNESS_TUI_FULLSCREEN=0` is what changes how the next one starts.
//
// `/quiet-tools` is the same kind of setting, and carries the name the owner's
// pi extension gave it. It changes the renderer and only the renderer: what a
// tool returned to the conversation was settled before the transcript drew a
// line of it.

import type { RendererMode } from '../renderer.js'
import type { Command } from './types.js'

import { wordItems } from './types.js'

const FULLSCREEN_WORDS = ['on', 'off', 'status'] as const

const QUIET_WORDS = ['on', 'off', 'toggle', 'status'] as const

export function viewCommands(): Command[] {
  return [
    {
      argumentHint: '[on|off|status]',
      description: 'alternate screen with its own scrollback and search (on by default), or the plain terminal',
      getArgumentCompletions: prefix => wordItems(FULLSCREEN_WORDS, prefix),
      name: 'fullscreen',
      run: (ctx, arg) => {
        const word = arg.trim().toLowerCase()

        if (word && !(FULLSCREEN_WORDS as readonly string[]).includes(word)) {
          return ctx.transcript.print(`usage: /fullscreen [${FULLSCREEN_WORDS.join('|')}]`, 'warn')
        }

        const renderer = ctx.renderer

        if (!renderer) {
          return ctx.transcript.print('this build draws on the main screen only', 'warn')
        }

        const current = renderer.mode()

        if (word === 'status') {
          return ctx.transcript.print(`fullscreen: ${current === 'fullscreen' ? 'on' : 'off'}`)
        }

        const wanted: RendererMode =
          word === 'on'
            ? 'fullscreen'
            : word === 'off'
              ? 'regular'
              : current === 'fullscreen'
                ? 'regular'
                : 'fullscreen'

        if (wanted === current) {
          return ctx.transcript.print(`fullscreen is already ${current === 'fullscreen' ? 'on' : 'off'}`)
        }

        const refused = renderer.switchTo(wanted)

        ctx.transcript.print(
          refused ?? `fullscreen: ${wanted === 'fullscreen' ? 'on' : 'off'}`,
          refused ? 'warn' : 'muted'
        )
      }
    },

    {
      argumentHint: '[on|off|toggle|status]',
      description: 'one-line collapsed tool rows with their output hidden (on by default)',
      getArgumentCompletions: prefix => wordItems(QUIET_WORDS, prefix),
      name: 'quiet-tools',
      run: (ctx, arg) => {
        const word = arg.trim().toLowerCase() || 'toggle'

        if (!(QUIET_WORDS as readonly string[]).includes(word)) {
          return ctx.transcript.print(`usage: /quiet-tools [${QUIET_WORDS.join('|')}]`, 'warn')
        }

        const current = ctx.quietTools.get()
        const key = ctx.keybindings.getKeys('app.tools.expand').join('/') || 'ctrl+o'

        if (word === 'status') {
          return ctx.transcript.print(
            current
              ? `quiet tools: on — a collapsed tool row is one line and hides its output; ${key} expands it`
              : `quiet tools: off — a collapsed tool row shows the first lines of what the tool returned; ${key} expands it`
          )
        }

        const wanted = word === 'toggle' ? !current : word === 'on'

        ctx.quietTools.set(wanted)
        // Said whether or not it moved: a command that answers nothing reads as
        // a command that failed. The model sees the same tool results either
        // way, which is the one thing worth saying twice.
        ctx.transcript.print(
          wanted
            ? `quiet tools: on — collapsed tool rows are one line, ${key} expands one. The model still gets every result in full.`
            : 'quiet tools: off — collapsed tool rows show the first lines of a result again.'
        )
      }
    }
  ]
}
