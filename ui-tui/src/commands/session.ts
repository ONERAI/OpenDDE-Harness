// Commands that change the conversation or the model behind it.
//
// A note on `config.set`: the gateway's v0.1 whitelist is four keys plus the
// special `model` (`opendde_harness/tui_rpc/methods/config.py`). `yolo`, `fast`,
// `busy` and `verbose` are not on it yet, so those commands do the part that is
// local, ask the gateway for the rest, and say plainly when it declines.

import type { Command, CommandContext } from './types.js'

import { ConfigValidationError, ModelNotAvailableError } from '../rpc/index.js'
import { configErrorMessage, errorMessage, requireSession } from './passthrough.js'
import { wordItems } from './types.js'

const THINKING_LEVELS = ['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'default'] as const
const FAST_WORDS = ['normal', 'fast', 'status', 'on', 'off', 'toggle'] as const

/** The wire contract is `<channel>:<chat_id>`; users type the bare id they see. */
function normalizeSessionId(id: string): string {
  return id.includes(':') ? id : `tui:${id}`
}

function bareSessionId(key: string): string {
  return key.includes(':') ? key.slice(key.indexOf(':') + 1) : key
}

/** True when the session cannot be replaced right now, in which case the
 *  command says which of the several reasons it is. */
function switchBlocked(ctx: CommandContext, what: string): boolean {
  if (!ctx.session.busy()) {
    return false
  }

  const reason = ctx.session.blockedReason() ?? 'something else is using this session'

  ctx.transcript.print(`cannot ${what} yet: ${reason}`, 'warn')

  return true
}

/** Switch to an exact model reference. False when the gateway knew no such
 *  model, which is the caller's cue to open the selector with the text -- pi's
 *  own rule for `/model <text>`. */
async function setModel(ctx: CommandContext, rest: string, asDefault: boolean): Promise<boolean> {
  // The model id carries its provider ("openai-codex/gpt-5.6-luna"), and the
  // server refuses one that does not. There is no --provider flag to parse: it
  // used to name the provider separately, which overrode what the id said.
  const value = rest.trim()

  try {
    const result = await ctx.gateway.configSet({
      key: 'model',
      scope: asDefault ? 'default' : 'session',
      session_id: ctx.session.id(),
      value
    })

    if (!result.value) {
      ctx.transcript.print('error: the model switch returned no model', 'error')

      return true
    }

    if (!result.applied) {
      // Nothing was built, so nothing was validated — reporting a switch here
      // would leave the footer on a model no turn will use.
      ctx.transcript.print(`error: model switch was not applied: ${result.value}`, 'error')

      return true
    }

    // pi's own two sentences.
    ctx.transcript.print(asDefault ? `Default model: ${result.value}` : `Model: ${result.value}`)

    if (result.applies_to_session !== false) {
      ctx.session.patchInfo({ model: result.value })
    }

    return true
  } catch (err) {
    if (err instanceof ModelNotAvailableError || err instanceof ConfigValidationError) {
      return false
    }

    ctx.transcript.print(configErrorMessage(err, 'model'), 'error')

    return true
  }
}

export const sessionCommands: Command[] = [
  {
    argumentHint: '[title]',
    description: 'start a new session',
    name: 'new',
    run: async (ctx, arg) => {
      if (switchBlocked(ctx, 'switch sessions')) {
        return
      }

      await ctx.session.create(arg.trim() || undefined)
    }
  },

  {
    argumentHint: '[id]',
    description: 'resume a prior session',
    name: 'resume',
    run: async (ctx, arg) => {
      if (switchBlocked(ctx, 'switch sessions')) {
        return
      }

      const id = arg.trim()

      if (!id) {
        return ctx.session.pick()
      }

      await ctx.session.resume(normalizeSessionId(id))
    }
  },

  {
    description: 'list the saved sessions',
    name: 'sessions',
    run: async ctx => {
      // The list, not a picker: `/resume` opens the interactive switcher, and
      // this one puts the sessions in the transcript where they can be read,
      // scrolled and copied. The verbs that used to hang off it live at the
      // top level now -- `/new`, `/resume`, `/fork`, `/export` -- and deleting
      // is Ctrl+D inside the picker.
      try {
        const { sessions } = await ctx.gateway.sessionList()

        if (!sessions.length) {
          return ctx.transcript.print('no saved sessions')
        }

        const body = sessions
          .map(
            item =>
              `${bareSessionId(item.id).padEnd(12)} ${String(item.message_count).padStart(4)} msg  ${item.title || item.preview}`
          )
          .join('\n')

        return ctx.transcript.printBlock(body, 'Sessions')
      } catch (err) {
        return ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
      }
    }
  },

  {
    argumentHint: '[title]',
    description: 'show or set this session title',
    name: 'title',
    run: async (ctx, arg) => {
      const id = requireSession(ctx, '/title')

      if (!id) {
        return
      }

      const title = arg.trim()

      try {
        const result = await ctx.gateway.sessionTitle(id, title || undefined)

        if (!title) {
          const current = (result.title ?? '').trim()

          return ctx.transcript.print(current ? `title: ${current}` : 'no title set')
        }

        const next = (result.title ?? title).trim()

        ctx.session.patchInfo({ title: next })
        ctx.transcript.print(
          `session title set: ${next}${result.pending ? ' (queued while the session initializes)' : ''}`
        )
      } catch (err) {
        ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
      }
    }
  },

  {
    argumentHint: '[name]',
    description: 'fork this session into a new one',
    // `fork` is what the CLI and pi both call it. One concept, one name.
    name: 'fork',
    run: async (ctx, arg) => {
      // Forking replaces the live session, so it is a switch and waits like
      // one. Without this the gateway ended up running the parent and the
      // child at once, with the UI subscribed only to the child.
      if (switchBlocked(ctx, 'fork the session')) {
        return
      }

      const previous = ctx.session.id()

      try {
        const result = await ctx.gateway.sessionBranch(previous, arg.trim())

        if (!result.session_id) {
          return ctx.transcript.print('nothing to fork', 'warn')
        }

        // The fork is a full copy of what is on screen, so the transcript
        // stays. Adoption closes the parent itself — closing it here as well
        // sent `session.close` for the same id twice — and it is awaited, so
        // the lines below describe a session the UI is already on.
        await ctx.session.adopt(result.session_id, result.title ?? undefined)

        const count = result.message_count ?? 0

        ctx.transcript.print(
          `⑂ forked "${result.title || '(untitled)'}" · ${count} message${count === 1 ? '' : 's'} carried`
        )
        ctx.transcript.print(`   parent  ${previous ? bareSessionId(previous) : '(none)'}`)
        ctx.transcript.print(`   forked  ${bareSessionId(result.session_id)}`)
      } catch (err) {
        ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
      }
    }
  },

  {
    argumentHint: '[id]',
    description: 'write this session to a markdown file',
    name: 'export',
    run: async (ctx, arg) => {
      // The raw id, so the backend can resolve it cross-channel: forcing `tui:`
      // onto a bare id breaks that resolution.
      const target = arg.trim() || ctx.session.id()

      if (!target) {
        return ctx.transcript.print('no active session to export', 'warn')
      }

      try {
        const result = await ctx.gateway.sessionExport(target)

        if (result.exported && result.path) {
          return ctx.transcript.print(`exported to ${result.path}`, 'ok')
        }

        if (result.reason === 'ambiguous') {
          return ctx.transcript.print(
            `ambiguous session id — candidates: ${(result.candidates ?? []).join(', ')}`,
            'warn'
          )
        }

        if (result.reason === 'write_failed') {
          return ctx.transcript.print('error: could not write the export file', 'error')
        }

        ctx.transcript.print(`no such session: ${target}`, 'warn')
      } catch (err) {
        ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
      }
    }
  },

  {
    argumentHint: '[provider/model] [--default]',
    description: 'change the model; --default moves what new sessions start on',
    name: 'model',
    run: async (ctx, arg) => {
      // No busy guard: the server binds the model at turn entry, so a switch
      // asked for mid-answer lands on the next turn instead of being refused.
      const raw = arg.trim()
      const asDefault = /(^|\s)--default(\s|$)/.test(raw)
      const rest = raw.replace(/(^|\s)--default(?=\s|$)/g, '').trim()

      if (!rest) {
        return ctx.openModelPicker(asDefault ? 'default' : 'session')
      }

      // pi's own rule: an exact model reference switches at once; anything
      // else opens the selector with the text already in its search.
      if (!(await setModel(ctx, rest, asDefault))) {
        ctx.openModelPicker(asDefault ? 'default' : 'session', rest)
      }
    }
  },

  {
    description: 'choose which models /model shows under its scoped tab',
    name: 'scoped-models',
    run: ctx => ctx.openScopedModelsPicker()
  },

  {
    argumentHint: '[level]',
    description: "set the model's thinking level",
    getArgumentCompletions: prefix => wordItems(THINKING_LEVELS, prefix),
    name: 'thinking',
    run: async (ctx, arg) => {
      const level = arg.trim().toLowerCase()

      if (!level) {
        // Bare `/thinking` opens the picker, the way bare `/model` does.
        return ctx.openThinkingPicker()
      }

      try {
        const result = await ctx.gateway.modelOverlay('reasoning_effort', level, ctx.session.id())
        const effort = typeof result.value === 'string' ? result.value : undefined

        ctx.session.patchInfo({ effort })
        ctx.transcript.print(`thinking: ${effort ?? 'default'} for ${result.model}`)
      } catch (err) {
        ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
      }
    }
  },

  {
    argumentHint: '[normal|fast|status]',
    description: 'switch the model between normal and fast service',
    getArgumentCompletions: prefix => wordItems(FAST_WORDS, prefix),
    name: 'fast',
    run: async (ctx, arg) => {
      const mode = arg.trim().toLowerCase()

      if (mode && !(FAST_WORDS as readonly string[]).includes(mode)) {
        return ctx.transcript.print(`usage: /fast [${FAST_WORDS.join('|')}]`, 'warn')
      }

      if (!mode || mode === 'status') {
        return ctx.transcript.print(`fast mode: ${ctx.session.info().fast ? 'fast' : 'normal'}`)
      }

      try {
        const result = await ctx.gateway.configSet({ key: 'fast', session_id: ctx.session.id(), value: mode })
        const next = result.value === 'fast' ? 'fast' : 'normal'

        ctx.session.patchInfo({ fast: next === 'fast' })
        ctx.transcript.print(`fast mode: ${next}`)
      } catch (err) {
        ctx.transcript.print(configErrorMessage(err, 'fast'), 'warn')
      }
    }
  },

  {
    argumentHint: '[cycle|on|off]',
    description: 'cycle how much tool output the agent reports',
    getArgumentCompletions: prefix => wordItems(['cycle', 'on', 'off'], prefix),
    name: 'verbose',
    run: async (ctx, arg) => {
      try {
        const result = await ctx.gateway.configSet({
          key: 'verbose',
          session_id: ctx.session.id(),
          value: arg.trim() || 'cycle'
        })

        ctx.transcript.print(`verbose: ${result.value ?? 'changed'}`)
      } catch (err) {
        ctx.transcript.print(configErrorMessage(err, 'verbose'), 'warn')
      }
    }
  }
]

/** Shared by the Shift+Tab binding, which is the only way to reach it: a
 *  toggle with a key does not also need a command. */
export async function toggleYolo(ctx: CommandContext): Promise<void> {
  const id = requireSession(ctx, 'yolo')

  if (!id) {
    return
  }

  try {
    const result = await ctx.gateway.configSet({ key: 'yolo', session_id: id, value: 'toggle' })

    ctx.transcript.print(`yolo ${result.value === '1' || result.value === 'on' ? 'on' : 'off'}`)
  } catch (err) {
    ctx.transcript.print(configErrorMessage(err, 'yolo'), 'warn')
  }
}
