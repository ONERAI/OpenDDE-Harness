// How hard the model thinks, for the model this session is on.
//
// `model.overlay` stores the effort against the session's *current model*, not
// against the session and not as a global default. The label says so, and the
// response is trusted over the request: if another client moved the session to
// a different model while this was open, the overlay landed on that model and
// the note names it.

import type { Selector, SelectorLease } from '../components/selector.js'
import type { SelectorDeps } from './deps.js'

import { SelectorList, standardHints } from '../components/selectorList.js'
import { selectorError } from './deps.js'

/** The seven levels the gateway accepts, plus clearing the override. */
export const THINKING_ROWS: { description: string; id: string }[] = [
  { description: 'clear the override and follow the model default', id: 'default' },
  { description: 'no reasoning', id: 'off' },
  { description: 'the shortest reasoning the model offers', id: 'minimal' },
  { description: 'light reasoning', id: 'low' },
  { description: 'moderate reasoning', id: 'medium' },
  { description: 'deep reasoning', id: 'high' },
  { description: 'deeper still, where the model has the tier', id: 'xhigh' },
  { description: 'the most the model offers', id: 'max' }
]

export interface ThinkingSelectorOptions {
  deps: SelectorDeps
  lease: SelectorLease
  /** The applied value and the model it landed on, as the server reported them. */
  onApplied: (effort: string | undefined, model: string) => void
}

export function createThinkingSelector({ deps, lease, onApplied }: ThinkingSelectorOptions): Selector {
  const session = deps.session()
  const sessionId = session.id
  let model = session.model
  let sequence = 0
  let pending = false

  const list = new SelectorList({
    emptyText: 'no levels',
    keybindings: deps.keybindings,
    onCancel: () => lease.done('cancel'),
    onSelect: id => void apply(id),
    rows: THINKING_ROWS.map(row => ({
      description: row.description,
      id: row.id,
      kind: 'item' as const,
      label: `${row.id === (session.effort ?? 'default') ? '✓ ' : '  '}${row.id}`,
      searchText: `${row.id} ${row.description}`
    })),
    theme: deps.theme,
    title: title(),
    tui: deps.tui
  })

  list.setHints(standardHints(deps.theme, deps.keybindings))

  function title(): string {
    return `Thinking — ${model ?? 'current model'} · saved for this model`
  }

  // The current model is what the label is about, so fetch it once when the
  // session never reported one.
  if (!model) {
    const mine = ++sequence

    void deps.gateway
      .modelOptions({ include_catalog: false, session_id: sessionId })
      .then(result => {
        if (!lease.isCurrent() || mine !== sequence) {
          return
        }

        model = result.model
        list.setTitle(title())
        deps.tui.requestRender()
      })
      .catch(() => {
        // The label falls back to "current model"; the overlay still works.
      })
  }

  async function apply(level: string): Promise<void> {
    if (pending) {
      // Local edits to one session's model are serialized: the gateway has no
      // expected-model condition, so two overlapping overlays could land on
      // different models without either caller noticing.
      return
    }

    pending = true
    list.setStatus(`saving ${level}…`)
    list.setError(undefined)
    deps.tui.requestRender()

    const capturedModel = model

    try {
      const result = await deps.gateway.modelOverlay('reasoning_effort', level, sessionId)

      if (!lease.isCurrent()) {
        // The mutation was sent and stands; only the UI is gone.
        return
      }

      const effort = typeof result.value === 'string' ? result.value : undefined

      if (capturedModel && result.model && result.model !== capturedModel) {
        deps.transcript.print(`thinking applies to ${result.model}, which this session is on now`, 'warn')
      }

      onApplied(effort, result.model)
      deps.transcript.print(`thinking: ${effort ?? 'default'} for ${result.model}`)
      lease.done('done')
    } catch (err) {
      if (!lease.isCurrent()) {
        return
      }

      list.setStatus(undefined)
      list.setError(selectorError(err))
      deps.tui.requestRender()
    } finally {
      pending = false
    }
  }

  return {
    component: list,
    dispose: () => {
      // Nothing to release: the list holds no timers and no sockets.
    },
    focus: list
  }
}
