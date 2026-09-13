// Past conversations: open one, or delete one.
//
// The backend works by session id, so none of pi's local-file session tree,
// trash or rename behaviour is ported — only its interaction shape: search as
// you type, Ctrl+D to start a delete, Enter to confirm it, Esc to back out.
//
// Two results are easy to misread and are checked rather than assumed:
// `session.resume` of an unknown id mints a *new empty* session instead of
// failing, and `session.delete` answers `{deleted: null}` when it removed
// nothing. Neither is reported here as if it had worked.

import type { Selector, SelectorLease } from '../components/selector.js'
import type { StageBody } from '../components/stage.js'
import type { SessionListItem } from '../rpc/index.js'
import type { SelectorDeps } from './deps.js'

import { SelectorList, standardHints } from '../components/selectorList.js'
import { StageView } from '../components/stage.js'
import { keyHint } from '../lib/keybindings.js'
import { ConfirmPrompt } from '../prompts/confirmPrompt.js'
import { selectorError } from './deps.js'

const LIST_LIMIT = 200
const LOAD_ALL = 'action:load-all'

export interface SessionSelectorActions {
  /** Why a switch cannot happen right now, or null when it can. */
  blockedReason(): null | string
  currentId(): null | string
  /** Delete a session the app is currently attached to: detach, close, delete,
   *  then adopt a fresh one. Returns whether the record was removed. */
  removeCurrent(id: string): Promise<boolean>
  /** Adopt `id`, but only while `stillCurrent()` holds. */
  resume(id: string, stillCurrent: () => boolean): Promise<void>
}

export interface SessionSelectorOptions {
  actions: SessionSelectorActions
  deps: SelectorDeps
  lease: SelectorLease
  /** Injected so the age column is deterministic in tests. */
  now?: () => number
}

/** `started_at` is Unix **seconds**. */
export function formatAge(startedAtSeconds: number, nowMs: number): string {
  if (!startedAtSeconds) {
    return '—'
  }

  const minutes = Math.max(0, Math.floor((nowMs - startedAtSeconds * 1000) / 60_000))

  if (minutes < 60) {
    return `${minutes}m`
  }

  if (minutes < 60 * 24) {
    return `${Math.floor(minutes / 60)}h`
  }

  return `${Math.floor(minutes / (60 * 24))}d`
}

export function sessionLabel(item: SessionListItem): string {
  return item.title?.trim() || item.preview?.trim() || '(untitled)'
}

export function createSessionSelector({ actions, deps, lease, now = Date.now }: SessionSelectorOptions): Selector {
  const stage = new StageView()
  const { theme } = deps
  let sessions: SessionListItem[] = []
  let sequence = 0
  let loadAll = false

  const list = new SelectorList({
    emptyText: 'no saved sessions',
    keybindings: deps.keybindings,
    onCancel: () => lease.done('cancel'),
    onSelect: id => void choose(id),
    placeholder: 'search title or id',
    rows: [],
    theme,
    title: 'Sessions',
    tui: deps.tui
  })

  list.setHints(
    standardHints(theme, deps.keybindings, [keyHint(deps.keybindings, theme, 'app.session.delete', 'delete')])
  )
  list.setStatus('loading…')

  // Only this stage reads Ctrl+D. Everything the list does not claim is search
  // text, which is why `d` here is a letter and not a delete key.
  const listBody: StageBody = {
    get focused() {
      return list.focused
    },
    set focused(value: boolean) {
      list.focused = value
    },
    handleInput(data: string) {
      if (deps.keybindings.matches(data, 'app.session.delete')) {
        startDelete()

        return
      }

      list.handleInput(data)
    },
    invalidate: () => list.invalidate(),
    render: (width: number) => list.render(width)
  }

  stage.set(listBody)

  function refresh(): void {
    const mine = ++sequence

    void deps.gateway
      .sessionList(loadAll ? undefined : LIST_LIMIT)
      .then(result => {
        if (!lease.isCurrent() || mine !== sequence) {
          return
        }

        sessions = result.sessions ?? []
        paint()
      })
      .catch((err: unknown) => {
        if (!lease.isCurrent() || mine !== sequence) {
          return
        }

        list.setStatus(undefined)
        list.setError(selectorError(err))
        deps.tui.requestRender()
      })
  }

  function paint(): void {
    const current = actions.currentId()
    const at = now()

    list.setRows([
      ...sessions.map(item => ({
        description: `${item.id}  ·  ${item.message_count} msg  ·  ${formatAge(item.started_at, at)}`,
        id: item.id,
        kind: 'item' as const,
        label: `${item.id === current ? '✓ ' : '  '}${sessionLabel(item)}`,
        searchText: `${sessionLabel(item)} ${item.id} ${item.preview ?? ''}`
      })),
      ...(loadAll || sessions.length < LIST_LIMIT
        ? []
        : [
            {
              description: 'the list above is the most recent 200',
              id: LOAD_ALL,
              kind: 'action' as const,
              label: 'Load all sessions',
              searchText: 'load all sessions'
            }
          ])
    ])
    list.setStatus(`${sessions.length} session${sessions.length === 1 ? '' : 's'}`)
    list.setError(undefined)
    deps.tui.requestRender()
  }

  async function choose(id: string): Promise<void> {
    if (id === LOAD_ALL) {
      loadAll = true
      list.setStatus('loading…')
      deps.tui.requestRender()
      refresh()

      return
    }

    const blocked = actions.blockedReason()

    if (blocked) {
      list.setError(`${blocked} — Esc closes this picker`)
      deps.tui.requestRender()

      return
    }

    if (id === actions.currentId()) {
      lease.done('done')

      return
    }

    list.setStatus('opening…')
    list.setError(undefined)
    deps.tui.requestRender()

    try {
      // The app owns the switch: it detaches the turn controller, closes the
      // old session and renders the new transcript once. `stillCurrent` is
      // checked at the commit point, so a late answer cannot replace a session
      // opened after this one was asked for.
      await actions.resume(id, () => lease.isCurrent())
    } catch (err) {
      if (!lease.isCurrent()) {
        return
      }

      list.setStatus(undefined)
      list.setError(selectorError(err))
      deps.tui.requestRender()
    }
  }

  function startDelete(): void {
    const id = list.selectedId

    if (!id || id === LOAD_ALL) {
      return
    }

    // Captured now, independently of the list index: the list may be refiltered
    // or reloaded before the answer arrives.
    const item = sessions.find(entry => entry.id === id)
    const name = item ? sessionLabel(item) : id
    const isCurrent = id === actions.currentId()
    const blocked = isCurrent ? actions.blockedReason() : null

    if (blocked) {
      list.setError(`${blocked} — finish or cancel it before deleting this session`)
      deps.tui.requestRender()

      return
    }

    stage.set(
      new ConfirmPrompt({
        confirmLabel: `Delete ${name}`,
        defaultAnswer: false,
        denyLabel: 'Keep it',
        keybindings: deps.keybindings,
        mode: 'local',
        onAnswer: value => {
          stage.set(listBody)
          deps.tui.requestRender()

          if (value) {
            void remove(id, isCurrent, name)
          }
        },
        prompt: isCurrent
          ? `${id}\n\nThis is the session on screen. It is closed here first, then deleted, and a new one is opened. The gateway does not itself refuse to delete a session another client is running.`
          : `${id}\n\nDeleting removes the stored conversation. The gateway does not itself refuse to delete a session another client is running.`,
        theme,
        title: 'Delete session'
      })
    )
    deps.tui.requestRender()
  }

  async function remove(id: string, isCurrent: boolean, name: string): Promise<void> {
    list.setStatus('deleting…')
    deps.tui.requestRender()

    try {
      if (isCurrent) {
        const removed = await actions.removeCurrent(id)

        // The transcript belongs to the app, not to this picker: adopting the
        // replacement session already closed the slot, and the outcome of a
        // delete the user asked for still has to be said.
        deps.transcript.print(removed ? `deleted ${name}` : `nothing was deleted for ${id}`, removed ? 'muted' : 'warn')

        if (lease.isCurrent()) {
          lease.done('done')
        }

        return
      }

      const result = await deps.gateway.sessionDelete(id)

      if (!lease.isCurrent()) {
        return
      }

      if (result.deleted !== id) {
        // `deleted: null` means nothing was removed, which is not success.
        list.setStatus(undefined)
        list.setError(`nothing was deleted for ${id}`)
        deps.tui.requestRender()

        return
      }

      sessions = sessions.filter(entry => entry.id !== id)
      paint()
      deps.transcript.print(`deleted ${name}`)
    } catch (err) {
      if (!lease.isCurrent()) {
        return
      }

      list.setStatus(undefined)
      list.setError(selectorError(err))
      deps.tui.requestRender()
    }
  }

  refresh()

  return {
    component: stage,
    dispose: () => {
      // A sent delete or resume is not rolled back by closing the picker; the
      // sequence guard is what keeps its answer from touching a newer screen.
      sequence += 1
    },
    focus: stage
  }
}
