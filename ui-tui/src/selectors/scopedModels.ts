// pi's `/scoped-models`: which models `/model` shows under its "scoped" tab.
//
// pi's own scoped-models selector (`coding-agent/.../components/
// scoped-models-selector.ts`): every available model, Enter toggles one,
// Ctrl+A enables all (the search's matches when searching), Ctrl+X clears,
// Ctrl+P toggles the selected model's whole provider, Alt+Up/Down reorders the
// enabled ones, and Ctrl+S saves the list to the config. Until saved the
// selection is session-only, which is pi's rule too — the footer says
// "(unsaved)" while it differs from what was saved.
//
// The list is pi's shape: `null` means every model is enabled (no scope); an
// explicit list is the enabled ids in the order they are cycled through. An id
// saved for a model no longer served is kept and shown struck through as
// "[unavailable]", so saving does not silently drop it.

import type { Focusable } from '@earendil-works/pi-tui'

import { Container, fuzzyFilter, Input, Spacer, Text } from '@earendil-works/pi-tui'

import type { Selector, SelectorLease } from '../components/selector.js'
import type { SelectorDeps } from './deps.js'
import type { ModelItem } from './modelList.js'

import { HorizontalRule, OptionalText } from '../components/selectorList.js'
import { keyDisplayText } from '../lib/keybindings.js'
import { selectorError } from './deps.js'
import { flattenModels, REFRESHED, REFRESHING } from './modelList.js'

/** pi's EnabledIds: `null` is every model, a list is the enabled ones in order. */
export type EnabledIds = null | string[]

export interface ScopedModelsOptions {
  deps: SelectorDeps
  lease: SelectorLease
  /** The session's current scope, when one was chosen this session. */
  sessionScope: EnabledIds | undefined
  /** The scope changed: session-only until saved. */
  onChange(enabled: EnabledIds): void
}

const MAX_VISIBLE = 8

function isEnabled(enabled: EnabledIds, id: string): boolean {
  return enabled === null || enabled.includes(id)
}

/** Collapse an explicit list back to null (= all enabled) when it covers every available model. */
function normalize(result: string[], all: string[]): EnabledIds {
  return result.length === all.length && result.every(id => all.includes(id)) ? null : result
}

export function toggle(enabled: EnabledIds, all: string[], id: string): EnabledIds {
  if (enabled === null) {
    return all.filter(item => item !== id)
  }
  const index = enabled.indexOf(id)
  if (index >= 0) {
    return [...enabled.slice(0, index), ...enabled.slice(index + 1)]
  }
  return normalize([...enabled, id], all)
}

export function enableAll(enabled: EnabledIds, all: string[], targets?: string[]): EnabledIds {
  if (enabled === null) {
    return null
  }
  const result = [...enabled]
  for (const id of targets ?? all) {
    if (!result.includes(id)) {
      result.push(id)
    }
  }
  return normalize(result, all)
}

export function clearAll(enabled: EnabledIds, all: string[], targets?: string[]): EnabledIds {
  if (enabled === null) {
    return targets ? all.filter(id => !targets.includes(id)) : []
  }
  const drop = new Set(targets ?? enabled)
  return enabled.filter(id => !drop.has(id))
}

function move(enabled: EnabledIds, id: string, delta: number): EnabledIds {
  if (enabled === null) {
    return null
  }
  const index = enabled.indexOf(id)
  const next = index + delta
  if (index < 0 || next < 0 || next >= enabled.length) {
    return enabled
  }
  const result = [...enabled]
  ;[result[index], result[next]] = [result[next]!, result[index]!]
  return result
}

function sortedIds(enabled: EnabledIds, all: string[]): string[] {
  if (enabled === null) {
    return all
  }
  const set = new Set(enabled)
  return [...enabled, ...all.filter(id => !set.has(id))]
}

interface Row {
  enabled: boolean
  full: string
  model: ModelItem | undefined
}

/** pi's search text for lists of models (`model-search.ts`). */
function searchText(item: ModelItem): string {
  const name = item.name && item.name !== item.id ? ` ${item.name}` : ''
  return `${item.id} ${item.provider} ${item.provider}/${item.id} ${item.provider} ${item.id}${name}`
}

export function createScopedModelsSelector({ deps, lease, onChange, sessionScope }: ScopedModelsOptions): Selector {
  const { theme } = deps
  const sessionId = lease.sessionId

  const byFull = new Map<string, ModelItem>()
  let allIds: string[] = []
  let enabled: EnabledIds = sessionScope ?? null
  let saved: EnabledIds = null
  let rows: Row[] = []
  let selectedIndex = 0
  let dirty = false
  let closed = false

  const view = new Container()
  const input = new Input({
    placeholder: '',
    placeholderStyle: text => theme.fg('muted', text),
    prompt: theme.fg('muted', '❯ ')
  })
  const list = new Container()
  const status = new OptionalText()
  const footer = new OptionalText()

  view.addChild(new HorizontalRule(theme))
  view.addChild(new Spacer(1))
  view.addChild(new Text(theme.fg('accent', theme.bold('Model Configuration')), 0, 0))
  view.addChild(
    new Text(
      theme.fg('muted', `Session-only. ${keyDisplayText(deps.keybindings, 'app.models.save')} to save to settings.`),
      0,
      0
    )
  )
  view.addChild(new Spacer(1))
  view.addChild(input)
  view.addChild(new Spacer(1))
  view.addChild(list)
  view.addChild(new Spacer(1))
  view.addChild(status)
  view.addChild(footer)
  view.addChild(new HorizontalRule(theme))

  input.onSubmit = () => toggleSelected()
  input.onEscape = () => cancel()

  function key(binding: Parameters<typeof keyDisplayText>[1]): string {
    return keyDisplayText(deps.keybindings, binding)
  }

  function footerText(): string {
    const enabledCount = enabled?.filter(id => byFull.has(id)).length ?? allIds.length
    const unavailable = enabled?.filter(id => !byFull.has(id)).length ?? 0
    const count =
      enabled === null
        ? 'all enabled'
        : `${enabledCount}/${allIds.length} enabled${unavailable ? ` · ${unavailable} unavailable` : ''}`
    const parts = [
      `${key('tui.select.confirm')} toggle`,
      `${key('app.models.enableAll')} all`,
      `${key('app.models.clearAll')} clear`,
      `${key('app.models.toggleProvider')} provider`,
      `${key('app.models.reorderUp')}/${key('app.models.reorderDown')} reorder`,
      `${key('app.models.save')} save`,
      count
    ]
    return dirty
      ? `${theme.fg('muted', `  ${parts.join(' · ')} `)}${theme.fg('warn', '(unsaved)')}`
      : theme.fg('muted', `  ${parts.join(' · ')}`)
  }

  function buildRows(): Row[] {
    return sortedIds(enabled, allIds).map(full => ({
      enabled: isEnabled(enabled, full),
      full,
      model: byFull.get(full)
    }))
  }

  function refresh(): void {
    const query = input.getValue()
    const built = buildRows()
    rows = query ? fuzzyFilter(built, query, row => (row.model ? searchText(row.model) : row.full)) : built
    selectedIndex = Math.min(selectedIndex, Math.max(0, rows.length - 1))
    paint()
  }

  function paint(): void {
    list.clear()
    if (rows.length === 0) {
      list.addChild(new Text(theme.fg('muted', '  No matching models'), 0, 0))
    } else {
      const start = Math.max(0, Math.min(selectedIndex - Math.floor(MAX_VISIBLE / 2), rows.length - MAX_VISIBLE))
      const end = Math.min(start + MAX_VISIBLE, rows.length)
      for (let i = start; i < end; i++) {
        const row = rows[i]!
        const selected = i === selectedIndex
        const prefix = selected ? theme.fg('accent', '→ ') : '  '
        const id = row.model?.id ?? row.full
        const styled = row.model ? id : theme.strikethrough(id)
        const text = selected ? theme.fg('accent', styled) : styled
        const badge = theme.fg('muted', row.model ? ` [${row.model.provider}]` : ' [unavailable]')
        const mark = row.model && row.enabled ? theme.fg('accent', '✓ ') : '  '
        list.addChild(new Text(`${prefix}${mark}${text}${badge}`, 0, 0))
      }
      if (start > 0 || end < rows.length) {
        list.addChild(new Text(theme.fg('muted', `  (${selectedIndex + 1}/${rows.length})`), 0, 0))
      }
      const selected = rows[selectedIndex]
      if (selected) {
        list.addChild(new Spacer(1))
        list.addChild(
          new Text(
            theme.fg('muted', `  ${selected.model ? `Model Name: ${selected.model.name}` : 'Model unavailable'}`),
            0,
            0
          )
        )
      }
    }
    footer.setText(footerText())
    deps.tui.requestRender()
  }

  function change(next: EnabledIds): void {
    enabled = next
    dirty = true
    refresh()
    onChange(enabled === null ? null : [...enabled])
  }

  function toggleSelected(): void {
    const row = rows[selectedIndex]
    if (row) {
      change(toggle(enabled, allIds, row.full))
    }
  }

  function cancel(): void {
    closed = true
    lease.done('cancel')
  }

  async function persist(): Promise<void> {
    try {
      const result = await deps.gateway.modelScope({ models: enabled, session_id: sessionId, write: true })
      if (closed || !lease.isCurrent()) {
        return
      }
      saved = result.models === null ? null : [...result.models]
      dirty = false
      deps.transcript.print('Model selection saved to settings')
      footer.setText(footerText())
      deps.tui.requestRender()
    } catch (err) {
      if (closed || !lease.isCurrent()) {
        return
      }
      status.setText(theme.fg('error', `  ${selectorError(err)}`))
      deps.tui.requestRender()
    }
  }

  const focusable: Container & Focusable & { dispose(): void; handleInput(data: string): void } = Object.assign(view, {
    get focused(): boolean {
      return input.focused
    },
    set focused(value: boolean) {
      input.focused = value
    },
    dispose(): void {
      closed = true
    },
    handleInput(data: string): void {
      const kb = deps.keybindings
      if (kb.matches(data, 'tui.select.up')) {
        if (rows.length > 0) {
          selectedIndex = selectedIndex === 0 ? rows.length - 1 : selectedIndex - 1
          paint()
        }
        return
      }
      if (kb.matches(data, 'tui.select.down')) {
        if (rows.length > 0) {
          selectedIndex = selectedIndex === rows.length - 1 ? 0 : selectedIndex + 1
          paint()
        }
        return
      }
      const up = kb.matches(data, 'app.models.reorderUp')
      const down = kb.matches(data, 'app.models.reorderDown')
      if (up || down) {
        const row = rows[selectedIndex]
        if (enabled !== null && row && isEnabled(enabled, row.full)) {
          const delta = up ? -1 : 1
          const at = enabled.indexOf(row.full)
          if (at + delta >= 0 && at + delta < enabled.length) {
            selectedIndex += delta
            change(move(enabled, row.full, delta))
          }
        }
        return
      }
      if (kb.matches(data, 'tui.select.confirm')) {
        toggleSelected()
        return
      }
      if (kb.matches(data, 'app.models.enableAll')) {
        change(enableAll(enabled, allIds, input.getValue() ? rows.map(row => row.full) : undefined))
        return
      }
      if (kb.matches(data, 'app.models.clearAll')) {
        change(clearAll(enabled, allIds, input.getValue() ? rows.map(row => row.full) : undefined))
        return
      }
      if (kb.matches(data, 'app.models.toggleProvider')) {
        const row = rows[selectedIndex]
        if (row?.model) {
          const provider = row.model.provider
          const ids = allIds.filter(full => byFull.get(full)?.provider === provider)
          const every = ids.every(full => isEnabled(enabled, full))
          change(every ? clearAll(enabled, allIds, ids) : enableAll(enabled, allIds, ids))
        }
        return
      }
      if (kb.matches(data, 'app.models.save')) {
        void persist()
        return
      }
      if (kb.matches(data, 'tui.select.cancel')) {
        // pi: Ctrl+C clears a search first; Esc cancels outright.
        if (data === '\x03' && input.getValue()) {
          input.setValue('')
          refresh()
          return
        }
        cancel()
        return
      }
      input.handleInput(data)
      refresh()
    }
  })

  // ── boot: the saved scope and the last known list, then the endpoints asked ──

  status.setText(theme.fg('muted', `  ${REFRESHING}`))
  paint()

  void (async () => {
    try {
      const [options, scope] = await Promise.all([
        deps.gateway.modelOptions({ include_catalog: true, session_id: sessionId }),
        deps.gateway.modelScope({ session_id: sessionId })
      ])
      if (closed || !lease.isCurrent()) {
        return
      }
      saved = scope.models === null ? null : [...scope.models]
      if (sessionScope === undefined) {
        enabled = saved === null ? null : [...saved]
      }
      absorb(flattenModels(options))
      const refreshed = await deps.gateway.modelOptions({ include_catalog: true, refresh: true, session_id: sessionId })
      if (closed || !lease.isCurrent()) {
        return
      }
      absorb(flattenModels(refreshed))
      const failed = Object.keys(refreshed.refresh_errors ?? {})
      status.setText(
        failed.length === 0
          ? theme.fg('ok', `  ${REFRESHED}`)
          : theme.fg('warn', `  Could not refresh ${failed.join(', ')}; showing cached models.`)
      )
      deps.tui.requestRender()
    } catch (err) {
      if (closed || !lease.isCurrent()) {
        return
      }
      status.setText(theme.fg('error', `  ${selectorError(err)}`))
      deps.tui.requestRender()
    }
  })()

  function absorb(items: ModelItem[]): void {
    const keep = rows[selectedIndex]?.full
    byFull.clear()
    allIds = []
    for (const item of items) {
      byFull.set(item.full, item)
      allIds.push(item.full)
    }
    refresh()
    const index = keep ? rows.findIndex(row => row.full === keep) : -1
    if (index >= 0) {
      selectedIndex = index
      paint()
    }
  }

  return { component: view, dispose: () => focusable.dispose(), focus: focusable }
}
