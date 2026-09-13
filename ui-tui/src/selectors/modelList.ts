// pi's `/model`: every model the connected providers serve, in one list.
//
// pi's own model selector (`coding-agent/.../components/model-selector.ts`),
// drawn the way pi draws it: a scope line when a scope exists ("Scope: all |
// scoped", Tab flips it), a search input, ten rows of `→ ✓ id [provider] ·
// default`, the `(n/N)` scroll mark, "Model Name:" under the list, the
// catalog-refresh status, and pi's hint line. Enter moves this session, Ctrl+S
// saves the default, Esc cancels.
//
// The rows come from `model.options`: every provider that is connected, every
// model it serves. The picker paints that at once and then asks the gateway to
// refresh the declared endpoints' catalogs in the background, the way pi
// refreshes its dynamic providers when the selector opens — "Model catalogs
// refreshed." when they answer, pi's "showing cached models" sentence when one
// does not. Providers are not added here: that is `/login`, as in pi.

import type { Focusable } from '@earendil-works/pi-tui'

import { Container, fuzzyFilter, Input, Spacer, Text } from '@earendil-works/pi-tui'

import type { Selector, SelectorLease } from '../components/selector.js'
import type { ModelOptionProvider, ModelOptionsResult } from '../rpc/index.js'
import type { SelectorDeps } from './deps.js'

import { HorizontalRule, OptionalText } from '../components/selectorList.js'
import { keyDisplayText, keyText } from '../lib/keybindings.js'
import { ConfigValidationError, ModelNotAvailableError, NotSupportedInV01Error } from '../rpc/index.js'
import { selectorError } from './deps.js'

export type ModelScope = 'default' | 'session'
type ListScope = 'all' | 'scoped'

/** One model as the list holds it: pi's `{provider, id, model}`. */
export interface ModelItem {
  /** The qualified id `config.set` takes: `<provider>/<id>`. */
  full: string
  /** pi's model id, bare. */
  id: string
  /** pi's `model.name`: what "Model Name:" shows. */
  name: string
  provider: string
}

export interface ModelListActions {
  applied(update: { model?: string; provider?: string }): void
}

export interface ModelListOptions {
  actions: ModelListActions
  deps: SelectorDeps
  /** `/model <text>`: typed into the search before the list is shown. */
  initialQuery?: string
  lease: SelectorLease
  /** What Enter writes: this session's model, or the default for new sessions. */
  scope: ModelScope
  /** pi's session-only scope, when `/scoped-models` chose one this session;
   *  undefined follows the saved scope (`model.scope`). */
  sessionScope?: null | string[]
}

/** pi's own hint above the list when no scope exists. */
export const NO_SCOPE_HINT = 'Only showing models from configured providers. Use /login to add providers.'
export const REFRESHING = 'Refreshing model catalogs…'
export const REFRESHED = 'Model catalogs refreshed.'
const MAX_VISIBLE = 10

/** pi's search text for the `/model` selector (`model-search.ts`): the bare id
 *  is kept out of the leading position so `provider/id` queries rank first. */
export function selectorSearchText(item: ModelItem): string {
  const name = item.name && item.name !== item.id ? ` ${item.name}` : ''
  return `${item.provider} ${item.provider}/${item.id} ${item.provider} ${item.id}${name}`
}

/** The flat list `model.options` describes: every connected provider's models. */
export function flattenModels(result: Pick<ModelOptionsResult, 'providers'>): ModelItem[] {
  const items: ModelItem[] = []
  for (const provider of result.providers ?? []) {
    if (!provider.authenticated && !provider.is_current) {
      continue
    }
    for (const full of provider.models) {
      items.push(modelItem(provider, full))
    }
  }
  return items
}

function modelItem(provider: ModelOptionProvider, full: string): ModelItem {
  const prefix = `${provider.slug}/`
  const id = full.startsWith(prefix) ? full.slice(prefix.length) : full
  return { full, id, name: provider.model_labels?.[full]?.label || id, provider: provider.slug }
}

function explain(err: unknown): string {
  if (err instanceof ModelNotAvailableError) {
    return 'that model is not available — the provider may need a key or a sign-in first'
  }
  if (err instanceof ConfigValidationError) {
    const detail = (err.data as { detail?: unknown } | null)?.detail
    return typeof detail === 'string' && detail.trim() ? detail : 'the gateway rejected that value'
  }
  if (err instanceof NotSupportedInV01Error) {
    return 'this gateway build cannot do that yet — use the ddeharness CLI'
  }
  return selectorError(err)
}

export function createModelListSelector({
  actions,
  deps,
  initialQuery,
  lease,
  scope,
  sessionScope
}: ModelListOptions): Selector {
  const { theme } = deps
  const sessionId = lease.sessionId

  let allModels: ModelItem[] = []
  /** The scoped ids in force: the session's, else the saved ones. Empty is pi's "no scope". */
  let scopedModels: readonly string[] = sessionScope ?? []
  let currentFull = ''
  let defaultFull = ''
  let listScope: ListScope = scopedModels.length > 0 ? 'scoped' : 'all'
  let filtered: ModelItem[] = []
  let selectedIndex = 0
  let errorMessage = ''
  let refreshStatus = REFRESHING
  let refreshOk = false
  let busy = false
  let closed = false

  // ── the view ───────────────────────────────────────────────────────

  const view = new Container()
  const scopeText = new OptionalText()
  const scopeHint = new OptionalText()
  const input = new Input({
    placeholder: '',
    placeholderStyle: text => theme.fg('muted', text),
    prompt: theme.fg('muted', '❯ ')
  })
  const list = new Container()

  view.addChild(new HorizontalRule(theme))
  view.addChild(new Spacer(1))
  view.addChild(scopeText)
  view.addChild(scopeHint)
  view.addChild(new Spacer(1))
  view.addChild(input)
  view.addChild(new Spacer(1))
  view.addChild(list)
  view.addChild(new Spacer(1))
  view.addChild(
    new Text(
      theme.fg(
        'muted',
        `  ${keyDisplayText(deps.keybindings, 'tui.select.confirm')} to select · ${keyDisplayText(deps.keybindings, 'app.models.save')} to set as default · ${keyDisplayText(deps.keybindings, 'tui.select.cancel')} to cancel`
      ),
      0,
      0
    )
  )
  view.addChild(new HorizontalRule(theme))

  if (initialQuery) {
    input.setValue(initialQuery)
  }
  input.onSubmit = () => confirm(scope)
  input.onEscape = () => cancel()

  function scopeLine(): string {
    const all = listScope === 'all' ? theme.fg('accent', 'all') : theme.fg('muted', 'all')
    const scoped = listScope === 'scoped' ? theme.fg('accent', 'scoped') : theme.fg('muted', 'scoped')
    return `${theme.fg('muted', 'Scope: ')}${all}${theme.fg('muted', ' | ')}${scoped}`
  }

  /** pi's two lines when a scope exists, or its one hint when none does. */
  function paintScope(): void {
    if (scopedModels.length > 0) {
      scopeText.setText(scopeLine())
      scopeHint.setText(
        `${theme.fg('label', keyText(deps.keybindings, 'tui.input.tab'))} scope${theme.fg('muted', ' (all/scoped)')}`
      )
    } else {
      scopeText.setText(theme.fg('warn', NO_SCOPE_HINT))
      scopeHint.setText('')
    }
  }

  // ── the data ───────────────────────────────────────────────────────

  function isCurrent(item: ModelItem): boolean {
    return item.full === currentFull
  }

  function isDefault(item: ModelItem): boolean {
    return defaultFull !== '' && item.full === defaultFull
  }

  /** pi's order: the current model first, the default second, then by provider. */
  function sorted(items: ModelItem[]): ModelItem[] {
    return [...items].sort((a, b) => {
      if (isCurrent(a) !== isCurrent(b)) {
        return isCurrent(a) ? -1 : 1
      }
      if (isDefault(a) !== isDefault(b)) {
        return isDefault(a) ? -1 : 1
      }
      return a.provider.localeCompare(b.provider)
    })
  }

  function active(): ModelItem[] {
    if (listScope !== 'scoped') {
      return allModels
    }
    // pi's scoped list is in the saved order, whether or not the model is
    // known right now.
    const byFull = new Map(allModels.map(item => [item.full, item]))
    return scopedModels.flatMap(full => {
      const known = byFull.get(full)
      if (known) {
        return [known]
      }
      const slash = full.indexOf('/')
      return [
        {
          full,
          id: slash > 0 ? full.slice(slash + 1) : full,
          name: full,
          provider: slash > 0 ? full.slice(0, slash) : ''
        }
      ]
    })
  }

  function isDefaultSearch(query: string): boolean {
    const normalized = query.trim().toLowerCase()
    return normalized.length > 0 && 'default'.startsWith(normalized)
  }

  function filter(): void {
    const query = input.getValue()
    const items = active()
    if (query) {
      const matched = fuzzyFilter(
        items,
        query,
        item => `${selectorSearchText(item)}${isDefault(item) ? ' default' : ''}`
      )
      if (isDefaultSearch(query)) {
        const defaults = items.filter(isDefault)
        const keys = new Set(defaults.map(item => item.full))
        filtered = [...defaults, ...matched.filter(item => !keys.has(item.full))]
      } else {
        filtered = matched
      }
    } else {
      filtered = items
    }
    selectedIndex = query ? 0 : Math.min(selectedIndex, Math.max(0, filtered.length - 1))
    paintList()
  }

  function load(result: ModelOptionsResult): void {
    currentFull = result.model ?? ''
    allModels = sorted(flattenModels(result))
    const currentIndex = filtered.findIndex(isCurrent)
    selectedIndex = currentIndex >= 0 ? currentIndex : Math.min(selectedIndex, Math.max(0, filtered.length - 1))
    filter()
    const nowCurrent = filtered.findIndex(isCurrent)
    if (nowCurrent >= 0 && !input.getValue()) {
      selectedIndex = nowCurrent
      paintList()
    }
  }

  // ── painting ───────────────────────────────────────────────────────

  function paintList(): void {
    list.clear()
    const start = Math.max(0, Math.min(selectedIndex - Math.floor(MAX_VISIBLE / 2), filtered.length - MAX_VISIBLE))
    const end = Math.min(start + MAX_VISIBLE, filtered.length)

    for (let i = start; i < end; i++) {
      const item = filtered[i]
      if (!item) {
        continue
      }
      const selected = i === selectedIndex
      const cursor = selected ? theme.fg('accent', '→ ') : '  '
      const mark = isCurrent(item) ? theme.fg('accent', '✓ ') : '  '
      const id = selected ? theme.fg('accent', item.id) : item.id
      const badge = theme.fg('muted', `[${item.provider}]`)
      const asDefault = isDefault(item) ? theme.fg('muted', ' · default') : ''
      list.addChild(new Text(`${cursor}${mark}${id} ${badge}${asDefault}`, 0, 0))
    }

    if (start > 0 || end < filtered.length) {
      list.addChild(new Text(theme.fg('muted', `  (${selectedIndex + 1}/${filtered.length})`), 0, 0))
    }

    if (errorMessage) {
      for (const line of errorMessage.split('\n')) {
        list.addChild(new Text(theme.fg('error', line), 0, 0))
      }
    } else if (filtered.length === 0) {
      list.addChild(new Text(theme.fg('muted', '  No matching models'), 0, 0))
    } else {
      const selected = filtered[selectedIndex]
      if (selected) {
        list.addChild(new Spacer(1))
        list.addChild(new Text(theme.fg('muted', `  Model Name: ${selected.name}`), 0, 0))
      }
    }

    if (refreshStatus) {
      list.addChild(new Spacer(1))
      list.addChild(new Text(theme.fg(refreshOk ? 'ok' : 'muted', `  ${refreshStatus}`), 0, 0))
    }
    deps.tui.requestRender()
  }

  // ── keys ───────────────────────────────────────────────────────────

  function cancel(): void {
    closed = true
    lease.done('cancel')
  }

  function setScope(next: ListScope): void {
    if (listScope === next) {
      return
    }
    listScope = next
    const currentIndex = active().findIndex(isCurrent)
    selectedIndex = currentIndex >= 0 ? currentIndex : 0
    paintScope()
    filter()
  }

  function moveBy(delta: number): void {
    if (filtered.length === 0) {
      return
    }
    selectedIndex = (selectedIndex + delta + filtered.length) % filtered.length
    paintList()
  }

  function confirm(as: ModelScope): void {
    const item = filtered[selectedIndex]
    if (item) {
      void apply(item, as)
    }
  }

  /** `as` is what gets written: this session, or the default for new sessions. */
  async function apply(item: ModelItem, as: ModelScope): Promise<void> {
    if (busy) {
      return
    }
    if (as === 'session' && !sessionId) {
      errorMessage = 'no session yet — use /model --default, or start a conversation first'
      paintList()
      return
    }
    busy = true
    try {
      const result = await deps.gateway.configSet({
        key: 'model',
        scope: as,
        value: item.full,
        ...(sessionId ? { session_id: sessionId } : {})
      })
      if (!lease.isCurrent()) {
        return
      }
      if (!result.value || !result.applied) {
        errorMessage = result.value ? `not applied: ${result.value}` : 'the gateway applied no model'
        paintList()
        return
      }
      closed = true
      // pi's own two sentences for the two keys.
      if (as === 'default') {
        deps.transcript.print(`Default model: ${result.value}`)
      } else {
        deps.transcript.print(`Model: ${item.id}`)
      }
      if (result.applies_to_session !== false) {
        actions.applied({ model: result.value, provider: item.provider })
      }
      lease.done('done')
    } catch (err) {
      if (!lease.isCurrent()) {
        return
      }
      errorMessage = explain(err)
      paintList()
    } finally {
      busy = false
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
      if (kb.matches(data, 'tui.input.tab')) {
        if (scopedModels.length > 0) {
          setScope(listScope === 'all' ? 'scoped' : 'all')
        }
        return
      }
      if (kb.matches(data, 'tui.select.up')) {
        moveBy(-1)
        return
      }
      if (kb.matches(data, 'tui.select.down')) {
        moveBy(1)
        return
      }
      if (kb.matches(data, 'tui.select.confirm')) {
        confirm(scope)
        return
      }
      if (kb.matches(data, 'tui.select.cancel')) {
        cancel()
        return
      }
      if (kb.matches(data, 'app.models.save')) {
        confirm('default')
        return
      }
      input.handleInput(data)
      filter()
    }
  })

  // ── boot: the last known list at once, the endpoints asked in the background ──

  paintScope()
  paintList()

  void (async () => {
    try {
      const [first, saved] = await Promise.all([
        deps.gateway.modelOptions({ include_catalog: true, session_id: sessionId }),
        sessionScope === undefined ? deps.gateway.modelScope({ session_id: sessionId }) : Promise.resolve(undefined)
      ])
      if (closed || !lease.isCurrent()) {
        return
      }
      if (saved !== undefined) {
        scopedModels = saved.models ?? []
        listScope = scopedModels.length > 0 ? 'scoped' : 'all'
        paintScope()
      }
      defaultFull = first.default_model ?? ''
      load(first)
      const refreshed = await deps.gateway.modelOptions({ include_catalog: true, refresh: true, session_id: sessionId })
      if (closed || !lease.isCurrent()) {
        return
      }
      const failed = Object.keys(refreshed.refresh_errors ?? {}).filter(key => key !== '*')
      if (refreshed.refresh_errors?.['*']) {
        refreshStatus = ''
        errorMessage = `Could not refresh model catalogs: ${refreshed.refresh_errors['*']}`
      } else if (failed.length === 1) {
        refreshStatus = ''
        errorMessage = `Could not refresh ${failed[0]}; showing cached models.`
      } else if (failed.length > 1) {
        refreshStatus = ''
        errorMessage = `Could not refresh ${failed.length} model catalogs (${failed.join(', ')}); showing cached models.`
      } else {
        refreshStatus = REFRESHED
        refreshOk = true
      }
      defaultFull = refreshed.default_model ?? defaultFull
      load(refreshed)
    } catch (err) {
      if (closed || !lease.isCurrent()) {
        return
      }
      refreshStatus = ''
      errorMessage = explain(err)
      paintList()
    }
  })()

  return { component: view, dispose: () => focusable.dispose(), focus: focusable }
}
