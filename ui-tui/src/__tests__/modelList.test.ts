// pi's /model: one flat list of every model the connected providers serve.
import { describe, expect, it, vi } from 'vitest'

import { createModelListSelector, flattenModels, NO_SCOPE_HINT, REFRESHED } from '../selectors/modelList.js'
import { createHarness, declaredRow, fakeLease, KEYS, providerRow, settle } from './selectorHarness.js'

const OPENAI = providerRow({
  is_current: true,
  model_labels: { 'openai/gpt-5': { label: 'GPT-5' }, 'openai/gpt-5-mini': { label: 'GPT-5 mini' } },
  models: ['openai/gpt-5', 'openai/gpt-5-mini'],
  total_models: 2
})
const RELAY = declaredRow({
  authenticated: true,
  model_labels: { 'yunjintao/anthropic/claude-opus-5': { label: 'Anthropic: Claude Opus 5' } },
  models: ['yunjintao/anthropic/claude-opus-5'],
  name: 'yunjintao',
  slug: 'yunjintao',
  total_models: 1
})
const DISCONNECTED = providerRow({ authenticated: false, is_current: false, name: 'DeepSeek', slug: 'deepseek' })

function options(
  providers = [OPENAI, RELAY, DISCONNECTED],
  model = 'openai/gpt-5',
  extra: Record<string, unknown> = {}
) {
  return { default_model: 'openai/gpt-5-mini', model, provider: 'openai', providers, refresh_errors: {}, ...extra }
}

function build(
  results: Record<string, unknown> = {},
  opts: {
    initialQuery?: string
    scope?: 'default' | 'session'
    sessionId?: null | string
    sessionScope?: null | string[]
  } = {}
) {
  const harness = createHarness({ 'model.options': options(), 'model.scope': { models: null }, ...results })
  const handle = fakeLease(opts.sessionId === undefined ? 'tui:abc' : opts.sessionId)
  const applied = vi.fn()
  const selector = createModelListSelector({
    actions: { applied },
    deps: harness.deps,
    lease: handle.lease,
    scope: opts.scope ?? 'session',
    ...(opts.initialQuery ? { initialQuery: opts.initialQuery } : {}),
    ...(opts.sessionScope !== undefined ? { sessionScope: opts.sessionScope } : {})
  })

  return {
    applied,
    handle,
    harness,
    screen: () => harness.draw(selector.component),
    selector,
    view: selector.focus as { handleInput(data: string): void }
  }
}

const called = (harness: ReturnType<typeof createHarness>, method: string) =>
  harness.transport.calls.filter(call => call.method === method)

describe('the list', () => {
  it("flattens every connected provider's models and leaves the disconnected ones out", () => {
    const items = flattenModels(options() as unknown as Parameters<typeof flattenModels>[0])

    expect(items.map(item => `${item.id} [${item.provider}]`)).toEqual([
      'gpt-5 [openai]',
      'gpt-5-mini [openai]',
      'anthropic/claude-opus-5 [yunjintao]'
    ])
    expect(items[2]?.name).toBe('Anthropic: Claude Opus 5')
  })

  it("draws pi's rows: the current model first with its mark, the default badged, the provider in brackets", async () => {
    const { screen } = build()

    await settle(8)

    const shown = screen()

    expect(shown).toContain(NO_SCOPE_HINT)
    expect(shown).toContain('→ ✓ gpt-5 [openai]')
    expect(shown).toContain('gpt-5-mini [openai] · default')
    expect(shown).toContain('anthropic/claude-opus-5 [yunjintao]')
    expect(shown).not.toContain('DeepSeek')
    expect(shown).toContain('Model Name: GPT-5')
    // pi's hint line names its keys as words (`keyDisplayText`), unlike the
    // scope hint below, where the key is chrome beside the word it labels.
    expect(shown).toContain('Enter to select · Ctrl+S to set as default · Escape/Ctrl+C to cancel')
    expect(shown.indexOf('gpt-5 [openai]')).toBeLessThan(shown.indexOf('gpt-5-mini'))
  })

  it('paints the last known list at once and refreshes the catalogs behind it', async () => {
    const refreshed = options([OPENAI, declaredRow({ ...RELAY, models: [...RELAY.models, 'yunjintao/qwen3-32b'] })])
    const { harness, screen } = build({
      'model.options': (params: { refresh?: boolean }) => (params.refresh ? refreshed : options())
    })

    await settle(8)

    const calls = called(harness, 'model.options')
    expect(calls.map(call => call.params)).toEqual([
      { include_catalog: true, session_id: 'tui:abc' },
      { include_catalog: true, refresh: true, session_id: 'tui:abc' }
    ])
    expect(screen()).toContain('qwen3-32b [yunjintao]')
    expect(screen()).toContain(REFRESHED)
  })

  it("says pi's sentence when an endpoint could not be refreshed, and keeps the cached rows", async () => {
    const { screen } = build({
      'model.options': (params: { refresh?: boolean }) =>
        params.refresh ? options(undefined, undefined, { refresh_errors: { yunjintao: 'answered 503' } }) : options()
    })

    await settle(8)

    expect(screen()).toContain('Could not refresh yunjintao; showing cached models.')
    expect(screen()).toContain('anthropic/claude-opus-5 [yunjintao]')
  })

  it('filters with the search, and starts filtered for /model <text>', async () => {
    const { screen, view } = build({}, { initialQuery: 'opus' })

    await settle(8)

    expect(screen()).toContain('claude-opus-5')
    expect(screen()).not.toContain('gpt-5-mini')

    for (const char of 'zzz') {
      view.handleInput(char)
    }

    expect(screen()).toContain('No matching models')
  })
})

describe('choosing', () => {
  it("Enter moves this session to the qualified id and says pi's sentence", async () => {
    const { applied, handle, harness, view } = build({
      'config.set': { applied: true, applies_to_session: true, previous: 'openai/gpt-5', value: 'openai/gpt-5-mini' }
    })

    await settle(8)
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)
    await settle(8)

    expect(called(harness, 'config.set')[0]?.params).toEqual({
      key: 'model',
      scope: 'session',
      session_id: 'tui:abc',
      value: 'openai/gpt-5-mini'
    })
    expect(harness.printed.map(line => line.text)).toContain('Model: gpt-5-mini')
    expect(applied).toHaveBeenCalledWith({ model: 'openai/gpt-5-mini', provider: 'openai' })
    expect(handle.closes).toEqual(['done'])
  })

  it('Ctrl+S saves the default for new sessions and says so the way pi does', async () => {
    const { applied, handle, harness, view } = build({
      'config.set': {
        applied: true,
        applies_to_session: false,
        previous: '',
        value: 'yunjintao/anthropic/claude-opus-5'
      }
    })

    await settle(8)
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.ctrlS)
    await settle(8)

    expect(called(harness, 'config.set')[0]?.params).toMatchObject({
      scope: 'default',
      value: 'yunjintao/anthropic/claude-opus-5'
    })
    expect(harness.printed.map(line => line.text)).toContain('Default model: yunjintao/anthropic/claude-opus-5')
    expect(applied).not.toHaveBeenCalled()
    expect(handle.closes).toEqual(['done'])
  })

  it('refuses a session-scoped change before a session exists', async () => {
    const { harness, screen, view } = build({}, { sessionId: null })

    await settle(8)
    view.handleInput(KEYS.enter)
    await settle()

    expect(called(harness, 'config.set')).toHaveLength(0)
    expect(screen()).toContain('/model --default')
  })

  it('cancels on escape without writing anything', async () => {
    const { handle, harness, view } = build()

    await settle(8)
    view.handleInput(KEYS.escape)

    expect(handle.closes).toEqual(['cancel'])
    expect(called(harness, 'config.set')).toHaveLength(0)
  })
})

describe('the scope', () => {
  it("opens on the saved scope, with pi's two lines, and Tab flips to all", async () => {
    const { screen, view } = build({ 'model.scope': { models: ['yunjintao/anthropic/claude-opus-5'] } })

    await settle(8)

    let shown = screen()
    expect(shown).toContain('Scope: all | scoped')
    expect(shown).toContain('tab scope (all/scoped)')
    expect(shown).toContain('claude-opus-5')
    expect(shown).not.toContain('gpt-5-mini')
    expect(shown).not.toContain(NO_SCOPE_HINT)

    view.handleInput(KEYS.tab)

    shown = screen()
    expect(shown).toContain('gpt-5-mini')
  })

  it("the session's own scope wins over the saved one", async () => {
    const { harness, screen } = build(
      { 'model.scope': { models: ['openai/gpt-5'] } },
      { sessionScope: ['openai/gpt-5-mini'] }
    )

    await settle(8)

    expect(called(harness, 'model.scope')).toHaveLength(0)
    expect(screen()).toContain('gpt-5-mini')
    expect(screen()).not.toContain('claude-opus-5')
  })
})
