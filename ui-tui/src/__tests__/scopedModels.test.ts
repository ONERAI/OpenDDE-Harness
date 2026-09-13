// pi's /scoped-models: which models /model shows under "scoped".
import { describe, expect, it, vi } from 'vitest'

import { createScopedModelsSelector, toggle } from '../selectors/scopedModels.js'
import { createHarness, declaredRow, fakeLease, KEYS, providerRow, settle } from './selectorHarness.js'

const OPENAI = providerRow({
  model_labels: { 'openai/gpt-5': { label: 'GPT-5' }, 'openai/gpt-5-mini': { label: 'GPT-5 mini' } },
  models: ['openai/gpt-5', 'openai/gpt-5-mini'],
  total_models: 2
})
const RELAY = declaredRow({
  authenticated: true,
  models: ['yunjintao/qwen3-32b'],
  name: 'yunjintao',
  slug: 'yunjintao',
  total_models: 1
})
const ALL = ['openai/gpt-5', 'openai/gpt-5-mini', 'yunjintao/qwen3-32b']
const ALT_DOWN = '\x1b[1;3B'
const CTRL_P = '\x10'
const CTRL_X = '\x18'

function build(results: Record<string, unknown> = {}, sessionScope?: null | string[]) {
  const harness = createHarness({
    'model.options': {
      default_model: '',
      model: 'openai/gpt-5',
      provider: 'openai',
      providers: [OPENAI, RELAY],
      refresh_errors: {}
    },
    'model.scope': { models: null },
    ...results
  })
  const handle = fakeLease('tui:abc')
  const onChange = vi.fn()
  const selector = createScopedModelsSelector({ deps: harness.deps, lease: handle.lease, onChange, sessionScope })

  return {
    handle,
    harness,
    onChange,
    screen: () => harness.draw(selector.component),
    view: selector.focus as { handleInput(data: string): void }
  }
}

const called = (harness: ReturnType<typeof createHarness>, method: string) =>
  harness.transport.calls.filter(call => call.method === method)

describe("pi's enabled set", () => {
  it('toggling out of "all" lists everything but the one; toggling back in collapses to all', () => {
    const without = toggle(null, ALL, 'openai/gpt-5')
    expect(without).toEqual(['openai/gpt-5-mini', 'yunjintao/qwen3-32b'])
    expect(toggle(without, ALL, 'openai/gpt-5')).toBeNull()
  })
})

describe('the selector', () => {
  it("shows pi's header, every model enabled, and the footer's key hints", async () => {
    const { screen } = build()

    await settle(8)

    const shown = screen()
    expect(shown).toContain('Model Configuration')
    expect(shown).toContain('Session-only. Ctrl+S to save to settings.')
    expect(shown).toContain('→ ✓ gpt-5 [openai]')
    expect(shown).toContain('✓ qwen3-32b [yunjintao]')
    // pi's footer, wrapped by the terminal: its keys are named as words.
    expect(shown).toContain('Enter toggle · Ctrl+A all · Ctrl+X clear · Ctrl+P provider')
    const modifier = process.platform === 'darwin' ? 'Option' : 'Alt'

    expect(shown.replace(/\s+/g, ' ')).toContain(`${modifier}+Up/${modifier}+Down reorder · Ctrl+S`)
    expect(shown).toContain('all enabled')
  })

  it('Enter toggles the selected model, session-only, and the footer says so', async () => {
    const { onChange, screen, view } = build()

    await settle(8)
    view.handleInput(KEYS.enter)

    expect(onChange).toHaveBeenLastCalledWith(['openai/gpt-5-mini', 'yunjintao/qwen3-32b'])
    expect(screen()).toContain('2/3 enabled')
    expect(screen()).toContain('(unsaved)')
    expect(screen()).not.toContain('✓ gpt-5 [openai]')
  })

  it('Ctrl+X clears, Ctrl+A enables all, Ctrl+P toggles a whole provider', async () => {
    const { onChange, screen, view } = build()

    await settle(8)
    view.handleInput(CTRL_X)
    expect(onChange).toHaveBeenLastCalledWith([])
    expect(screen()).toContain('0/3 enabled')

    view.handleInput(KEYS.ctrlA)
    expect(onChange).toHaveBeenLastCalledWith(null)
    expect(screen()).toContain('all enabled')

    view.handleInput(CTRL_P)
    expect(onChange).toHaveBeenLastCalledWith(['yunjintao/qwen3-32b'])
    expect(screen()).toContain('1/3 enabled')
  })

  it('Alt+Down moves an enabled model down the order', async () => {
    const { onChange, view } = build({}, ['openai/gpt-5', 'openai/gpt-5-mini'])

    await settle(8)
    view.handleInput(ALT_DOWN)

    expect(onChange).toHaveBeenLastCalledWith(['openai/gpt-5-mini', 'openai/gpt-5'])
  })

  it('Ctrl+S saves the list to the config and says so', async () => {
    const { harness, screen, view } = build({
      'model.scope': (params: { write?: boolean; models?: null | string[] }) => ({
        models: params.write ? params.models : null
      })
    })

    await settle(8)
    view.handleInput(KEYS.enter)
    view.handleInput(KEYS.ctrlS)
    await settle(8)

    expect(called(harness, 'model.scope').at(-1)?.params).toEqual({
      models: ['openai/gpt-5-mini', 'yunjintao/qwen3-32b'],
      session_id: 'tui:abc',
      write: true
    })
    expect(harness.printed.map(line => line.text)).toContain('Model selection saved to settings')
    expect(screen()).not.toContain('(unsaved)')
  })

  it('keeps a saved id no provider serves any more, struck through as unavailable', async () => {
    const { screen } = build({ 'model.scope': { models: ['openai/gpt-5', 'gone/old-model'] } })

    await settle(8)

    const shown = screen()
    expect(shown).toContain('old-model [unavailable]')
    expect(shown).toContain('1/3 enabled · 1 unavailable')
  })

  it('Esc cancels; Ctrl+C clears a search first', async () => {
    const { handle, screen, view } = build()

    await settle(8)
    view.handleInput('q')
    expect(screen()).not.toContain('gpt-5-mini')

    view.handleInput(KEYS.ctrlC)
    expect(screen()).toContain('gpt-5-mini')
    expect(handle.closes).toEqual([])

    view.handleInput(KEYS.escape)
    expect(handle.closes).toEqual(['cancel'])
  })
})
