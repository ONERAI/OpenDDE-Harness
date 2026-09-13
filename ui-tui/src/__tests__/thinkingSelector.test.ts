// The thinking selector: one overlay call, against the session's current model.

import { describe, expect, it, vi } from 'vitest'

import { createThinkingSelector, THINKING_ROWS } from '../selectors/thinkingSelector.js'
import { createHarness, fakeLease, KEYS, providerRow, settle } from './selectorHarness.js'

function build(results: Record<string, unknown> = {}, session: Record<string, unknown> = {}) {
  const harness = createHarness(results, { effort: 'medium', model: 'gpt-5', ...session })
  const handle = fakeLease()
  const applied = vi.fn()
  const selector = createThinkingSelector({ deps: harness.deps, lease: handle.lease, onApplied: applied })

  return { applied, handle, harness, selector, view: selector.focus as { handleInput(data: string): void } }
}

const overlayCalls = (harness: ReturnType<typeof createHarness>) =>
  harness.transport.calls.filter(call => call.method === 'model.overlay')

describe('thinking selector', () => {
  it('offers the seven levels plus default, and names the model the overlay is for', () => {
    const { harness, selector } = build()
    const drawn = harness.draw(selector.component)

    expect(THINKING_ROWS.map(row => row.id)).toEqual([
      'default',
      'off',
      'minimal',
      'low',
      'medium',
      'high',
      'xhigh',
      'max'
    ])
    expect(drawn).toContain('Thinking — gpt-5 · saved for this model')
    expect(drawn).toContain('xhigh')
  })

  it('marks the level the session is on', () => {
    const { harness, selector } = build()

    expect(harness.draw(selector.component)).toContain('✓ medium')
  })

  it('sends one overlay with the captured session id and closes', async () => {
    const { applied, handle, harness, view } = build({
      'model.overlay': { field: 'reasoning_effort', model: 'gpt-5', value: 'high' }
    })

    // default, off, minimal, low, medium, high
    for (let i = 0; i < 5; i += 1) {
      view.handleInput(KEYS.down)
    }

    view.handleInput(KEYS.enter)
    view.handleInput(KEYS.enter)
    await settle()

    expect(overlayCalls(harness)).toHaveLength(1)
    expect(overlayCalls(harness)[0]?.params).toEqual({
      field: 'reasoning_effort',
      session_id: 'tui:abc',
      value: 'high'
    })
    expect(applied).toHaveBeenCalledWith('high', 'gpt-5')
    expect(handle.closes).toEqual(['done'])
  })

  it('reads a null value as cleared rather than as a level', async () => {
    const { applied, harness, view } = build({
      'model.overlay': { field: 'reasoning_effort', model: 'gpt-5', value: null }
    })

    view.handleInput(KEYS.enter)
    await settle()

    expect(applied).toHaveBeenCalledWith(undefined, 'gpt-5')
    expect(harness.printed.some(line => line.text.includes('thinking: default'))).toBe(true)
  })

  it('reports the model the gateway actually changed', async () => {
    const { harness, view } = build({
      'model.overlay': { field: 'reasoning_effort', model: 'claude-opus-5', value: 'high' }
    })

    view.handleInput(KEYS.enter)
    await settle()

    expect(harness.printed.some(line => line.text.includes('claude-opus-5'))).toBe(true)
  })

  it('asks for the current model once when the session never reported one', async () => {
    const { harness } = build(
      { 'model.options': { model: 'gpt-5', provider: 'openai', providers: [providerRow()] } },
      { model: undefined }
    )

    await settle()

    const options = harness.transport.calls.filter(call => call.method === 'model.options')

    expect(options).toHaveLength(1)
    expect(options[0]?.params).toEqual({ include_catalog: false, session_id: 'tui:abc' })
  })

  it('shows the error and stays open when the overlay is refused', async () => {
    const { handle, harness, selector, view } = build({ 'model.overlay': new Error('no such level') })

    view.handleInput(KEYS.enter)
    await settle()

    expect(handle.closes).toHaveLength(0)
    expect(harness.draw(selector.component)).toContain('no such level')
  })

  it('ignores a result that arrives after the picker was replaced', async () => {
    const { applied, handle, view } = build({
      'model.overlay': { field: 'reasoning_effort', model: 'gpt-5', value: 'high' }
    })

    view.handleInput(KEYS.enter)
    handle.invalidate()
    await settle()

    // The mutation was sent and stands; only the screen is gone.
    expect(applied).not.toHaveBeenCalled()
  })

  it('cancels without calling anything', () => {
    const { handle, harness, view } = build()

    view.handleInput(KEYS.escape)

    expect(handle.closes).toEqual(['cancel'])
    expect(overlayCalls(harness)).toHaveLength(0)
  })
})
