// The session picker: open one, delete one, and refuse to do either while the
// session on screen is busy.

import { describe, expect, it, vi } from 'vitest'

import type { SessionSelectorActions } from '../selectors/sessionSelector.js'

import { createSessionSelector, formatAge, sessionLabel } from '../selectors/sessionSelector.js'
import { createHarness, fakeLease, KEYS, settle } from './selectorHarness.js'

const NOW = 1_700_000_000_000

function listing(overrides: Record<string, unknown>[] = []) {
  return {
    sessions: [
      { id: 'tui:one', message_count: 4, preview: '', source: 'tui', started_at: NOW / 1000 - 600, title: 'First' },
      {
        id: 'tui:two',
        message_count: 9,
        preview: 'about proteins',
        source: 'tui',
        started_at: NOW / 1000 - 7200,
        title: ''
      },
      ...overrides
    ]
  }
}

function build(results: Record<string, unknown> = {}, actions: Partial<SessionSelectorActions> = {}) {
  const harness = createHarness({ 'session.list': listing(), ...results })
  const handle = fakeLease()
  const spies = {
    blockedReason: vi.fn(() => null as null | string),
    currentId: vi.fn(() => 'tui:one' as null | string),
    removeCurrent: vi.fn(async (_id: string) => true),
    resume: vi.fn(async (_id: string, _stillCurrent: () => boolean) => {})
  }
  const selector = createSessionSelector({
    actions: { ...spies, ...actions },
    deps: harness.deps,
    lease: handle.lease,
    now: () => NOW
  })

  return {
    handle,
    harness,
    selector,
    spies,
    view: selector.focus as { handleInput(data: string): void }
  }
}

describe('session rows', () => {
  it('falls back from title to preview to (untitled)', () => {
    expect(sessionLabel({ id: 'x', message_count: 0, preview: '', source: 't', started_at: 0, title: 'T' })).toBe('T')
    expect(sessionLabel({ id: 'x', message_count: 0, preview: 'p', source: 't', started_at: 0, title: '' })).toBe('p')
    expect(sessionLabel({ id: 'x', message_count: 0, preview: '', source: 't', started_at: 0, title: '' })).toBe(
      '(untitled)'
    )
  })

  it('reads started_at as Unix seconds', () => {
    expect(formatAge(NOW / 1000 - 600, NOW)).toBe('10m')
    expect(formatAge(NOW / 1000 - 7200, NOW)).toBe('2h')
    expect(formatAge(NOW / 1000 - 3 * 86_400, NOW)).toBe('3d')
    expect(formatAge(0, NOW)).toBe('—')
  })
})

describe('session selector', () => {
  it('asks for 200 sessions and shows id, count and age', async () => {
    const { harness, selector } = build()

    await settle()

    const call = harness.transport.calls.find(entry => entry.method === 'session.list')

    expect(call?.params).toEqual({ limit: 200 })

    const drawn = harness.draw(selector.component)

    expect(drawn).toContain('First')
    expect(drawn).toContain('tui:one')
    expect(drawn).toContain('4 msg')
    expect(drawn).toContain('10m')
    expect(drawn).toContain('2 sessions')
  })

  it('marks the session on screen', async () => {
    const { harness, selector } = build()

    await settle()

    expect(harness.draw(selector.component)).toContain('✓ First')
  })

  it('searches title, id and preview', async () => {
    const { harness, selector, view } = build()

    await settle()
    view.handleInput('p')
    view.handleInput('r')
    view.handleInput('o')

    const drawn = harness.draw(selector.component)

    expect(drawn).toContain('about proteins')
    expect(drawn).not.toContain('First')
  })

  it('hands the switch to the app, guarded by the lease', async () => {
    const { handle, spies, view } = build()

    await settle()
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)
    await settle()

    expect(spies.resume).toHaveBeenCalledTimes(1)
    expect(spies.resume.mock.calls[0]?.[0]).toBe('tui:two')

    const guard = spies.resume.mock.calls[0]![1]

    expect(guard()).toBe(true)
    handle.invalidate()
    expect(guard()).toBe(false)
  })

  it('refuses a switch while a turn or a queue is in the way', async () => {
    const { harness, selector, spies, view } = build({}, { blockedReason: () => 'a turn is running' })

    await settle()
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)
    await settle()

    expect(spies.resume).not.toHaveBeenCalled()
    expect(harness.draw(selector.component)).toContain('a turn is running')
    expect(harness.draw(selector.component)).toContain('Esc closes this picker')
  })

  it('deletes an inactive session only when the gateway says it deleted that id', async () => {
    const { harness, selector, view } = build({ 'session.delete': { deleted: 'tui:two' } })

    await settle()
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.ctrlD)

    expect(harness.draw(selector.component)).toContain('Delete session')

    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)
    await settle()

    expect(harness.transport.calls.find(call => call.method === 'session.delete')?.params).toEqual({
      session_id: 'tui:two'
    })
    expect(harness.draw(selector.component)).not.toContain('about proteins')
  })

  it('treats deleted:null as failure, not success', async () => {
    const { harness, selector, view } = build({ 'session.delete': { deleted: null } })

    await settle()
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.ctrlD)
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)
    await settle()

    expect(harness.draw(selector.component)).toContain('nothing was deleted for tui:two')
  })

  it('keeps the session when the confirmation is declined', async () => {
    const { harness, view } = build({ 'session.delete': { deleted: 'tui:two' } })

    await settle()
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.ctrlD)
    view.handleInput(KEYS.escape)
    await settle()

    expect(harness.transport.calls.some(call => call.method === 'session.delete')).toBe(false)
  })

  it("says the backend cannot refuse a delete on another client's behalf", async () => {
    const { harness, selector, view } = build()

    await settle()
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.ctrlD)

    // The gateway has no active-turn guard, so the confirmation says so.
    expect(harness.draw(selector.component)).toContain('does not itself refuse')
  })

  it('routes deleting the session on screen through the app, which closes it first', async () => {
    const { handle, spies, view } = build()

    await settle()
    view.handleInput(KEYS.ctrlD)
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)
    await settle()

    expect(spies.removeCurrent).toHaveBeenCalledWith('tui:one')
    expect(handle.closes).toEqual(['done'])
  })

  it('refuses to delete the session on screen while it is busy', async () => {
    const { harness, selector, spies, view } = build({}, { blockedReason: () => 'a turn is running' })

    await settle()
    view.handleInput(KEYS.ctrlD)

    expect(spies.removeCurrent).not.toHaveBeenCalled()
    expect(harness.draw(selector.component)).toContain('finish or cancel it before deleting')
  })

  it('shows the list error rather than an empty picker', async () => {
    const { harness, selector } = build({ 'session.list': new Error('gateway down') })

    await settle()

    expect(harness.draw(selector.component)).toContain('gateway down')
  })

  it('ignores a listing that arrives after the picker was replaced', async () => {
    const { handle, harness, selector } = build()

    handle.invalidate()
    await settle()

    expect(harness.draw(selector.component)).toContain('loading…')
  })
})
