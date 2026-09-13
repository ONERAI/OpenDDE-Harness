// Who owns the editor slot, and what the shared list does with a key.

import { stripTerminalSequences, Text } from '@earendil-works/pi-tui'
import { describe, expect, it, vi } from 'vitest'

import type { Selector, SelectorLease } from '../components/selector.js'
import type { SelectorListOptions } from '../components/selectorList.js'

import { SelectorHost } from '../components/selector.js'
import { SelectorList, visibleRowCount } from '../components/selectorList.js'
import { Theme } from '../theme.js'
import { createHarness, KEYS } from './selectorHarness.js'

function build() {
  const harness = createHarness()
  const changes: boolean[] = []
  const host = new SelectorHost({
    editor: harness.editor,
    editorContainer: harness.editorContainer,
    onChange: active => changes.push(active),
    session: () => ({ epoch: harness.session.epoch, id: harness.session.id }),
    tui: harness.tui
  })

  return { changes, harness, host }
}

function panel(label: string, dispose = vi.fn()): Selector {
  const component = new Text(label, 0, 0)

  return { component, dispose, focus: component }
}

describe('SelectorHost', () => {
  it('replaces the editor and puts it back, keeping the same editor object', () => {
    const { changes, harness, host } = build()
    let lease!: SelectorLease

    host.show(handle => {
      lease = handle

      return panel('PICKER')
    })

    expect(harness.editorContainer.children).toHaveLength(1)
    expect(harness.draw(harness.editorContainer)).toContain('PICKER')
    expect(host.active).toBe(true)

    lease.done()

    expect(harness.editorContainer.children[0]).toBe(harness.editor)
    expect(host.active).toBe(false)
    expect(changes).toEqual([true, false])
  })

  it('ignores a second done, and a stale done cannot close a newer selector', () => {
    const { harness, host } = build()
    let first!: SelectorLease

    host.show(handle => {
      first = handle

      return panel('FIRST')
    })

    const firstDispose = vi.fn()

    first.done()
    first.done()

    host.show(() => panel('SECOND', firstDispose))
    // The lease from the first selector is long gone; calling it again must not
    // restore the editor over the second one.
    first.done()

    expect(harness.draw(harness.editorContainer)).toContain('SECOND')
    expect(firstDispose).not.toHaveBeenCalled()
  })

  it('disposes the previous selector when a new one takes the slot', () => {
    const { host } = build()
    const dispose = vi.fn()

    host.show(() => panel('FIRST', dispose))
    host.show(() => panel('SECOND'))

    expect(dispose).toHaveBeenCalledTimes(1)
  })

  it('aborts the lease signal when the slot is reset', () => {
    const { host } = build()
    let lease!: SelectorLease

    host.show(handle => {
      lease = handle

      return panel('ONE')
    })

    expect(lease.signal.aborted).toBe(false)
    host.reset('session-switch')
    expect(lease.signal.aborted).toBe(true)
    expect(lease.isCurrent()).toBe(false)
  })

  it('restores the editor when the factory throws', () => {
    const { harness, host } = build()

    expect(() =>
      host.show(() => {
        throw new Error('boom')
      })
    ).toThrow('boom')

    expect(harness.editorContainer.children[0]).toBe(harness.editor)
    expect(host.active).toBe(false)
  })

  it('disposes a selector that closed itself before it returned, and mounts nothing', () => {
    const { harness, host } = build()
    const dispose = vi.fn()

    host.show(lease => {
      lease.done()

      return panel('NEVER', dispose)
    })

    expect(dispose).toHaveBeenCalledTimes(1)
    expect(harness.editorContainer.children[0]).toBe(harness.editor)
    expect(host.active).toBe(false)
  })

  it('carries the session it was opened against in the lease', () => {
    const { harness, host } = build()
    let lease!: SelectorLease

    harness.session.epoch = 7
    host.show(handle => {
      lease = handle

      return panel('ONE')
    })

    expect(lease.sessionEpoch).toBe(7)
    expect(lease.sessionId).toBe('tui:abc')
  })
})

describe('SelectorList', () => {
  const rows = [
    { id: 'gpt-5', kind: 'item' as const, label: 'GPT-5', searchText: 'gpt-5 openai' },
    { id: 'claude-opus-5', kind: 'item' as const, label: 'Claude Opus 5', searchText: 'claude-opus-5 anthropic' },
    { id: 'action:add', kind: 'action' as const, label: 'Add provider…', searchText: 'add provider' }
  ]

  function list(extra: Partial<SelectorListOptions> = {}) {
    const harness = createHarness()
    const picked: string[] = []
    const cancelled: number[] = []
    const view = new SelectorList({
      keybindings: harness.deps.keybindings,
      onCancel: () => cancelled.push(1),
      onSelect: id => picked.push(id),
      rows,
      theme: harness.deps.theme,
      title: 'Model',
      tui: harness.tui,
      ...extra
    })

    return { cancelled, harness, picked, view }
  }

  it('tints the row under the cursor and nothing else', () => {
    // Rule one of the palette: violet is identity. In a picker the identity is
    // which row you are on, so the title, the hints and the unselected rows
    // take the terminal's own foreground and the greys.
    const colour = new Theme('dark', 3)
    const accent = colour.fg('accent', 'X').split('X')[0]!
    const { harness } = list()
    const view = new SelectorList({
      keybindings: harness.deps.keybindings,
      onCancel: () => {},
      onSelect: () => {},
      rows,
      theme: colour,
      title: 'Model',
      tui: harness.tui
    })
    const painted = view.render(80)
    const tinted = painted.filter(line => line.includes(accent))

    expect(tinted).toHaveLength(1)
    expect(stripTerminalSequences(tinted[0]!)).toContain(rows[0]!.label)
    expect(painted.find(line => stripTerminalSequences(line).includes('Model'))).not.toContain(accent)
  })

  it('matches in the middle of a value, which a prefix filter would not', () => {
    const { harness, view } = list()

    view.setQuery('opus')

    const drawn = harness.draw(view)

    expect(drawn).toContain('Claude Opus 5')
    expect(drawn).not.toContain('GPT-5')
  })

  it('keeps the selected row when the data is replaced', () => {
    const { view } = list()

    view.handleInput(KEYS.down)
    expect(view.selectedId).toBe('claude-opus-5')

    view.setRows([rows[2]!, rows[1]!, rows[0]!])
    expect(view.selectedId).toBe('claude-opus-5')
  })

  it('falls back to the first match when the selection is filtered away', () => {
    const { view } = list()

    view.handleInput(KEYS.down)
    view.setQuery('gpt')
    expect(view.selectedId).toBe('gpt-5')
  })

  it('treats d, j and k as search text rather than commands', () => {
    const { picked, view } = list()

    view.handleInput('d')
    view.handleInput('j')

    expect(view.query).toBe('dj')
    expect(picked).toHaveLength(0)
  })

  it('does nothing on Enter with no match', () => {
    const { picked, view } = list()

    view.setQuery('zzzz')
    view.handleInput(KEYS.enter)

    expect(picked).toHaveLength(0)
    expect(view.selectedId).toBeNull()
  })

  it('pages by a screenful and clamps at both ends', () => {
    const many = Array.from({ length: 30 }, (_, index) => ({
      id: `m${index}`,
      kind: 'item' as const,
      label: `model ${index}`,
      searchText: `model ${index}`
    }))
    const { harness, view } = list()

    view.setRows(many)
    view.handleInput(KEYS.pageDown)
    expect(view.selectedId).toBe(`m${visibleRowCount(harness.terminal.rows)}`)

    view.handleInput(KEYS.pageUp)
    view.handleInput(KEYS.pageUp)
    expect(view.selectedId).toBe('m0')
  })

  it('cancels on escape and selects on enter', () => {
    const { cancelled, picked, view } = list()

    view.handleInput(KEYS.enter)
    expect(picked).toEqual(['gpt-5'])

    view.handleInput(KEYS.escape)
    expect(cancelled).toHaveLength(1)
  })

  it('draws inside the width it is given', () => {
    const { harness, view } = list()

    for (const line of view.render(30)) {
      expect(line.length).toBeLessThanOrEqual(200)
    }

    expect(
      harness
        .draw(view, 30)
        .split('\n')
        .every(line => line.length <= 30)
    ).toBe(true)
  })

  it('keeps ten rows at most, and one on a very short terminal', () => {
    expect(visibleRowCount(40)).toBe(10)
    expect(visibleRowCount(14)).toBe(4)
    expect(visibleRowCount(3)).toBe(1)
  })
})
