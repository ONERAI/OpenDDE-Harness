// The long-run reassurance row inside a tool panel, on a clock the test drives.

import { stripTerminalSequences, TuiMainScreen } from '@earendil-works/pi-tui'
import { describe, expect, it } from 'vitest'

import { ToolExecution } from '../components/toolExecution.js'
import { waitPhrases } from '../content/charms.js'
import { Theme } from '../theme.js'
import { FakeTerminal } from './fakes.js'
import { ManualClock } from './selectorHarness.js'

const theme = new Theme('dark', 0)

function build(name = 'run_command') {
  const clock = new ManualClock(1_000_000)
  const tui = new TuiMainScreen(new FakeTerminal(80, 24))
  const tool = new ToolExecution(
    theme,
    tui,
    { name },
    // Always the first candidate, so the wording is the test's to assert on.
    { choose: options => options[0]!, clock }
  )

  const text = () =>
    tool
      .render(80)
      .map(line => stripTerminalSequences(line))
      .join('\n')

  return { clock, text, tool }
}

describe('long-run reassurance', () => {
  it('says nothing before eight seconds and one thing after', () => {
    const { clock, text } = build()

    clock.advance(7_999)
    expect(text()).not.toContain('waiting')

    clock.advance(1)
    expect(text()).toContain('waiting 8s')
  })

  it('rewrites that one row at eighteen seconds and never again', () => {
    const { clock, text } = build()

    clock.advance(8_000)

    const first = text()

    clock.advance(9_999)
    expect(text()).toBe(first)

    clock.advance(1)

    const second = text()

    expect(second).toContain('waiting 18s')
    expect(second).not.toBe(first)

    clock.advance(120_000)
    expect(text()).toBe(second)
  })

  it('keeps exactly one reassurance row however long it waits', () => {
    const { clock, tool } = build()

    clock.advance(60_000)

    const lines = tool.render(80).map(line => stripTerminalSequences(line))

    expect(lines.filter(line => line.includes('waiting'))).toHaveLength(1)
  })

  it('shows real progress above the reassurance, and never resets the age on it', () => {
    const { clock, text, tool } = build()

    clock.advance(7_000)
    tool.setProgress('folded 3 of 12')
    clock.advance(1_000)

    const lines = text()
      .split('\n')
      .filter(line => line.trim())

    expect(lines.some(line => line.includes('folded 3 of 12'))).toBe(true)
    expect(text()).toContain('waiting 8s')
  })

  it('goes straight to the row that is due when the process was suspended', () => {
    const { clock, text } = build()

    // An external editor, or a stopped terminal: time moved on without a tick.
    clock.advance(30_000)

    expect(text()).toContain('waiting 30s')
    expect(
      text()
        .split('\n')
        .filter(line => line.includes('waiting'))
    ).toHaveLength(1)
  })

  it('stops at completion, and a call that completes first never says anything', () => {
    const { clock, text, tool } = build()

    clock.advance(7_999)
    tool.complete({ result_preview: 'done', truncated: false })
    clock.advance(60_000)

    expect(text()).not.toContain('waiting')
    // The reassurance row gave way to the completed call: collapsed and quiet,
    // that is the invocation line and the expand hint.
    expect(tool.isPending).toBe(false)
    expect(text()).toContain('to expand')
  })

  it('stops when the turn abandons the call', () => {
    const { clock, text, tool } = build()

    clock.advance(8_000)
    expect(text()).toContain('waiting')

    tool.abandon('cancelled')
    clock.advance(60_000)

    expect(text()).not.toContain('waiting')
    expect(text()).toContain('cancelled')
  })

  it('releases its timer when it is disposed', () => {
    const clock = new ManualClock()
    const tui = new TuiMainScreen(new FakeTerminal(80, 24))
    const tool = new ToolExecution(theme, tui, { name: 'x' }, { clock })

    expect(clock.ticks.length).toBe(1)
    tool.dispose()
    expect(clock.ticks.length).toBe(0)
  })

  it('runs two calls independently', () => {
    const clock = new ManualClock(1_000_000)
    const tui = new TuiMainScreen(new FakeTerminal(80, 24))
    const first = new ToolExecution(theme, tui, { name: 'a' }, { choose: options => options[0]!, clock })

    clock.advance(8_000)

    const second = new ToolExecution(theme, tui, { name: 'b' }, { choose: options => options[0]!, clock })

    clock.advance(10_000)

    const draw = (tool: ToolExecution) =>
      tool
        .render(80)
        .map(line => stripTerminalSequences(line))
        .join('\n')

    // Each call counts from its own start: the first has reached its second
    // row, the second is still on its first.
    expect(draw(first)).toContain('waiting 18s')
    expect(draw(second)).toContain('waiting 10s')
  })

  it('has no reassurance at all without a clock', () => {
    const tui = new TuiMainScreen(new FakeTerminal(80, 24))
    const tool = new ToolExecution(theme, tui, { name: 'x' }, { now: () => 1_000_000 })

    expect(tool.render(80).join('\n')).not.toContain('waiting')
  })
})

describe('reassurance wording', () => {
  it('claims a domain only when the tool or its progress says one', () => {
    expect(waitPhrases('run_command')).toEqual(['still working', 'still running'])
    expect(waitPhrases('boltz_fold')[0]).toBe('still folding')
    expect(waitPhrases('rank_designs')[0]).toBe('scoring candidates')
    expect(waitPhrases('run_command', 'leasing a gpu on the compute service')[0]).toBe('waiting on the compute service')
  })
})
