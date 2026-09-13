// The busy indicator: the faces it wears, the row it draws, and what it says
// while a turn runs.

import { Container, stripTerminalSequences, TuiAltScreen, TuiMainScreen, visibleWidth } from '@earendil-works/pi-tui'
import { describe, expect, it } from 'vitest'

import { HarnessApp } from '../app.js'
import { createChatViewport } from '../chatViewport.js'
import { AssistantMessage } from '../components/assistantMessage.js'
import { HarnessEditor } from '../components/editor.js'
import { MessageQueue } from '../components/queue.js'
import {
  formatElapsed,
  glimmerAt,
  inBand,
  indicatorOptions,
  SHIMMER_BAND,
  SHIMMER_LEAD,
  SHIMMER_MS,
  StatusIndicator,
  StatusLine
} from '../components/statusIndicator.js'
import { UserMessage } from '../components/userMessage.js'
import { FACES } from '../content/faces.js'
import { VERBS } from '../content/verbs.js'
import { Gateway } from '../gateway.js'
import { QUIT_ARM_MS, QUIT_HINT } from '../lib/keybindings.js'
import { Theme } from '../theme.js'
import { FakeTerminal, FakeTransport } from './fakes.js'
import { ManualClock } from './selectorHarness.js'

const theme = new Theme('dark', 0)

/** An indicator in its line, on a fake terminal, with a clock the test moves. */
function indicator(message = 'folding…', kind?: 'retry') {
  const now = { ms: 1_000_000 }
  const clock = new ManualClock(now.ms)
  const tui = new TuiMainScreen(new FakeTerminal(80, 24))
  const line = new StatusLine({ clock, tui })
  const status = new StatusIndicator(tui, theme, {
    hint: () => 'esc to interrupt',
    message,
    now: () => now.ms,
    ...(kind ? { kind } : {})
  })

  line.set(status)

  return {
    advance: (ms: number) => {
      now.ms += ms
      clock.advance(ms)
    },
    clock,
    line,
    // The content rows only. The blank either side of the row is spacing, and
    // it has its own test rather than an index in every other one.
    rendered: (width = 60) =>
      line
        .render(width)
        .map(row => stripTerminalSequences(row))
        .filter(row => row.length > 0),
    status,
    terminal: tui.terminal as FakeTerminal
  }
}

describe('the faces', () => {
  it('wears the product’s own kaomoji, padded to one width', () => {
    const options = indicatorOptions()

    expect(options.frames?.[0]).toContain(FACES[0])
    expect(options.frames).toHaveLength(FACES.length)
    // Padded, so a narrower face does not shift the verb beside it.
    expect(new Set(options.frames?.map(frame => visibleWidth(frame))).size).toBe(1)
    // Slow: a face that flickers cannot be read.
    expect(options.intervalMs).toBeGreaterThan(1000)
  })
})

describe('the status line above the editor', () => {
  it("holds two blank rows when nothing is running, as pi's idle status does", () => {
    const tui = new TuiMainScreen(new FakeTerminal(80, 24))
    const line = new StatusLine({ clock: new ManualClock(), tui })

    // Spaces, not empty strings: they stand in for what was drawn there.
    expect(line.render(80)).toEqual([' '.repeat(80), ' '.repeat(80)])
    expect(line.render(40)).toEqual([' '.repeat(40), ' '.repeat(40)])
    expect(line.active).toBe(false)
  })

  it('keeps its height whether or not a turn is running, so nothing below it shifts', () => {
    const { line } = indicator()
    const rows = line.render(80)

    // pi's shape: a blank row, then the line. Against the transcript above, an
    // unspaced row would read as part of it.
    expect(rows).toHaveLength(2)
    expect(rows[0]).toBe('')
    expect(stripTerminalSequences(rows[1]!).trim()).not.toBe('')

    line.set(undefined)

    // Idle, the two rows stay: the editor does not move when a turn ends.
    expect(line.render(80)).toHaveLength(2)
    expect(line.render(80).join('').trim()).toBe('')
  })

  it('stands the activity verb beside the face, with the clock and the way out', () => {
    const { advance, rendered } = indicator()

    advance(12_000)

    const [row] = rendered()

    expect(row).toContain(FACES[0])
    expect(row).toContain('folding…')
    expect(row).toContain('(12s · esc to interrupt)')
  })

  it('reads the clock in minutes once a turn is long', () => {
    const { advance, rendered } = indicator()

    advance(125_000)
    expect(rendered()[0]).toContain('(2m 05s')
    expect(formatElapsed(3_725)).toBe('1h 2m')
  })

  it('says why it is retrying rather than what it was doing', () => {
    const { rendered } = indicator('retrying 2/3', 'retry')

    expect(rendered()[0]).toContain('retrying 2/3')
  })

  it('draws one row inside the width, at every width', () => {
    const { advance, line, status } = indicator()

    advance(9_000)

    for (const width of [40, 80, 120]) {
      const rows = line.render(width)

      expect(rows).toHaveLength(2)
      expect(visibleWidth(rows[1]!)).toBeLessThanOrEqual(width)
    }

    status.dispose()
  })

  it('clips rather than wraps when the line has almost no room', () => {
    const { advance, line } = indicator()

    advance(9_000)

    for (const width of [1, 4, 12, 20]) {
      const rows = line.render(width)

      expect(rows).toHaveLength(2)
      expect(visibleWidth(rows[1]!)).toBeLessThanOrEqual(width)
    }
  })

  it('runs its clock for the whole turn, retries included', () => {
    const { advance, rendered, status } = indicator()

    advance(30_000)

    expect(rendered()[0]).toContain('30s')
    expect(status.elapsedSeconds).toBe(30)
  })

  it('shimmers like Claude Code: a three-character tint sweeps the face and the verb, then rests', () => {
    // The face and the verb are one run of characters. A band three wide, in
    // the accent's lighter step, moves one character per tick. It starts ten
    // characters before the run and ends ten past it, so most of every cycle
    // the band is off the text and the line rests; that pause is what reads
    // as gentle. Everything outside the band keeps the accent, the details
    // keep their muted paint, and the faces and verbs themselves are untouched.
    expect([0, SHIMMER_LEAD, SHIMMER_LEAD + 1].map(tick => glimmerAt(tick, 10))).toEqual([-SHIMMER_LEAD, 0, 1])
    expect(glimmerAt(10 + 2 * SHIMMER_LEAD, 10)).toBe(-SHIMMER_LEAD)
    expect([false, true, true, true, false]).toEqual([2, 3, 4, 5, 6].map(index => inBand(index, 4)))

    const theme = new Theme('dark', 3)
    const now = { ms: 1_000_000 }
    const status = new StatusIndicator(new TuiMainScreen(new FakeTerminal(80, 24)), theme, {
      hint: () => 'esc to interrupt',
      message: 'folding…',
      now: () => now.ms
    })
    // eslint-disable-next-line no-control-regex -- the tint is an SGR sequence
    const tinted = (line: string) => Array.from(line.matchAll(/\x1b\[38;2;169;122;255m(.)/g), match => match[1] ?? '')
    const at = () => status.renderLine(80)

    // At rest: the band is ten characters short of the run.
    expect(tinted(at())).toHaveLength(0)
    now.ms += SHIMMER_MS * (SHIMMER_LEAD + 1)
    expect(tinted(at())).toHaveLength(SHIMMER_BAND)
    // The run is `<face> folding…`; once past the face, the band is in the verb.
    const faceWidth = Array.from(indicatorOptions().frames?.[0] ?? '').length
    now.ms = 1_000_000 + SHIMMER_MS * (SHIMMER_LEAD + faceWidth + 2)
    expect(tinted(at()).join('')).toBe('fol')
    // And past the end it rests again before coming round.
    now.ms = 1_000_000 + SHIMMER_MS * (SHIMMER_LEAD + faceWidth + 1 + 'folding…'.length + 2)
    expect(tinted(at())).toHaveLength(0)
    const elapsed = formatElapsed((now.ms - 1_000_000) / 1000)
    expect(stripTerminalSequences(at())).toContain(`folding… (${elapsed} · esc to interrupt)`)
    expect(at()).toContain(theme.fg('muted', `(${elapsed} · esc to interrupt)`))
  })

  it('asks for a redraw on every shimmer step, so the clock is never stale', () => {
    const { advance, clock, line, status } = indicator()

    expect(clock.ticks).toHaveLength(1)
    advance(SHIMMER_MS)

    // And it gives the timer back when the turn ends.
    line.set(undefined)
    status.dispose()
    expect(clock.ticks).toHaveLength(0)
    expect(line.render(80).join('').trim()).toBe('')
  })
})

describe('the app while a turn runs', () => {
  const SCRIPT = {
    'commands.catalog': { categories: [], pairs: [], skill_count: 0 },
    'turn.send': { accepted: true, turn_id: 't1' },
    'config.get': { config: {} },
    'session.create': { info: { cwd: '/work', model: 'gpt-5', provider: 'openai' }, session_id: 'tui:abc' },
    'setup.status': { provider_configured: true },
    'system.hello': { server_capabilities: [], server_version: '9.9.9', session: {} }
  }

  function build(overrides: Record<string, unknown> = {}) {
    const transport = new FakeTransport({ ...SCRIPT, ...overrides })
    const terminal = new FakeTerminal(100, 30)
    const tui = new TuiMainScreen(terminal)
    const app = new HarnessApp({
      cwd: '/nonexistent',
      env: {},
      gateway: new Gateway(transport),
      listenResize: () => () => {},
      onExit: () => {},
      theme: new Theme('dark', 0),
      tui
    })

    tui.start()

    return {
      app,
      screen: () => stripTerminalSequences(terminal.output()),
      transport,
      tui
    }
  }

  it('stands one of the domain’s activity verbs in the editor’s border', async () => {
    const { app, screen, transport, tui } = build()

    await app.boot()
    await app.submit('design a VHH')
    transport.emit({ payload: { turn_id: 't1' }, type: 'message.start' })
    tui.renderNow()

    const drawn = screen()

    expect(VERBS.some(verb => drawn.includes(`${verb}…`))).toBe(true)
  })
})

/**
 * The screen as the alternate renderer actually painted it.
 *
 * Each row arrives as `CSI <row>;1H CSI 2K <content>`, so the frame is
 * reassembled from those rather than from a component's own `render`. The
 * spacing around the docked status row is a property of the whole layout —
 * the viewport pads between a short conversation and the dock — and a
 * component on its own cannot show it.
 */
function screen(output: string, rows: number): string[] {
  const painted = Array.from({ length: rows }, () => '')
  // eslint-disable-next-line no-control-regex
  const row = /\x1b\[(\d+);1H\x1b\[2K([^\x1b]*(?:\x1b(?!\[\d+;1H)[^\x1b]*)*)/g
  let match: null | RegExpExecArray

  while ((match = row.exec(output)) !== null) {
    const index = Number(match[1]) - 1

    if (index >= 0 && index < rows) {
      painted[index] = stripTerminalSequences(match[2] ?? '').replace(/\s+$/, '')
    }
  }

  return painted
}

/** The docked stack as the app mounts it, at a fixed terminal size. */
function docked(replies: number, rows: number): string[] {
  const terminal = new FakeTerminal(80, rows)
  const tui = new TuiAltScreen(terminal, false, undefined, { mouse: false })
  const chat = new Container()

  chat.addChild(new UserMessage(theme, 'design something'))

  for (let index = 0; index < replies; index += 1) {
    const reply = new AssistantMessage(theme, {})

    reply.appendText(`Reply ${index + 1}.`)
    reply.finish()
    chat.addChild(reply)
  }

  const document = new Container()

  document.addChild(chat)

  const queue = new MessageQueue({ actions: { interrupt: () => {}, send: () => {} }, isTurnActive: () => true, theme })
  const status = new StatusLine({ clock: new ManualClock(), tui })
  const now = { ms: 1_000_000 }

  status.set(new StatusIndicator(tui, theme, { hint: () => 'esc', message: 'folding…', now: () => now.ms }))
  now.ms += 3000

  const editor = new Container()

  editor.addChild(new HarnessEditor(tui, theme.editorTheme()))

  const footer = new Container()
  const taskBar = new Container()
  const viewport = createChatViewport({ document, editor, footer, queue: queue.view, status, taskBar, theme })

  for (const child of [document, taskBar, queue.view, status, editor, footer]) {
    tui.addChild(child)
  }

  tui.setLayoutRoot(viewport.root)
  tui.start()
  terminal.writes.length = 0
  tui.renderNow()
  tui.stop({ preserveScreen: true })

  return screen(terminal.output(), rows)
}

describe('the armed-quit hint', () => {
  it('takes the status row when nothing is running, and gives it up when the arm lapses', () => {
    const clock = new ManualClock(1_000_000)
    const tui = new TuiMainScreen(new FakeTerminal(80, 24))
    const line = new StatusLine({ clock, tui })

    expect(line.render(80).join('').trim()).toBe('')

    line.setNotice(theme.fg('muted', QUIT_HINT), QUIT_ARM_MS)

    const shown = line.render(80)

    expect(stripTerminalSequences(shown[1] ?? '')).toBe(QUIT_HINT)
    // Spaced like the busy row, so arming does not shift the editor.
    expect(shown).toHaveLength(2)
    expect(shown[0]).toBe('')

    // Nothing else redraws an idle screen, so the row has to clear itself.
    clock.advance(QUIT_ARM_MS)
    expect(line.render(80).join('').trim()).toBe('')
  })

  it('shows over an activity row, and gives it back when it lapses', () => {
    // A gateway slash command borrows this row without being a turn, so the
    // ladder reads an idle prompt and arms. With the activity on top the offer
    // was made invisibly and the next press exited having shown nothing.
    const clock = new ManualClock(1_000_000)
    const tui = new TuiMainScreen(new FakeTerminal(80, 24))
    const line = new StatusLine({ clock, tui })

    line.set(new StatusIndicator(tui, theme, { message: 'doctor…', now: () => 1_000_000 }))
    expect(stripTerminalSequences(line.render(80)[1] ?? '')).toContain('doctor…')

    line.setNotice(theme.fg('muted', QUIT_HINT), QUIT_ARM_MS)
    expect(stripTerminalSequences(line.render(80)[1] ?? '')).toBe(QUIT_HINT)

    // The command is still running and takes its row back.
    clock.advance(QUIT_ARM_MS)
    expect(stripTerminalSequences(line.render(80)[1] ?? '')).toContain('doctor…')
  })
})

describe('the rows around the docked status line', () => {
  /** Where the status row landed, and what is immediately around it. */
  const around = (painted: string[]) => {
    const at = painted.findIndex(line => line.includes('folding…'))

    expect(at).toBeGreaterThan(0)

    return { above: painted[at - 1]!, rule: painted[at + 1]!, at }
  }

  it("leaves one blank between the last reply and the row, and the editor's rule under it", () => {
    // Tall enough that the conversation fills the viewport, so every blank on
    // screen is one a component asked for.
    const painted = docked(8, 14)
    const { above, rule, at } = around(painted)

    expect(painted[at - 2]).toContain('Reply')
    expect(above).toBe('')
    // pi's shape: the row sits directly on the editor's rule.
    expect(rule.startsWith('─')).toBe(true)
  })

  it('adds nothing to the gap a short conversation already leaves', () => {
    const painted = docked(2, 18)
    const { rule, at } = around(painted)

    // The dock is pinned to the bottom, so the viewport pads above it. A row
    // the status line adds is a row the viewport stops padding with: the gap
    // is the same either way, which is why the blank above earns its place in
    // the full-viewport case above and costs nothing here.
    expect(rule.startsWith('─')).toBe(true)
    expect(
      painted
        .slice(0, at)
        .filter(line => line !== '')
        .at(-1)
    ).toContain('Reply 2')
  })
})
