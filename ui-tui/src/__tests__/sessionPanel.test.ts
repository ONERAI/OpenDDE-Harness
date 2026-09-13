import { TuiMainScreen, visibleWidth } from '@earendil-works/pi-tui'
import { describe, expect, it } from 'vitest'

import { SessionPanel } from '../components/sessionPanel.js'
import { systemLine } from '../components/systemLine.js'
import { Theme } from '../theme.js'
import { FakeTerminal } from './fakes.js'

const DATA = {
  commandCount: 41,
  model: 'gpt-5',
  provider: 'openai',
  reasoningEffort: 'high',
  sessionId: 'tui:abc123',
  skillCount: 7,
  version: '0.4.2'
}

describe('SessionPanel', () => {
  it('shows the model and the session, and leaves the path to the footer', () => {
    const panel = new SessionPanel(new Theme('dark', 0), DATA)
    const text = panel.render(100).join('\n')

    expect(text).toContain('OpenDDE Harness v0.4.2')
    expect(text).toContain('gpt-5')
    expect(text).toContain('openai')
    expect(text).toContain('high')
    expect(text).toContain('tui:abc123')
    expect(text).toContain('41 commands')
    expect(text).toContain('7 skills')
  })

  it('counts in the singular when there is one of something', () => {
    const panel = new SessionPanel(new Theme('dark', 0), { commandCount: 1, skillCount: 1 })

    expect(panel.render(60).join('\n')).toContain('1 command · 1 skill')
  })

  it('leaves out fields the gateway did not send', () => {
    const panel = new SessionPanel(new Theme('dark', 0), { sessionId: 'tui:abc123' })
    const lines = panel.render(60)

    expect(lines.join('\n')).toContain('tui:abc123')
    expect(lines.join('\n')).not.toContain('undefined')
  })

  it('keeps every line inside the width it was given', () => {
    const panel = new SessionPanel(new Theme('dark', 3), DATA)

    for (const width of [20, 32, 48, 80, 120]) {
      for (const line of panel.render(width)) {
        expect(visibleWidth(line)).toBeLessThanOrEqual(width)
      }
    }
  })

  it('caches per width and rebuilds after an update', () => {
    const panel = new SessionPanel(new Theme('dark', 3), DATA)
    const first = panel.render(80)

    expect(panel.render(80)).toBe(first)
    expect(panel.render(60)).not.toBe(first)

    panel.update({ model: 'claude-opus-5' })

    expect(panel.render(80)).not.toBe(first)
    expect(panel.render(80).join('\n')).toContain('claude-opus-5')
  })
})

describe('headless frame', () => {
  it('renders the panel and a system line to a fake terminal', () => {
    const terminal = new FakeTerminal(72, 20)
    const tui = new TuiMainScreen(terminal)
    const theme = new Theme('dark', 3)

    tui.addChild(new SessionPanel(theme, DATA))
    tui.addChild(systemLine(theme, 'read path=/etc/hosts'))
    tui.renderNow()

    const output = terminal.output()

    expect(terminal.byteCount).toBeGreaterThan(0)
    expect(output).toContain('tui:abc123')
    expect(output).toContain('read')
    // Colors reached the wire at truecolor: the primary is rgb(167,139,250).
    expect(output).toContain('38;2;167;139;250')
  })

  it('collapses a system line onto one row', () => {
    const theme = new Theme('dark', 0)
    const line = systemLine(theme, 'a very long tool argument '.repeat(20))

    // pi's blank row, then the note itself on one row however long it is.
    expect(line.render(40)).toHaveLength(2)
    expect(visibleWidth(line.render(40)[1]!)).toBeLessThanOrEqual(40)
  })

  it('counts the real inventory the session served, not the catalog guess', () => {
    const panel = new SessionPanel(new Theme('dark', 0), {
      ...DATA,
      skills: { design: ['relax', 'dock'] },
      tools: { builtin: ['read', 'write'], mcp: ['fold'] }
    })
    const text = panel.render(100).join('\n')

    expect(text).toContain('3 tools')
    expect(text).toContain('2 skills')
    // The catalog's stand-in count is gone once the real one exists.
    expect(text).not.toContain('7 skills')
  })

  it('counts what is loaded without naming any of it', () => {
    const panel = new SessionPanel(new Theme('dark', 0), {
      ...DATA,
      skills: { design: ['relax'] },
      tools: { builtin: ['write', 'read'], mcp: ['fold'] }
    })
    const lines = panel.render(100)
    const text = lines.join('\n')

    expect(text).toContain('3 tools')
    expect(text).toContain('1 skill')
    expect(text).not.toContain('read')
    expect(text).not.toContain('relax')

    for (const line of lines) {
      expect(visibleWidth(line)).toBeLessThanOrEqual(100)
    }
  })
})

describe('the panel and a palette that moves under it', () => {
  it('repaints its facts when the scheme changes after it was first rendered', () => {
    const theme = new Theme('dark', 3)
    const panel = new SessionPanel(theme, DATA, { rows: () => 40 })
    const dark = panel.render(100).join('\n')

    expect(theme.setScheme('light')).toBe(true)

    const light = panel.render(100).join('\n')

    expect(light).not.toBe(dark)
    // `muted`, one value per palette.
    expect(dark).toContain('38;2;153;153;153')
    expect(light).toContain('38;2;102;102;102')
    expect(light).not.toContain('38;2;153;153;153')
  })
})
