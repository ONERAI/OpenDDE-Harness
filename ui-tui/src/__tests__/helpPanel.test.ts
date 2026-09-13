// /help: what it lists, and how it behaves at a narrow width.

import { TuiMainScreen, visibleWidth } from '@earendil-works/pi-tui'
import { describe, expect, it } from 'vitest'

import { HelpPanel, wrapToWidth } from '../components/helpPanel.js'
import { createKeybindings } from '../lib/keybindings.js'
import { Theme } from '../theme.js'
import { FakeTerminal } from './fakes.js'

const theme = new Theme('dark', 0)

// eslint-disable-next-line no-control-regex
const ANSI = /\u001b\[[0-9;]*m/g

function panel(overrides: Partial<ConstructorParameters<typeof HelpPanel>[2]> = {}, userBindings = {}) {
  return new HelpPanel(theme, createKeybindings(userBindings), {
    catalog: ['compare', 'doctor'],
    commands: [
      { argumentHint: '<level>', description: "set the model's thinking level", name: 'thinking' },
      { description: 'exit OpenDDE Harness', name: 'quit' }
    ],
    hotkeys: ['app.editor.external', 'app.reset'],
    skillCount: 4,
    ...overrides
  })
}

function lines(component: HelpPanel, width = 80): string {
  return (
    component
      .render(width)
      .join('\n')
      // The panel paints through the theme; strip whatever it emitted.
      .replace(ANSI, '')
  )
}

/** The identity violets, as this palette paints them. A colour test needs a
 *  tier that actually emits sequences, which the panel's own theme does not. */
const colour = new Theme('dark', 3)
const IDENTITY = ['accent', 'primary', 'prompt'].map(token => colour.fg(token as 'accent', 'X').split('X')[0]!)

describe('what /help tints', () => {
  it('paints no identity colour on a reference list', () => {
    // Headings, command names and key names are all chrome: a list of what
    // exists is not the brand. Violet here is what made the screen look heavy.
    const painted = new HelpPanel(colour, createKeybindings({}), {
      catalog: ['compare'],
      commands: [{ description: 'exit', name: 'quit' }],
      hotkeys: ['app.reset'],
      skillCount: 1
    })
      .render(80)
      .join('\n')

    for (const sequence of IDENTITY) {
      expect(painted).not.toContain(sequence)
    }

    // Still a two-column list: the name in the terminal's own foreground and
    // the description in grey, rather than everything flattened into one tone.
    expect(painted).toContain(colour.fg('muted', 'exit'))
  })
})

describe('HelpPanel', () => {
  it('lists local commands with their argument hints', () => {
    expect(lines(panel())).toContain('/thinking <level>')
    expect(lines(panel())).toContain("set the model's thinking level")
  })

  it('lists the gateway catalog and the skill count', () => {
    const text = lines(panel())

    expect(text).toContain('/compare')
    expect(text).toContain('4 skill commands')
  })

  it('shows the keys as they are actually bound', () => {
    expect(lines(panel())).toContain('ctrl+g/alt+g')
    expect(lines(panel({}, { 'app.editor.external': 'ctrl+x' }))).toContain('ctrl+x')
  })

  it('stacks the description under the name on a narrow terminal', () => {
    const narrow = lines(panel(), 30).split('\n')
    const nameLine = narrow.findIndex(line => line.includes('/thinking <level>'))

    expect(nameLine).toBeGreaterThanOrEqual(0)
    // Wrapped onto the lines after the name, rather than clipped beside it.
    expect(
      narrow
        .slice(nameLine + 1, nameLine + 3)
        .join(' ')
        .replace(/\s+/g, ' ')
    ).toContain("set the model's thinking level")
  })

  it('never draws wider than the viewport', () => {
    for (const width of [24, 40, 80, 200]) {
      for (const line of lines(panel(), width).split('\n')) {
        expect(line.length).toBeLessThanOrEqual(width)
      }
    }
  })

  it('is a plain component: it has no focus and no input', () => {
    const component = panel() as unknown as { focused?: unknown; handleInput?: unknown }

    expect(component.handleInput).toBeUndefined()
    expect(component.focused).toBeUndefined()
  })

  it('breaks a word wider than the line rather than overflowing', () => {
    // No spaces to wrap on, and two cells per character.
    const long = '这是一个非常长的中文说明文字没有任何空格可以用来换行'

    for (const line of wrapToWidth(long, 40)) {
      expect(visibleWidth(line)).toBeLessThanOrEqual(40)
    }

    expect(wrapToWidth(long, 40).join('')).toBe(long)
  })

  it('keeps every line inside 40 columns with unbroken text', () => {
    const panel = new HelpPanel(theme, createKeybindings(), {
      catalog: ['a-very-long-catalogue-command-name-that-never-breaks-on-a-space'],
      commands: [
        {
          argumentHint: '<参数>',
          description: '这是一个非常长的中文说明文字没有任何空格可以用来换行所以必须按显示宽度切开',
          name: 'thinking'
        }
      ],
      hotkeys: ['app.editor.external'],
      skillCount: 4
    })

    for (const line of panel.render(40)) {
      expect(visibleWidth(line)).toBeLessThanOrEqual(40)
    }
  })

  it('renders through the real renderer at 40 columns without throwing', () => {
    const terminal = new FakeTerminal(40, 24)
    const tui = new TuiMainScreen(terminal)

    tui.addChild(
      new HelpPanel(theme, createKeybindings(), {
        catalog: ['another-extremely-long-unbreakable-catalogue-command-name'],
        commands: [{ description: '这是一个没有空格的很长的中文描述需要按宽度切开才能渲染', name: 'quit' }],
        hotkeys: ['app.reset'],
        skillCount: 2
      })
    )

    // The renderer refuses a line wider than the terminal; this is the check.
    expect(() => tui.renderNow(true)).not.toThrow()
  })
})
