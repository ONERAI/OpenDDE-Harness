// The brand lockup: the art the first screen opens with, and what it does when
// the terminal has no room for it.

import { stripTerminalSequences, TuiMainScreen, visibleWidth } from '@earendil-works/pi-tui'
import { describe, expect, it } from 'vitest'

import type { ColorTier } from '../lib/colorTier.js'
import type { McpServerInfo } from '../rpc/index.js'

import { HarnessApp } from '../app.js'
import { Branding, lockupSize, resolveLockupScale, scaleArt, WORDMARK_ART } from '../components/branding.js'
import { SessionPanel } from '../components/sessionPanel.js'
import { Gateway } from '../gateway.js'
import { Theme } from '../theme.js'
import { FakeTerminal, FakeTransport } from './fakes.js'

/** Every terminal-height-dependent test says what height it means: the art is
 *  sized against the window, and a test that inherited one would drift. */
function brand(width: number, rows = 40, tier: ColorTier = 3, version = '0.4.2'): string[] {
  return new Branding(new Theme('dark', tier), { rows: () => rows, version }).render(width)
}

const plain = (lines: string[]) => lines.map(line => stripTerminalSequences(line))

/** Rows carrying block-drawing characters — the art, as opposed to the text. */
const artRows = (lines: string[]) => plain(lines).filter(line => /[█▀▄]/.test(line))

describe('the brand lockup', () => {
  it('draws the art above the product name when there is room', () => {
    const lines = brand(80)

    expect(artRows(lines).length).toBeGreaterThan(6)
    expect(stripTerminalSequences(lines.at(-1)!)).toBe('ϒ  OpenDDE Harness v0.4.2')
  })

  it('keeps the name and drops the art on a narrow terminal', () => {
    const lines = brand(40)

    expect(artRows(lines)).toHaveLength(0)
    expect(lines).toHaveLength(1)
    expect(stripTerminalSequences(lines[0]!)).toBe('ϒ  OpenDDE Harness v0.4.2')
  })

  it('drops the art on a short terminal however wide it is', () => {
    expect(artRows(brand(120, 8))).toHaveLength(0)
    expect(artRows(brand(120, 40)).length).toBeGreaterThan(0)
  })

  it('stays inside the width it was given, at every width and every tier', () => {
    for (const tier of [0, 1, 2, 3] as ColorTier[]) {
      for (const width of [40, 80, 120]) {
        for (const line of brand(width, 40, tier)) {
          expect(visibleWidth(line)).toBeLessThanOrEqual(width)
        }
      }
    }
  })

  it('draws the wordmark only at a size its letters survive', () => {
    // Full size keeps every stroke; half keeps the strokes and the counters,
    // which is what the letters are. A quarter keeps neither, so at that size
    // it is not drawn at all and no width may show it.
    const legible = [1, 2].flatMap(factor =>
      scaleArt([...WORDMARK_ART], factor)
        .map(row => row.trim())
        .filter(Boolean)
    )

    for (let width = 20; width <= 160; width++) {
      const factor = resolveLockupScale(40, width)
      // Where the hero ends at whatever size this width draws it.
      const gutter = factor === 0 ? width : lockupSize(factor).heroCols

      for (const row of plain(brand(width, 40))) {
        // Anything to the right of the hero is the wordmark, and it has to be a
        // row of one of the two sizes that read.
        const right = row.slice(gutter).trim()

        if (right && /[█▀▄]/.test(right)) {
          expect({ right, width }).toMatchObject({ right: legible.find(known => known === right) })
        }
      }
    }
  })

  it('shows the wordmark only once the terminal is wide enough for it', () => {
    const hasWordmark = (width: number) => {
      const factor = resolveLockupScale(40, width)

      return factor !== 0 && plain(brand(width, 40)).some(row => /[█▀▄]/.test(row.slice(lockupSize(factor).heroCols)))
    }

    // Half size is hero, a four-column gap and wordmark: 67 columns.
    expect(lockupSize(2).cols).toBe(67)
    expect(hasWordmark(66)).toBe(false)
    expect(hasWordmark(67)).toBe(true)
    expect(hasWordmark(120)).toBe(true)

    // Below that the hero stands alone and keeps shrinking, and the name line
    // carries the words at every width.
    expect(artRows(brand(64, 40)).length).toBeGreaterThan(4)
    expect(stripTerminalSequences(brand(64, 40).at(-1)!)).toContain('OpenDDE Harness')
  })

  it('paints the wordmark as a gradient at truecolor and as one color at 16', () => {
    // Wide enough for the wordmark, which is the only thing with a gradient.
    // Foreground colors only: bold and reset carry no palette information.
    const colors = (lines: string[]) =>
      // No escape character in the pattern: nothing else the lockup draws has a `[` in it.
      new Set(lines.join('\n').match(/\[(?:38;[25];[0-9;]+|[39][0-7])m/g) ?? [])

    // Four ramp bands plus the hero's own token, and never a bare escape at
    // tier 0 — chalk is pinned to the tier, so this is the palette's doing.
    expect(colors(brand(140, 40, 3)).size).toBeGreaterThanOrEqual(4)
    expect(colors(brand(140, 40, 2)).size).toBeGreaterThanOrEqual(4)
    expect(colors(brand(140, 40, 1)).size).toBeLessThanOrEqual(3)
    expect(colors(brand(140, 40, 0)).size).toBe(0)
  })

  it('repaints when the palette is swapped under it', () => {
    const theme = new Theme('dark', 3)
    const branding = new Branding(theme, { rows: () => 40 })
    const before = branding.render(80).join('\n')

    theme.setScheme('light')
    branding.invalidate()

    expect(branding.render(80).join('\n')).not.toBe(before)
  })

  it('caches per width and height, and rebuilds when the version arrives', () => {
    const branding = new Branding(new Theme('dark', 3), { rows: () => 40 })
    const first = branding.render(80)

    expect(branding.render(80)).toBe(first)
    expect(branding.render(120)).not.toBe(first)

    branding.setVersion('1.2.3')

    expect(stripTerminalSequences(branding.render(80).at(-1)!)).toContain('v1.2.3')
  })

  it('reserves room for the wordmark only where it is drawn', () => {
    for (const factor of [1, 2]) {
      expect(lockupSize(factor).cols).toBeGreaterThan(lockupSize(factor).heroCols)
    }

    // A quarter-size hero stands alone, so its footprint is its own width.
    expect(lockupSize(4).cols).toBe(lockupSize(4).heroCols)
    expect(lockupSize(2).cols).toBeGreaterThan(lockupSize(4).cols)
  })

  it('picks the largest scale that fits, and none when nothing does', () => {
    expect(resolveLockupScale(40, 200)).toBe(1)
    // Too narrow for the full-size lockup, so the half-size one.
    expect(resolveLockupScale(40, 80)).toBe(2)
    // Too short for the full-size hero, whatever the width.
    expect(resolveLockupScale(16, 200)).toBe(2)
    // Too narrow even for the half-size lockup: the hero alone.
    expect(resolveLockupScale(40, 66)).toBe(4)
    expect(resolveLockupScale(8, 200)).toBe(4)
    expect(resolveLockupScale(2, 200)).toBe(0)
  })

  it('halves block art through half-block characters', () => {
    expect(scaleArt(['██', '██'], 1)).toEqual(['██', '██'])
    expect(scaleArt(['██', '██'], 2)).toEqual(['█'])
    expect(scaleArt(['██', '  '], 2)).toEqual(['▀'])
    expect(scaleArt(['  ', '██'], 2)).toEqual(['▄'])
    expect(scaleArt(['  ', '  '], 2)).toEqual([''])
  })
})

describe('the welcome panel', () => {
  const DATA = {
    model: 'gpt-5',
    provider: 'openai',
    releaseDate: '2026-09-10',
    sessionId: 'tui:abc123',
    version: '0.4.2'
  }

  const MCP: McpServerInfo[] = [
    { connected: true, name: 'files', tool_count: 3, transport: 'stdio' },
    { connected: false, name: 'search', tool_count: 0, transport: 'sse' }
  ]

  it('dates the version on the lockup, and drops the date without one', () => {
    const dated = new SessionPanel(new Theme('dark', 0), DATA, { rows: () => 40 })

    expect(plain(dated.render(100)).join('\n')).toContain('OpenDDE Harness v0.4.2 (2026-09-10)')

    // An install whose changelog records no date says the version alone, and a
    // date with no version to attach it to is not shown at all.
    const undated = new SessionPanel(new Theme('dark', 0), { ...DATA, releaseDate: null }, { rows: () => 40 })
    const versionless = new SessionPanel(new Theme('dark', 0), { releaseDate: '2026-09-10' }, { rows: () => 40 })

    expect(plain(undated.render(100)).join('\n')).toContain('OpenDDE Harness v0.4.2')
    expect(plain(undated.render(100)).join('\n')).not.toContain('2026-09-10')
    expect(plain(versionless.render(100)).join('\n')).not.toContain('2026-09-10')
  })

  it('picks up the date when the gateway answers after the first render', () => {
    const panel = new SessionPanel(new Theme('dark', 0), {}, { rows: () => 40 })

    panel.render(100)
    panel.update({ releaseDate: '2026-09-10', version: '0.4.2' })

    expect(plain(panel.render(100)).join('\n')).toContain('OpenDDE Harness v0.4.2 (2026-09-10)')
  })

  it('counts the MCP servers without naming them', () => {
    const panel = new SessionPanel(new Theme('dark', 0), { ...DATA, mcpServers: MCP }, { rows: () => 40 })
    const text = plain(panel.render(100)).join('\n')

    expect(text).toContain('2 MCP servers')
    // A count, not a listing: the panel says what is loaded, not what each of
    // them is called.
    expect(text).not.toContain('[stdio]')
    expect(text).not.toContain('files')
  })

  it('says nothing about MCP when none is configured', () => {
    const none = new SessionPanel(new Theme('dark', 0), { ...DATA, mcpServers: [] }, { rows: () => 40 })

    expect(plain(none.render(100)).join('\n')).not.toContain('MCP')
  })

  it('opens with the lockup and follows it with what this session is', () => {
    const panel = new SessionPanel(new Theme('dark', 0), DATA, { rows: () => 40 })
    const lines = plain(panel.render(100))
    const name = lines.findIndex(line => line.includes('OpenDDE Harness v0.4.2'))
    const facts = lines.findIndex(line => line.includes('tui:abc123'))

    expect(artRows(panel.render(100)).length).toBeGreaterThan(6)
    expect(name).toBeGreaterThan(0)
    expect(facts).toBeGreaterThan(name)
    // Model, provider and session on one row, as the old panel had.
    expect(lines[facts]).toBe('gpt-5 · openai · session tui:abc123')
  })

  it('draws the lockup once, however often it renders', () => {
    const panel = new SessionPanel(new Theme('dark', 0), DATA, { rows: () => 40 })

    panel.render(100)
    panel.update({ model: 'claude-opus-5' })

    const lines = plain(panel.render(100))

    expect(lines.filter(line => line.includes('OpenDDE Harness'))).toHaveLength(1)
  })

  it('keeps every row inside the terminal it draws on', () => {
    const terminal = new FakeTerminal(72, 20)
    const tui = new TuiMainScreen(terminal)

    tui.addChild(new SessionPanel(new Theme('dark', 3), DATA, { rows: () => terminal.rows }))
    tui.renderNow()

    for (const line of stripTerminalSequences(terminal.output()).split('\n')) {
      expect(visibleWidth(line)).toBeLessThanOrEqual(72)
    }
  })
})

describe('the welcome panel on a real boot', () => {
  it('shows what the gateway said about this build and this agent', async () => {
    const info = {
      cwd: '/work/proj',
      mcp_servers: [{ connected: true, name: 'files', tool_count: 3, transport: 'stdio' }],
      model: 'gpt-5',
      provider: 'openai',
      release_date: '2026-09-10',
      version: '0.4.2'
    }
    const transport = new FakeTransport({
      'commands.catalog': { categories: [], pairs: [], skill_count: 0 },
      'config.get': { config: {} },
      'session.create': { info, session_id: 'tui:abc123' },
      'setup.status': { provider_configured: true },
      'system.hello': { server_capabilities: [], server_version: '0.4.2', session: {} }
    })
    const terminal = new FakeTerminal(100, 34)
    const tui = new TuiMainScreen(terminal)
    const app = new HarnessApp({ env: {}, gateway: new Gateway(transport), theme: new Theme('dark', 0), tui })

    tui.start()
    await app.boot()
    tui.renderNow()

    const screen = stripTerminalSequences(terminal.output())

    expect(screen).toContain('OpenDDE Harness v0.4.2 (2026-09-10)')
    expect(screen).toContain('1 MCP server')
    // The working directory is the footer's, and only the footer's: the panel
    // used to print it too, which said the same thing twice on one screen.
    expect(screen.match(/\/work\/proj/g) ?? []).toHaveLength(1)
  })
})

describe('the lockup and a palette that moves under it', () => {
  it('repaints the art when the scheme changes after it was first rendered', () => {
    const theme = new Theme('dark', 3)
    const branding = new Branding(theme, { rows: () => 40, version: '0.4.2' })
    const dark = branding.render(100).join('\n')

    expect(theme.setScheme('light')).toBe(true)

    const light = branding.render(100).join('\n')

    // Held lines are keyed on the palette, not only the width: the terminal
    // answers the background query after the app is built, and `tui.theme`
    // lands after that. Without the key the lockup kept the boot's guess
    // while everything painted later was correct.
    expect(light).not.toBe(dark)
    expect(dark).toContain('38;2;167;139;250')
    expect(light).toContain('38;2;124;58;237')
    expect(light).not.toContain('38;2;167;139;250')
  })

  it('serves the held lines when nothing about the palette or the size moved', () => {
    const branding = new Branding(new Theme('dark', 3), { rows: () => 40, version: '0.4.2' })

    expect(branding.render(100)).toBe(branding.render(100))
  })
})
