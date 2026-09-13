import { stripTerminalSequences, TuiMainScreen } from '@earendil-works/pi-tui'
import { Chalk } from 'chalk'
import { describe, expect, it } from 'vitest'

import type { ColorTier } from '../lib/colorTier.js'
import type { ColorScheme, ThemeToken } from '../theme.js'

import { HarnessApp } from '../app.js'
import { settleScheme } from '../boot.js'
import { Gateway } from '../gateway.js'
import { parseColorOverride, resolveColorTier } from '../lib/colorTier.js'
import {
  applyConfiguredTheme,
  createTheme,
  detectScheme,
  INHERIT,
  resolvePalette,
  resolveThemeName,
  schemeForThemeName,
  schemeFromBackground,
  Theme,
  THEME_NAMES
} from '../theme.js'
import { FakeTerminal, FakeTransport } from './fakes.js'

const ESC = String.fromCharCode(27)
const BEL = String.fromCharCode(7)

describe('color tier', () => {
  it('reads the override channels', () => {
    expect(parseColorOverride({ OPENDDE_HARNESS_TUI_COLOR: 'truecolor' })).toBe(3)
    expect(parseColorOverride({ OPENDDE_HARNESS_TUI_COLOR: '256' })).toBe(2)
    expect(parseColorOverride({ OPENDDE_HARNESS_TUI_COLOR: 'ansi' })).toBe(1)
    expect(parseColorOverride({ OPENDDE_HARNESS_TUI_COLOR: 'none' })).toBe(0)
    expect(parseColorOverride({ OPENDDE_HARNESS_TUI_COLOR: 'auto' })).toBeNull()
    expect(parseColorOverride({ OPENDDE_HARNESS_TUI_TRUECOLOR: 'yes' })).toBe(3)
    expect(parseColorOverride({})).toBeNull()
  })

  it('lets NO_COLOR beat an explicit --color', () => {
    expect(resolveColorTier({ NO_COLOR: '', OPENDDE_HARNESS_TUI_COLOR: 'truecolor' }, 3)).toBe(0)
  })

  it('pins the requested tier instead of detecting', () => {
    expect(resolveColorTier({ OPENDDE_HARNESS_TUI_COLOR: '16' }, 3)).toBe(1)
    expect(resolveColorTier({ OPENDDE_HARNESS_TUI_COLOR: 'truecolor', TMUX: '/tmp/x' }, 1)).toBe(3)
  })

  it('boosts xterm.js and clamps tmux and Terminal.app', () => {
    expect(resolveColorTier({ TERM_PROGRAM: 'vscode' }, 2)).toBe(3)
    expect(resolveColorTier({ TMUX: '/tmp/tmux-0/default' }, 3)).toBe(2)
    expect(resolveColorTier({ TERM_PROGRAM: 'Apple_Terminal' }, 3)).toBe(2)
    // tmux inside VS Code: boosted to 3, then clamped back to what tmux passes.
    expect(resolveColorTier({ TERM_PROGRAM: 'vscode', TMUX: '/tmp/x' }, 2)).toBe(2)
  })

  it('passes an undecided environment through untouched', () => {
    expect(resolveColorTier({}, 3)).toBe(3)
    expect(resolveColorTier({}, 0)).toBe(0)
  })
})

describe('scheme detection', () => {
  it('puts the explicit flags above every measured signal', () => {
    expect(detectScheme({ COLORFGBG: '0;15', OPENDDE_HARNESS_TUI_LIGHT: 'no' })).toBe('dark')
    expect(detectScheme({ OPENDDE_HARNESS_TUI_LIGHT: '1', OPENDDE_HARNESS_TUI_THEME: 'dark' })).toBe('light')
    expect(detectScheme({ OPENDDE_HARNESS_TUI_BACKGROUND: '#ffffff', OPENDDE_HARNESS_TUI_THEME: 'dark' })).toBe('dark')
  })

  it('reads a background hint by luminance', () => {
    expect(detectScheme({ OPENDDE_HARNESS_TUI_BACKGROUND: '#ffffff' })).toBe('light')
    expect(detectScheme({ OPENDDE_HARNESS_TUI_BACKGROUND: 'fff' })).toBe('light')
    expect(detectScheme({ OPENDDE_HARNESS_TUI_BACKGROUND: '#1e1e1e' })).toBe('dark')
    // A truncating hex parse would read this as white; it must be ignored.
    expect(detectScheme({ COLORFGBG: '15;0', OPENDDE_HARNESS_TUI_BACKGROUND: 'fffgff' })).toBe('dark')
  })

  it('treats COLORFGBG as authoritative over the TERM_PROGRAM allow-list', () => {
    const allowList = new Set(['SomeLightTerminal'])

    expect(detectScheme({ COLORFGBG: '0;15' }, allowList)).toBe('light')
    expect(detectScheme({ COLORFGBG: '15;0', TERM_PROGRAM: 'SomeLightTerminal' }, allowList)).toBe('dark')
    expect(detectScheme({ COLORFGBG: '15;', TERM_PROGRAM: 'SomeLightTerminal' }, allowList)).toBe('light')
    expect(detectScheme({ TERM_PROGRAM: 'SomeLightTerminal' }, allowList)).toBe('light')
  })

  it('stays dark when nothing says otherwise', () => {
    expect(detectScheme({})).toBe('dark')
    expect(detectScheme({ TERM_PROGRAM: 'iTerm.app' })).toBe('dark')
  })

  it('reads a measured background the same way', () => {
    expect(schemeFromBackground({ b: 255, g: 255, r: 255 })).toBe('light')
    expect(schemeFromBackground({ b: 30, g: 30, r: 30 })).toBe('dark')
  })
})

describe('palettes', () => {
  it('gives each scheme and tier its own curated values', () => {
    expect(resolvePalette('dark', 3).primary).toBe('#A78BFA')
    expect(resolvePalette('light', 3).primary).toBe('#7C3AED')
    expect(resolvePalette('dark', 2).primary).toBe('ansi256(141)')
    expect(resolvePalette('light', 2).primary).toBe('ansi256(93)')
    expect(resolvePalette('dark', 1).primary).toBe('ansi:magentaBright')
    expect(resolvePalette('light', 1).primary).toBe('ansi:magenta')
    // Tier 0 shares the hex palette; chalk strips the codes anyway.
    expect(resolvePalette('dark', 0)).toBe(resolvePalette('dark', 3))
  })

  it('carries every token into every tier', () => {
    const tokens = Object.keys(resolvePalette('dark', 3)).sort()

    for (const scheme of ['dark', 'light'] as const) {
      for (const tier of [1, 2, 3] as const) {
        expect(Object.keys(resolvePalette(scheme, tier)).sort()).toEqual(tokens)
      }
    }
  })
})

describe('Theme', () => {
  it('emits the escape form that belongs to its tier', () => {
    expect(new Theme('dark', 3).fg('primary', 'x')).toContain('38;2;167;139;250')
    expect(new Theme('dark', 2).fg('primary', 'x')).toContain('38;5;141')
    expect(new Theme('dark', 1).fg('primary', 'x')).toContain('95m')
    expect(new Theme('dark', 0).fg('primary', 'x')).toBe('x')
  })

  it('paints backgrounds from the same token', () => {
    expect(new Theme('dark', 2).bg('selectionBg', 'x')).toContain('48;5;60')
  })

  it('re-themes in place when the scheme flips', () => {
    const theme = new Theme('dark', 3)
    // Body text is the terminal's own in both schemes, so a token that does
    // change is what says the palette moved.
    const before = theme.fg('muted', 'x')

    expect(theme.setScheme('dark')).toBe(false)
    expect(theme.setScheme('light')).toBe(true)
    expect(theme.scheme).toBe('light')
    expect(theme.fg('muted', 'x')).not.toBe(before)
  })

  it('hands pi-tui adapters that follow the live palette', () => {
    const theme = new Theme('dark', 3)
    const markdown = theme.markdownTheme()
    const dark = markdown.link('pi')

    theme.setScheme('light')

    expect(markdown.link('pi')).not.toBe(dark)
    expect(theme.editorTheme().selectList.description('note')).toBe(theme.fg('muted', 'note'))
  })

  it('keeps brand color out of ordinary prose', () => {
    const theme = new Theme('dark', 3)
    const markdown = theme.markdownTheme()

    // A heading and a list bullet are the terminal's own foreground, with
    // weight doing the work on the heading. Neither carries a color at all.
    expect(stripTerminalSequences(markdown.heading('Title'))).toBe('Title')
    expect(markdown.heading('Title')).toContain('Title')
    expect(markdown.listBullet('-')).toBe('-')

    // Links and inline code take the periwinkle label, never the brand accent.
    expect(markdown.link('pi')).toBe(theme.fg('label', 'pi'))
    expect(markdown.code('npm ci')).toBe(theme.fg('label', 'npm ci'))
    expect(markdown.linkUrl('https://x')).toBe(theme.fg('dim', 'https://x'))

    for (const painted of [markdown.heading('Title'), markdown.link('pi'), markdown.code('npm ci')]) {
      expect(painted).not.toContain(theme.colors.accent.replace('#', ''))
    }
  })

  it('builds from the environment without touching the global chalk', () => {
    const globalLevel = new Chalk().level
    const theme = createTheme({ OPENDDE_HARNESS_TUI_COLOR: '256', OPENDDE_HARNESS_TUI_LIGHT: '1' }, 3)

    expect(theme.tier).toBe(2)
    expect(theme.scheme).toBe('light')
    expect(new Chalk().level).toBe(globalLevel)
  })
})

describe('tui.theme', () => {
  it('names the two palettes this UI ships, plus detection', () => {
    expect([...THEME_NAMES]).toEqual(['default', 'dark', 'light'])
    expect(schemeForThemeName('dark')).toBe('dark')
    expect(schemeForThemeName('light')).toBe('light')
    // `default` fixes nothing: the environment and the terminal decide.
    expect(schemeForThemeName('default')).toBeNull()
  })

  it('reads a name however it was typed, and rejects anything else', () => {
    expect(resolveThemeName('Light')).toBe('light')
    expect(resolveThemeName('  DARK ')).toBe('dark')
    expect(resolveThemeName('solarized')).toBeUndefined()
    expect(resolveThemeName('')).toBeUndefined()
    expect(resolveThemeName(undefined)).toBeUndefined()
    expect(resolveThemeName(7)).toBeUndefined()
  })

  it('applies a stored palette and asks for the repaint', () => {
    const theme = new Theme('dark', 3)

    expect(applyConfiguredTheme(theme, 'light')).toEqual({ invalid: false, repaint: true })
    expect(theme.scheme).toBe('light')

    // The same name again changes nothing, so nothing is repainted.
    expect(applyConfiguredTheme(theme, 'light')).toEqual({ invalid: false, repaint: false })
  })

  it('keeps the detected palette for `default` and for no value at all', () => {
    for (const stored of ['default', '', '   ', undefined, null]) {
      const theme = new Theme('dark', 3)

      expect(applyConfiguredTheme(theme, stored)).toEqual({ invalid: false, repaint: false })
      expect(theme.scheme).toBe('dark')
      expect(theme.pinned).toBe(false)
    }
  })

  it('falls back to what was detected when the name is not one we ship', () => {
    const theme = new Theme('dark', 3)

    expect(applyConfiguredTheme(theme, 'solarized')).toEqual({ invalid: true, repaint: false })
    // Fell back rather than failed: the palette is still usable.
    expect(theme.scheme).toBe('dark')
    expect(theme.fg('primary', 'x')).toContain('38;2;167;139;250')
  })

  it('lets the environment outrank the stored name', () => {
    const theme = createTheme({ OPENDDE_HARNESS_TUI_THEME: 'dark' }, 3)

    expect(theme.pinned).toBe(true)
    expect(applyConfiguredTheme(theme, 'light')).toEqual({ invalid: false, repaint: false })
    expect(theme.scheme).toBe('dark')
  })

  it('leaves a measured background to correct only an unpinned palette', () => {
    const measured = createTheme({}, 3)

    expect(measured.pinned).toBe(false)
    expect(measured.setScheme(schemeFromBackground({ b: 255, g: 255, r: 255 }))).toBe(true)
    expect(measured.scheme).toBe('light')

    // Pinned by `tui.theme`, the same reply is ignored: a measurement must not
    // overrule what the user asked for.
    const chosen = new Theme('dark', 3)

    applyConfiguredTheme(chosen, 'dark')
    expect(chosen.pinned).toBe(true)
    expect(chosen.setScheme('light')).toBe(false)
    expect(chosen.scheme).toBe('dark')
  })
})

describe('reading tui.theme at boot', () => {
  const SCRIPT = {
    'commands.catalog': { categories: [], pairs: [], skill_count: 0 },
    'config.get': { config: {} },
    'session.create': { info: { cwd: '/work', model: 'gpt-5', provider: 'openai' }, session_id: 'tui:abc' },
    'setup.status': { provider_configured: true },
    'system.hello': { server_capabilities: [], server_version: '9.9.9', session: {} }
  }

  /** The real app on a fake terminal, at truecolor so the palette is visible
   *  on the wire. The theme is built here rather than from the environment, so
   *  nothing outside the test can pin it. */
  function build(config: Record<string, unknown>) {
    const transport = new FakeTransport({ ...SCRIPT, 'config.get': { config } })
    const terminal = new FakeTerminal(100, 30)
    const tui = new TuiMainScreen(terminal)
    const theme = new Theme('dark', 3)
    const app = new HarnessApp({
      cwd: '/nonexistent',
      env: {},
      gateway: new Gateway(transport),
      listenResize: () => () => {},
      onExit: () => {},
      theme,
      tui
    })

    tui.start()

    return { app, screen: () => terminal.output(), theme, transport, tui }
  }

  it('paints with the palette the gateway has stored', async () => {
    const { app, screen, theme, transport, tui } = build({ 'tui.theme': 'light' })

    await app.boot()
    tui.renderNow()

    expect(theme.scheme).toBe('light')
    // The light palette reached the wire: #666666, the light muted.
    expect(screen()).toContain('38;2;102;102;102')
    // No second read: boot already holds the whole config.
    expect(transport.methods.filter(method => method === 'config.get')).toHaveLength(1)
  })

  it('says so once when the stored name is not a palette, and keeps painting', async () => {
    const { app, screen, theme, tui } = build({ 'tui.theme': 'solarized' })

    await app.boot()
    tui.renderNow()

    expect(stripTerminalSequences(screen())).toContain('tui.theme: solarized is not a palette')
    expect(theme.scheme).toBe('dark')
  })

  it('leaves the palette alone when the gateway serves no preference', async () => {
    const { app, screen, theme, tui } = build({})

    await app.boot()
    tui.renderNow()

    expect(theme.scheme).toBe('dark')
    expect(stripTerminalSequences(screen())).not.toContain('tui.theme')
  })
})

describe('the palette split', () => {
  const SCHEMES: ColorScheme[] = ['dark', 'light']
  const TIERS: ColorTier[] = [0, 1, 2, 3]

  /** Chrome: the frame around the work, which the palette keeps neutral. */
  const CHROME: ThemeToken[] = ['border', 'muted', 'dim', 'userMessageBg', 'toolPendingBg', 'toolSuccessBg']

  /** Identity: the few places the brand violet is allowed. */
  const IDENTITY: ThemeToken[] = ['primary', 'accent', 'prompt']

  const NEUTRAL_256 = new Set([234, 236, 237, 145, 240, 241, 243, 244, 246, 255])
  const NEUTRAL_NAMES = new Set(['ansi:black', 'ansi:blackBright', 'ansi:white', 'ansi:whiteBright'])
  const VIOLET_256 = new Set([92, 93, 99, 141])

  /** How far apart a hex value's channels are. Zero is a pure grey. */
  function spread(hex: string): number {
    const channels = [1, 3, 5].map(at => parseInt(hex.slice(at, at + 2), 16))

    return Math.max(...channels) - Math.min(...channels)
  }

  function slot(value: string): number {
    return Number(/^ansi256\((\d+)\)$/.exec(value)?.[1])
  }

  it('never lets chrome take the brand hue, at any tier', () => {
    for (const scheme of SCHEMES) {
      for (const tier of TIERS) {
        const palette = resolvePalette(scheme, tier)

        for (const token of CHROME) {
          const value = palette[token]

          if (value.startsWith('#')) {
            // A grey, or near enough: the popup's ground carries four points of
            // blue and nothing else does. The brand violet is a hundred points
            // of blue over green, so the threshold is nowhere near it.
            expect({ scheme, tier, token, value, tinted: spread(value) > 6 }).toMatchObject({ tinted: false })
          } else if (value.startsWith('ansi256(')) {
            expect({ scheme, tier, token, neutral: NEUTRAL_256.has(slot(value)) }).toMatchObject({ neutral: true })
          } else {
            expect({ scheme, tier, token, neutral: NEUTRAL_NAMES.has(value) }).toMatchObject({ neutral: true })
          }
        }
      }
    }
  })

  it('keeps the brand violet for identity, at every tier', () => {
    for (const scheme of SCHEMES) {
      for (const tier of TIERS) {
        const palette = resolvePalette(scheme, tier)

        for (const token of IDENTITY) {
          const value = palette[token]

          if (value.startsWith('#')) {
            const [r, g, b] = [1, 3, 5].map(at => parseInt(value.slice(at, at + 2), 16)) as [number, number, number]

            // Violet: blue leads, green trails, red between the two.
            expect({ scheme, tier, token, violet: b > r && r > g }).toMatchObject({ violet: true })
          } else if (value.startsWith('ansi256(')) {
            expect({ scheme, tier, token, violet: VIOLET_256.has(slot(value)) }).toMatchObject({ violet: true })
          } else {
            expect(value).toContain('magenta')
          }
        }
      }
    }
  })

  it('leaves body text to the terminal', () => {
    for (const scheme of SCHEMES) {
      for (const tier of TIERS) {
        expect(resolvePalette(scheme, tier).text).toBe(INHERIT)
      }
    }

    // And the painter really does leave it alone, rather than emitting a code
    // that happens to look like the default.
    const painted = new Theme('dark', 3)

    expect(painted.fg('text', 'hello')).toBe('hello')
    expect(painted.bg('text', 'hello')).toBe('hello')
  })

  it('takes the reference theme\u2019s greys and semantics verbatim', () => {
    const dark = resolvePalette('dark', 3)
    const light = resolvePalette('light', 3)

    // pi-claude-theme's own values, so the two products' chrome matches.
    expect([dark.muted, dark.dim, dark.border]).toEqual(['#999999', '#666666', '#505050'])
    expect([dark.ok, dark.error, dark.warn]).toEqual(['#4EBA65', '#FF6B80', '#FFC107'])
    expect([light.ok, light.error, light.warn]).toEqual(['#2C7A39', '#AB2B3F', '#966C1E'])
    // The band is a ground, not an accent: the reference's flat grey.
    expect([dark.userMessageBg, light.userMessageBg]).toEqual(['#373737', '#F0F0F0'])
    // Only a failed tool tints its frame.
    expect(dark.toolPendingBg).toBe(dark.toolSuccessBg)
    expect(dark.toolErrorBg).not.toBe(dark.toolPendingBg)
  })
})

describe('settling the palette before the first paint', () => {
  /** A terminal that answers OSC 11 with `background`, or never answers. */
  function query(background?: { b: number; g: number; r: number }) {
    const asked: number[] = []

    return {
      asked,
      queryTerminalBackgroundColor: async ({ timeoutMs }: { timeoutMs: number }) => {
        asked.push(timeoutMs)

        return background
      }
    }
  }

  it('does not ask when the environment already chose', async () => {
    const theme = createTheme({ OPENDDE_HARNESS_TUI_LIGHT: '1' }, 3)
    const terminal = query({ b: 0, g: 0, r: 0 })

    expect(await settleScheme(theme, terminal)).toBe(false)
    // Not one round trip spent on an answer that would be ignored.
    expect(terminal.asked).toEqual([])
    expect(theme.scheme).toBe('light')
  })

  it('asks when nothing has, and paints what the terminal says', async () => {
    const theme = createTheme({}, 3)
    const terminal = query({ b: 255, g: 255, r: 255 })

    expect(theme.scheme).toBe('dark')
    expect(await settleScheme(theme, terminal, { timeoutMs: 60 })).toBe(true)
    // One question, on a deadline that outlives the first paint: the paint is
    // bounded by racing this, not by giving up on it.
    expect(terminal.asked).toEqual([2060])
    expect(theme.scheme).toBe('light')
    // Applied, not pinned: `tui.theme` still outranks a measurement.
    expect(theme.pinned).toBe(false)
  })

  it('keeps the guess when the terminal never answers', async () => {
    const theme = createTheme({}, 3)
    const terminal = query(undefined)

    expect(await settleScheme(theme, terminal, { lateTimeoutMs: 0, timeoutMs: 5 })).toBe(false)
    expect(terminal.asked.length).toBeGreaterThan(0)
    expect(theme.scheme).toBe('dark')
  })

  it('takes the answer through the real terminal plumbing', async () => {
    const terminal = new FakeTerminal(80, 24)
    const tui = new TuiMainScreen(terminal)
    const theme = createTheme({}, 3)

    tui.start()

    const settled = settleScheme(theme, tui, { timeoutMs: 500 })

    // The reply as a terminal sends it, through the input path pi parses.
    await new Promise(resolve => setTimeout(resolve, 5))
    terminal.onInput?.(`${ESC}]11;rgb:ffff/ffff/ffff${BEL}`)

    expect(await settled).toBe(true)
    expect(theme.scheme).toBe('light')
    tui.stop()
  })

  it('applies an answer that arrives after the first paint, and says so', async () => {
    // A terminal at the far end of an ssh hop can be slower than the paint
    // budget. Abandoning detection there left it permanently on the guess,
    // which is the failure the query was added to prevent, not a smaller one.
    const terminal = new FakeTerminal(80, 24)
    const tui = new TuiMainScreen(terminal)
    const theme = createTheme({}, 3)
    const repaints: string[] = []

    tui.start()

    const settled = await settleScheme(theme, tui, {
      lateTimeoutMs: 500,
      onLateChange: () => repaints.push(theme.scheme),
      timeoutMs: 10
    })

    // The first paint went ahead on the guess rather than waiting.
    expect(settled).toBe(false)
    expect(theme.scheme).toBe('dark')
    expect(repaints).toEqual([])

    await new Promise(resolve => setTimeout(resolve, 40))
    terminal.onInput?.(`${ESC}]11;rgb:ffff/ffff/ffff${BEL}`)
    await new Promise(resolve => setTimeout(resolve, 5))

    // Late, so what is on screen has to be repainted rather than painted.
    expect(theme.scheme).toBe('light')
    expect(repaints).toEqual(['light'])
    tui.stop()
  })

  it('asks again when a reply is shaped like an answer but carries no color', async () => {
    // pi settles the pending query on any reply it recognizes as an OSC 11
    // response, including one whose color it cannot parse. That consumes the
    // question without answering it, and is indistinguishable from silence
    // unless the question is put again.
    const terminal = new FakeTerminal(80, 24)
    const tui = new TuiMainScreen(terminal)
    const theme = createTheme({}, 3)

    tui.start()

    const settled = settleScheme(theme, tui, { lateTimeoutMs: 0, timeoutMs: 500 })

    await new Promise(resolve => setTimeout(resolve, 5))
    terminal.onInput?.(`${ESC}]11;rgb:zzzz/zzzz/zzzz${BEL}`)
    await new Promise(resolve => setTimeout(resolve, 5))
    terminal.onInput?.(`${ESC}]11;rgb:ffff/ffff/ffff${BEL}`)

    expect(await settled).toBe(true)
    expect(theme.scheme).toBe('light')
    tui.stop()
  })

  it('treats a terminal that fails the question as one that did not answer', async () => {
    const theme = createTheme({}, 3)

    await expect(
      settleScheme(theme, {
        queryTerminalBackgroundColor: async () => {
          throw new Error('no tty')
        }
      })
    ).resolves.toBe(false)
    expect(theme.scheme).toBe('dark')
  })
})
