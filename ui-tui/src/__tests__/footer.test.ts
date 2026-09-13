import { stripTerminalSequences, visibleWidth } from '@earendil-works/pi-tui'
import { describe, expect, it } from 'vitest'

import type { FooterData } from '../components/footer.js'

import { emptyUsage, Footer, formatCwd, formatTokens } from '../components/footer.js'
import { Theme } from '../theme.js'

const theme = new Theme('dark', 0)

const BASE: FooterData = {
  branch: 'tui/pi-tui',
  cwd: '/home/dev/work/opendde-harness',
  effort: 'high',
  model: 'gpt-5',
  provider: 'openai',
  providerCount: 3,
  usage: {
    cacheHitPercent: 74,
    contextTokens: 36_000,
    contextWindow: 200_000,
    costUsd: 0.0314,
    input: 12_345,
    output: 3_456
  }
}

function render(data: FooterData, width: number, home = '/home/dev'): string[] {
  return new Footer(theme, () => data, home).render(width)
}

describe('formatTokens', () => {
  it.each([
    [0, '0'],
    [999, '999'],
    [1234, '1.2k'],
    [45_678, '46k'],
    [1_234_567, '1.2M'],
    [12_345_678, '12M']
  ])('%i renders as %s', (count, expected) => {
    expect(formatTokens(count)).toBe(expected)
  })
})

describe('formatCwd', () => {
  it('collapses the home directory', () => {
    expect(formatCwd('/home/dev/work', '/home/dev')).toBe('~/work')
    expect(formatCwd('/home/dev', '/home/dev')).toBe('~')
  })

  it('leaves paths outside home alone', () => {
    expect(formatCwd('/srv/app', '/home/dev')).toBe('/srv/app')
    expect(formatCwd('/srv/app', undefined)).toBe('/srv/app')
  })
})

describe('Footer', () => {
  it('renders two lines that fit the width', () => {
    for (const width of [60, 100, 160]) {
      const lines = render(BASE, width)

      expect(lines).toHaveLength(2)

      for (const line of lines) {
        expect(visibleWidth(line)).toBeLessThanOrEqual(width)
      }
    }
  })

  it('puts the location and the branch on the first line, and nothing else', () => {
    const [location] = render(BASE, 160)

    // pi's whole line. It appends a session name only for one the user named
    // by hand; a session here is titled from its first message, so the line
    // used to carry that -- or the raw session id before the first message.
    expect(stripTerminalSequences(location ?? '')).toBe('~/work/opendde-harness (tui/pi-tui)')
  })

  it('shows usage on the left and the model on the right', () => {
    const [, stats] = render(BASE, 160)

    expect(stats).toContain('↑12k')
    expect(stats).toContain('↓3.5k')
    expect(stats).not.toContain('↓~')
    expect(stats).toContain('$0.031')
    expect(stats).toContain('18.0%/200k')
    expect(stats!.trimEnd().endsWith('(openai) gpt-5 • high')).toBe(true)
  })

  it('names the cache figures the way pi does, when there are any', () => {
    const usage = { ...BASE.usage, cacheHitPercent: 74.2, cacheRead: 40_000, cacheWrite: 2_000 }
    const [, stats] = render({ ...BASE, usage }, 160)

    expect(stats).toContain('R40k W2.0k CH74.2%')
  })

  it('says nothing about a hit rate with no cache figures beside it', () => {
    // The gateway reports the rate and no counts. A rate on its own is a
    // number with nothing to read it against, and pi does not print one.
    const [, stats] = render({ ...BASE, usage: { ...BASE.usage, cacheHitPercent: 50 } }, 160)

    expect(stats).not.toContain('CH')
    expect(stats).not.toContain('cache')
  })

  it('marks an output figure that is still an estimate', () => {
    const [, stats] = render({ ...BASE, usage: { ...BASE.usage, outputEstimated: true } }, 160)

    expect(stats).toContain('↓~3.5k')
  })

  it('drops the provider before it truncates the model', () => {
    const wide = render(BASE, 160)[1]!
    // Narrow enough that the provider and the model together do not fit
    // beside the stats.
    const narrow = render(BASE, 48)[1]!

    expect(wide).toContain('(openai)')
    expect(narrow).not.toContain('(openai)')
    expect(narrow).toContain('gpt-5')
  })

  it('keeps the provider hidden when only one is configured', () => {
    const [, stats] = render({ ...BASE, providerCount: 1 }, 160)

    expect(stats).not.toContain('(openai)')
  })

  it('marks fast mode', () => {
    expect(render({ ...BASE, fast: true }, 160)[1]).toContain('[fast]')
  })

  it('says what it does not know', () => {
    const bare: FooterData = { branch: null, cwd: '/srv', usage: emptyUsage() }
    const [location, stats] = render(bare, 60)

    expect(location).toBe('/srv')
    expect(stats).toContain('no model')
  })

  it('reads 0.0% before the first call rather than a question mark', () => {
    // Nothing has been sent, so nothing is in the window. That is a number,
    // not an unknown, and `?/128k` was reading as a failure.
    const usage = { ...emptyUsage(), contextWindow: 128_000 }
    const [, stats] = render({ ...BASE, usage }, 100)

    expect(stats).toContain('0.0%/128k')
    expect(stats).not.toContain('?')
  })

  it('truncates the location line rather than wrapping it', () => {
    const long = { ...BASE, cwd: `/home/dev/${'deep/'.repeat(20)}leaf` }
    const [location] = render(long, 40)

    expect(visibleWidth(location!)).toBe(40)
    expect(location).toContain('…')
  })

  it('colours the context bar as it fills up', () => {
    const colour = new Theme('dark', 3)
    const at = (tokens: number) =>
      new Footer(colour, () => ({ ...BASE, usage: { ...BASE.usage, contextTokens: tokens } }), '/home/dev').render(
        160
      )[1]!

    expect(at(20_000)).toContain(colour.fg('dim', '10.0%/200k'))
    expect(at(160_000)).toContain(colour.fg('warn', '80.0%/200k'))
    expect(at(190_000)).toContain(colour.fg('error', '95.0%/200k'))
  })

  it("paints both lines in pi's dim, and keeps the whole stats line in it", () => {
    // pi paints its footer with `dim`, the grey one step below `muted`. A
    // colour nested inside a line ends with a reset, which would clear the
    // line's own colour for everything after it -- so the estimate marker
    // carries none, and the run of figures stays one colour.
    const colour = new Theme('dark', 3)
    const [location, stats] = new Footer(
      colour,
      () => ({ ...BASE, usage: { ...BASE.usage, outputEstimated: true } }),
      '/home/dev'
    ).render(160)

    expect(location).toContain(colour.fg('dim', '~/work/opendde-harness (tui/pi-tui)'))
    expect(location).not.toContain(colour.fg('muted', '~/work/opendde-harness (tui/pi-tui)'))
    expect(stats).toContain(colour.fg('dim', '(openai) gpt-5 • high'))
    expect(stripTerminalSequences(stats ?? '')).toContain('↑12k ↓~3.5k')
    expect(stats).toContain(colour.fg('dim', '↑12k ↓~3.5k $0.031'))
  })

  it('reuses the cached lines until the data or the width changes', () => {
    let data = BASE
    const footer = new Footer(theme, () => data, '/home/dev')

    expect(footer.render(100)).toBe(footer.render(100))

    data = { ...BASE, branch: 'main' }
    expect(footer.render(100)[0]).toContain('(main)')
  })
})

describe('the update notice', () => {
  it('puts the notice in front of the path so a long path cannot hide it', () => {
    const data: FooterData = { ...BASE, update: { command: 'ddeharness self-update', version: 'v0.5.0' } }
    const [location] = render(data, 100)

    expect(location).toContain('v0.5.0')
    expect(location).toContain('ddeharness self-update')
    expect(location).toContain('~/work/opendde-harness')
    expect(visibleWidth(location!)).toBeLessThanOrEqual(100)
  })

  it('keeps the notice and drops the path when the line is too narrow for both', () => {
    const data: FooterData = { ...BASE, update: { command: 'ddeharness self-update', version: 'v0.5.0' } }

    for (const width of [1, 10, 24, 40]) {
      const [location] = render(data, width)

      expect(visibleWidth(location!)).toBeLessThanOrEqual(width)
    }
  })

  it('says nothing when there is no update', () => {
    const [location] = render(BASE, 100)

    expect(location).not.toContain('↑ v')
  })
})

describe('the stats line reads as pi prints it', () => {
  /** The owner's reference line, from a Codex session:
   *  `↑11k ↓41 $0.056 (sub) 2.0%/272k (auto)`. */
  const CODEX: FooterData = {
    autoCompact: true,
    branch: null,
    cwd: '/home/dev/work',
    effort: 'high',
    model: 'gpt-5',
    provider: 'openai',
    providerCount: 3,
    subscription: true,
    usage: {
      ...emptyUsage(),
      contextTokens: 5_440,
      contextWindow: 272_000,
      costUsd: 0.056,
      input: 11_000,
      output: 41
    }
  }

  /** The left group alone, with the padding and the model taken off. */
  const statsOf = (data: FooterData, width: number): string => {
    const [, line] = render(data, width)

    return line!.replace(/\s{2,}.*$/, '')
  }

  it('matches the reference line exactly', () => {
    expect(statsOf(CODEX, 160)).toBe('↑11k ↓41 $0.056 (sub) 2.0%/272k (auto)')
  })

  it('reads the same at every width the terminal can be', () => {
    for (const width of [60, 100, 160]) {
      expect(statsOf(CODEX, width)).toBe('↑11k ↓41 $0.056 (sub) 2.0%/272k (auto)')
    }
  })

  it('drops (sub) and keeps the price on a metered provider', () => {
    expect(statsOf({ ...CODEX, subscription: false }, 160)).toBe('↑11k ↓41 $0.056 2.0%/272k (auto)')
  })

  it('still says (sub) on a plan that has spent nothing', () => {
    const free = { ...CODEX, usage: { ...CODEX.usage, costUsd: 0 } }

    expect(statsOf(free, 160)).toBe('↑11k ↓41 $0.000 (sub) 2.0%/272k (auto)')
  })

  it('says nothing about cost on a metered provider that has spent nothing', () => {
    const free = { ...CODEX, subscription: false, usage: { ...CODEX.usage, costUsd: 0 } }

    expect(statsOf(free, 160)).toBe('↑11k ↓41 2.0%/272k (auto)')
  })

  it('drops (auto) when nothing will compact the context', () => {
    expect(statsOf({ ...CODEX, autoCompact: false }, 160)).toBe('↑11k ↓41 $0.056 (sub) 2.0%/272k')
  })

  it('opens a session at 0.0% of the window', () => {
    const fresh = { ...CODEX, usage: { ...emptyUsage(), contextWindow: 272_000 } }

    expect(statsOf(fresh, 160)).toBe('$0.000 (sub) 0.0%/272k (auto)')
  })

  it('counts tokens the way pi counts them', () => {
    const sizes: [number, string][] = [
      [999, '999'],
      [1_234, '1.2k'],
      [34_000, '34k'],
      [1_234_567, '1.2M']
    ]

    for (const [count, expected] of sizes) {
      const line = statsOf({ ...CODEX, usage: { ...CODEX.usage, input: count } }, 160)

      expect(line.startsWith(`↑${expected} `)).toBe(true)
    }
  })

  it('writes a question mark while a compaction leaves the window unmeasured', () => {
    const unknown = { ...CODEX, usage: { ...CODEX.usage, contextTokens: null } }

    // pi's own shape: no percent sign after the question mark, the window
    // still stated, and no colour, because 0 is not a number to warn about.
    expect(statsOf(unknown, 160)).toBe('↑11k ↓41 $0.056 (sub) ?/272k (auto)')

    // Unmeasured is not nearly full. On a terminal that has colour, the one
    // the footer gives a window at 99% is nowhere in a line stating no figure.
    const colour = new Theme('dark', 3)
    const paint = (data: FooterData): string => new Footer(colour, () => data, '/home/dev').render(160)[1]!
    const full = { ...CODEX, usage: { ...CODEX.usage, contextTokens: 271_000 } }
    const half = { ...CODEX, usage: { ...CODEX.usage, contextTokens: 136_000 } }
    const escape = new RegExp(`${String.fromCodePoint(27)}\\[[\\d;]+m`, 'g')
    const codes = (data: FooterData): string[] => paint(data).match(escape) ?? []
    const ordinary = new Set(codes(half))
    const alarm = codes(full).filter(code => !ordinary.has(code))
    const calm = paint(unknown)

    expect(alarm.length).toBeGreaterThan(0)

    for (const code of alarm) {
      expect(calm.includes(code)).toBe(false)
    }
  })

  it('paints the context group once it gets tight, and only that group', () => {
    const at = (percent: number): string => {
      const usage = { ...CODEX.usage, contextTokens: Math.round(272_000 * (percent / 100)) }

      return new Footer(theme, () => ({ ...CODEX, usage }), '/home/dev').render(160)[1]!
    }

    // The text is the same at every level; what changes is the colour, and
    // only on the part that is running out.
    for (const [percent, reads] of [
      [50, '50.0%/272k (auto)'],
      [80, '80.0%/272k (auto)'],
      [95, '95.0%/272k (auto)']
    ] as const) {
      expect(stripTerminalSequences(at(percent))).toContain(reads)
    }

    expect(at(50)).not.toBe(at(80))
    expect(at(80)).not.toBe(at(95))
    expect(stripTerminalSequences(at(50))).not.toBe(stripTerminalSequences(at(80)))
  })
})
