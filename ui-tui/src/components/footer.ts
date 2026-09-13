// The two-line footer, in pi's shape:
//
//   ~/work/proj (main)
//   ↑11k ↓41 $0.056 (sub) 2.0%/272k (auto)        (openai) gpt-5 • high [fast]
//
// The two figures answer different questions and are easy to read as one: the
// tokens and the dollars are the *session's* totals, and the percentage is the
// *last call's* share of the window. `/status` says so in a line of its own.
//
// Line two is a port of pi's, group for group, from
// `pi-mono/packages/coding-agent/src/modes/interactive/components/footer.ts`:
// `↑input ↓output`, `R`/`W` cache figures and `CH` hit rate where the provider
// reports them, `$cost` with `(sub)` on a plan, then the share of the context
// window with `(auto)` when compaction will keep it in bounds — single spaces
// throughout, and the model right-aligned. What this harness knows and pi does
// not stays out of it and lives in `/status`.
//
// `↓~3.4k` is the one addition: the `~` says the tail of that figure is this
// UI's estimate of what is streaming right now, and it is dimmer than the
// number it qualifies. The next usage report replaces it with the vendor's.
//
// Both lines are painted in pi's `dim` -- the grey one step below `muted` --
// as pi paints its own footer; the context percentage is the only thing that
// takes a colour, once it gets tight.
//
// Everything it draws comes from one `FooterData` the app fills from
// `session.status`, `turn.usage` and `config.get`; the footer itself reads
// nothing and calls nothing.

import type { Component } from '@earendil-works/pi-tui'

import { truncateToWidth, visibleWidth } from '@earendil-works/pi-tui'
import { isAbsolute, relative, resolve, sep } from 'node:path'

import type { Theme } from '../theme.js'

/** Two columns of air between the stats and the model, at minimum. */
const MIN_PADDING = 2

const CONTEXT_WARN_PERCENT = 70
const CONTEXT_ERROR_PERCENT = 90

export interface FooterUsage {
  /** Share of the most recent call's prompt served from cache, 0-100, as the
   *  server reported it. Kept as a percentage: the wire states no absolute
   *  cached-token count, and a count reconstructed from a rounded percentage
   *  and the session's summed prompt read exact and was not. Shown as pi shows
   *  it, beside the cache figures, so it is never a rate on its own. */
  cacheHitPercent?: null | number
  /** Cached prompt tokens read and written, summed over the session. The
   *  gateway reports neither today; pi's providers do, and the line has their
   *  place ready. */
  cacheRead?: number
  cacheWrite?: number
  /** Tokens the last call put in the window: its whole prompt plus its reply,
   *  which is pi's `input + output + cacheRead + cacheWrite`. */
  /** Null is pi's unknown: a compaction left nothing measuring the window.
   *  Shown as `?`, because the figure from before it is about a conversation
   *  that no longer exists. */
  contextTokens?: null | number
  contextWindow?: number
  /** What the whole session has spent, at the vendor's published price: the
   *  gateway's own running total, which is the number `/status` reports as
   *  well. Not the last turn's — a thirty-call turn's figure shown here read
   *  as the session's and disagreed with `/status` by every turn before it. */
  costUsd: number
  /** The session's total prompt tokens, cached ones included. Summed by the
   *  gateway over every billed call, this process's and the ones stored before
   *  it, rather than over the one call per turn the wire used to report. */
  input: number
  output: number
  /** Part of `output` is an estimate of what is streaming now. Marked with a
   *  `~` so a reader is not told a guess is a vendor count. */
  outputEstimated?: boolean
}

export interface FooterData {
  /** Compaction will keep the context in bounds, so a high percentage is not
   *  a cliff. pi prints `(auto)` for the same reason. */
  autoCompact?: boolean
  branch: null | string
  cwd: string
  /** Reasoning effort, when the model has one. */
  effort?: string
  fast?: boolean
  model?: string
  provider?: string
  /** The provider is only worth naming when more than one is configured. */
  providerCount?: number
  /** The model runs on a plan rather than per token, so the cost beside it is
   *  not what this conversation is costing. pi marks it `(sub)`. */
  subscription?: boolean
  /** `tui.show_token_usage`. False hides the counters; the counters themselves
   *  keep accumulating, so turning it back on shows the real totals. */
  showTokenUsage?: boolean
  /** `session.info`'s `update_available` / `update_command`. */
  update?: { command?: string; version: string }
  usage: FooterUsage
}

export function emptyUsage(): FooterUsage {
  return { costUsd: 0, input: 0, output: 0 }
}

/** Compact token counts: 1234 to 1.2k, 45678 to 46k, 1234567 to 1.2M. */
export function formatTokens(count: number): string {
  if (count < 1000) {
    return String(count)
  }

  if (count < 10_000) {
    return `${(count / 1000).toFixed(1)}k`
  }

  if (count < 1_000_000) {
    return `${Math.round(count / 1000)}k`
  }

  if (count < 10_000_000) {
    return `${(count / 1_000_000).toFixed(1)}M`
  }

  return `${Math.round(count / 1_000_000)}M`
}

/** `$HOME/work` becomes `~/work`; paths outside home are left alone. */
export function formatCwd(cwd: string, home: string | undefined): string {
  if (!home) {
    return cwd
  }

  const rel = relative(resolve(home), resolve(cwd))
  const inside = rel === '' || (rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel))

  if (!inside) {
    return cwd
  }

  return rel === '' ? '~' : `~${sep}${rel}`
}

export class Footer implements Component {
  private cacheKey = ''
  private cacheLines: string[] = []

  constructor(
    private readonly theme: Theme,
    private readonly data: () => FooterData,
    private readonly home: string | undefined = process.env.HOME ?? process.env.USERPROFILE
  ) {}

  invalidate(): void {
    this.cacheKey = ''
    this.cacheLines = []
  }

  render(width: number): string[] {
    const data = this.data()
    // The palette joins the key: see `UserMessage.render`.
    const key = `${width} ${this.theme.scheme} ${JSON.stringify(data)}`

    if (key === this.cacheKey) {
      return this.cacheLines
    }

    this.cacheKey = key
    this.cacheLines = [this.locationLine(data, width), this.statsLine(data, width)]

    return this.cacheLines
  }

  /** `cwd (branch)`, behind the update notice when there is one.
   *
   *  pi's own line, and all of it: pi appends ` • <name>` only for a session
   *  the user named by hand, and a session here is titled from its first
   *  message instead -- an id before that. Neither is a name anyone chose, and
   *  the panel above says which session this is. */
  private locationLine(data: FooterData, width: number): string {
    let text = formatCwd(data.cwd, this.home)

    if (data.branch) {
      text = `${text} (${data.branch})`
    }

    const location = truncateToWidth(this.theme.fg('dim', text), width, this.theme.fg('dim', '…'))

    if (!data.update) {
      return location
    }

    // In front of the path, not after it: appended, a long cwd would truncate
    // the one part of the line that is asking the user to do something.
    const notice = this.theme.fg(
      'warn',
      `↑ ${data.update.version}${data.update.command ? ` · ${data.update.command}` : ''}`
    )
    const noticeWidth = visibleWidth(notice)

    if (noticeWidth + MIN_PADDING >= width) {
      return truncateToWidth(notice, width, this.theme.fg('warn', '…'))
    }

    const rest = truncateToWidth(location, width - noticeWidth - 2, this.theme.fg('dim', '…'))

    return `${notice}  ${rest}`
  }

  /** Token/cost/context on the left, model right-aligned. */
  private statsLine(data: FooterData, width: number): string {
    let left = this.stats(data)
    let leftWidth = visibleWidth(left)

    if (leftWidth > width) {
      left = truncateToWidth(left, width, '…')
      leftWidth = visibleWidth(left)
    }

    const model = this.model(data)
    let right = model.full

    if (leftWidth + MIN_PADDING + visibleWidth(right) > width) {
      right = model.short
    }

    const rightWidth = visibleWidth(right)

    if (leftWidth + MIN_PADDING + rightWidth <= width) {
      return left + ' '.repeat(width - leftWidth - rightWidth) + right
    }

    const room = width - leftWidth - MIN_PADDING

    if (room <= 0) {
      return left
    }

    const clipped = truncateToWidth(right, room, '')

    return left + ' '.repeat(Math.max(0, width - leftWidth - visibleWidth(clipped))) + clipped
  }

  /**
   * pi's stats line, part for part.
   *
   * Ported from `pi-mono/packages/coding-agent/src/modes/interactive/
   * components/footer.ts`: the same groups, the same order, the same
   * separator, the same thresholds. Anything this harness knows and pi does
   * not is auxiliary and belongs in `/status`, not here — the owner asked for
   * this line to read the same as pi's.
   */
  private stats(data: FooterData): string {
    if (data.showTokenUsage === false) {
      return ''
    }

    const { usage } = data
    const parts: string[] = []

    if (usage.input) {
      parts.push(`↑${formatTokens(usage.input)}`)
    }

    if (usage.output) {
      // The `~` says the tail of this figure is this UI's estimate of what is
      // streaming, not a vendor count. Uncoloured: the whole line is already
      // pi's dim, and a colour of its own would end in a reset that cleared
      // that dim for everything after it -- which is the very thing pi's
      // footer splits its own line to avoid.
      const estimate = usage.outputEstimated ? '~' : ''

      parts.push(`↓${estimate}${formatTokens(usage.output)}`)
    }

    if (usage.cacheRead) {
      parts.push(`R${formatTokens(usage.cacheRead)}`)
    }

    if (usage.cacheWrite) {
      parts.push(`W${formatTokens(usage.cacheWrite)}`)
    }

    // Gated on the counts, as pi gates it: a hit rate beside no cache figures
    // is a number with nothing to read it against. The gateway reports the
    // rate but no counts today, so this stays quiet until it does.
    if ((usage.cacheRead || usage.cacheWrite) && usage.cacheHitPercent != null) {
      parts.push(`CH${usage.cacheHitPercent.toFixed(1)}%`)
    }

    // Shown for a subscription even at zero: on a plan there is no per-token
    // price to report, and `(sub)` is the answer to "what is this costing".
    if (usage.costUsd || data.subscription) {
      parts.push(`$${usage.costUsd.toFixed(3)}${data.subscription ? ' (sub)' : ''}`)
    }

    const line = parts.length ? `${this.theme.fg('dim', parts.join(' '))} ` : ''

    return line + this.context(data)
  }

  /**
   * `2.0%/272k (auto)`, coloured once it gets tight.
   *
   * Never `?`: the percentage is what the last call reported over the window
   * the model has, and before the first call of a session that is 0.0%, which
   * is true rather than unknown. `(auto)` says compaction will keep it from
   * running out.
   */
  private context(data: FooterData): string {
    const { usage } = data
    const window = usage.contextWindow ?? 0
    const unknown = usage.contextTokens === null
    const percent = !unknown && window > 0 ? ((usage.contextTokens ?? 0) / window) * 100 : 0
    // `?/272k`, as pi writes it: no percent sign after a question mark, and
    // the window still stated, because that much is known.
    const shown = unknown ? '?' : `${percent.toFixed(1)}%`
    const text = `${shown}/${formatTokens(window)}${data.autoCompact ? ' (auto)' : ''}`

    if (percent > CONTEXT_ERROR_PERCENT) {
      return this.theme.fg('error', text)
    }

    if (percent > CONTEXT_WARN_PERCENT) {
      return this.theme.fg('warn', text)
    }

    return this.theme.fg('dim', text)
  }

  /** The right-hand side, with and without the provider prefix. */
  private model(data: FooterData): { full: string; short: string } {
    const parts = [data.model ?? 'no model']

    if (data.effort) {
      parts.push(`• ${data.effort}`)
    }

    if (data.fast) {
      parts.push('[fast]')
    }

    const short = parts.join(' ')
    const named = data.provider && (data.providerCount ?? 0) > 1 ? `(${data.provider}) ${short}` : short

    return { full: this.theme.fg('dim', named), short: this.theme.fg('dim', short) }
  }
}
