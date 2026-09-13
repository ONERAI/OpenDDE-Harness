// The furniture: the session panel above the transcript, the busy row under
// it, the slot the editor (or a picker standing in for it) lives in, and the
// footer at the bottom.
//
// Layout, top to bottom: session panel → transcript → queued messages → the
// busy indicator's own row → editor → footer.
//
// The footer draws from live state rather than being told when to change, so
// what it reads is a function of the session's facts and the turn's counters.

import { Container, type TUI } from '@earendil-works/pi-tui'

import type { CountdownClock } from '../components/countdownTimer.js'
import type { UsageTotals } from '../turn.js'
import type { SessionFacts } from './sessionFacts.js'

import { Footer } from '../components/footer.js'
import { SessionPanel } from '../components/sessionPanel.js'
import { StatusLine } from '../components/statusIndicator.js'
import { type Theme } from '../theme.js'

export interface LayoutOptions {
  clock: CountdownClock
  facts: () => SessionFacts
  /** Whether the footer prints its counters at all (`tui.show_token_usage`). */
  showTokenUsage: () => boolean
  theme: Theme
  tui: TUI
  /** The turn model's figures, or null before a turn has reported any. */
  usage: () => null | UsageTotals
}

export class Layout {
  /** The editor slot: the editor, or the picker or prompt standing in for it. */
  readonly editorContainer = new Container()
  readonly footer: Footer
  readonly panel: SessionPanel
  /** The busy indicator's row, between the queued messages and the editor. */
  readonly status: StatusLine

  constructor(opts: LayoutOptions) {
    this.status = new StatusLine({ clock: opts.clock, tui: opts.tui })
    this.panel = new SessionPanel(opts.theme, {}, { rows: () => opts.tui.terminal.rows })
    this.footer = new Footer(opts.theme, () => {
      const facts = opts.facts()
      const usage = opts.usage()

      return {
        autoCompact: facts.info.autoCompact ?? false,
        branch: facts.branchName(),
        cwd: facts.cwd,
        effort: facts.info.effort,
        fast: facts.info.fast,
        model: facts.info.model,
        provider: facts.info.provider,
        showTokenUsage: opts.showTokenUsage(),
        subscription: facts.info.subscription ?? false,
        ...(facts.update ? { update: facts.update } : {}),
        usage: {
          cacheHitPercent: usage?.cacheHitPercent ?? null,
          // The turn model's figure, which is null while a compaction leaves
          // the window unmeasured; before any snapshot at all, what the session
          // was opened holding.
          contextTokens: usage ? usage.contextUsed : facts.info.contextTokens,
          contextWindow: usage?.contextMax || facts.info.contextWindow,
          // The session's total, not the last turn's: every completion carries
          // the gateway's own running sum, which is what `/status` reports too.
          // Before the first completion of a resumed session the turn model has
          // counted nothing, and what the session opened having spent is the
          // whole answer — the same sum, from the same tracker, so the two
          // never disagree and the footer never opens a long conversation at
          // zero. The list price is the figure pi shows; the vendor's own is
          // null on a plan, and `/status` is where that one belongs.
          costUsd: usage?.listCost ?? usage?.cost ?? facts.opening.listCost ?? facts.opening.cost ?? 0,
          input: usage?.input || facts.opening.input,
          output: usage?.output || facts.opening.output,
          outputEstimated: usage?.outputEstimated ?? false
        }
      }
    })
  }
}
