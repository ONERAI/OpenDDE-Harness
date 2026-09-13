// What the user sent, echoed into the transcript.
//
// Claude Code's shape: a prompt glyph, a space, the text, and a tinted band
// running the full width of the terminal for exactly as many rows as the text
// occupies. pi pads its block a row above and below, which for one line of
// text is three rows of colour; the row count here is the line count and
// nothing else.
//
// The text is echoed literally. What the user typed is not a document to
// render — `**bold**` was typed, so `**bold**` is shown — so this wraps and
// does nothing else to it.

import type { Component } from '@earendil-works/pi-tui'

import { truncateToWidth, visibleWidth, wrapTextWithAnsi } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

const OSC133_ZONE_START = '\x1b]133;A\x07'
const OSC133_ZONE_END = '\x1b]133;B\x07'
const OSC133_ZONE_FINAL = '\x1b]133;C\x07'

/** The glyph in front of what was sent. */
const PROMPT = '❯'

/** Where the text starts: the glyph and the space after it. Wrapped lines are
 *  indented to the same column, so the message reads as one block rather than
 *  as a first line with a hanging remainder. */
const GUTTER = visibleWidth(PROMPT) + 1

export class UserMessage implements Component {
  readonly text: string

  private cachedKey = ''
  private cachedLines: string[] | undefined
  /** Cells of inset before the glyph. */
  private readonly paddingX: number
  private readonly theme: Theme

  /**
   * Whether the gateway is believed to hold this exchange.
   *
   * `session.undo` removes the last *saved* exchange, which is not always the
   * last one on screen: a turn that failed before it produced anything leaves
   * an echo here and nothing there. `/undo` and `/retry` read this so they act
   * on the exchange the gateway actually removed. True until something proves
   * otherwise, because every stored message replayed on resume is saved.
   */
  private stored = true

  constructor(theme: Theme, text: string, paddingX = 1) {
    this.text = text
    this.theme = theme
    this.paddingX = paddingX
  }

  /** Whether the gateway is believed to hold an exchange for this prompt. */
  get saved(): boolean {
    return this.stored
  }

  markUnsaved(): void {
    this.stored = false
  }

  invalidate(): void {
    this.cachedKey = ''
    this.cachedLines = undefined
  }

  render(width: number): string[] {
    // The palette is part of the key, not only the width: these lines are
    // painted once and held, and the palette can still move under them. A
    // caller that forgets to invalidate would otherwise leave this band in the
    // old palette while everything painted later is correct.
    const key = `${width}:${this.theme.scheme}`

    if (this.cachedLines && this.cachedKey === key) {
      return this.cachedLines
    }

    this.cachedKey = key
    this.cachedLines = this.build(Math.max(1, width))

    return this.cachedLines
  }

  private build(width: number): string[] {
    const inset = Math.min(this.paddingX, Math.max(0, width - 1))
    const content = Math.max(1, width - inset - GUTTER)
    const wrapped = wrapTextWithAnsi(this.text, content)

    if (wrapped.length === 0) {
      return []
    }

    const t = this.theme
    const pad = ' '.repeat(inset)
    const lines = wrapped.map((line, index) => {
      const head = index === 0 ? `${t.fg('prompt', PROMPT)} ` : ' '.repeat(GUTTER)
      // Clipped before it is filled: a terminal narrower than the glyph and a
      // character still gets a band that fits in it.
      const row = truncateToWidth(pad + head + line, width, '')
      // Every row runs edge to edge: the band is the full width of the
      // terminal, and the text sits at the left of it.
      const fill = ' '.repeat(Math.max(0, width - visibleWidth(row)))

      return t.bg('userMessageBg', row + fill)
    })

    lines[0] = OSC133_ZONE_START + lines[0]
    lines[lines.length - 1] = OSC133_ZONE_END + OSC133_ZONE_FINAL + lines[lines.length - 1]!

    // One blank in front, none behind: every block below leads with its own,
    // and so does the status row, so a trailing one here would double up.
    return ['', ...lines]
  }
}
