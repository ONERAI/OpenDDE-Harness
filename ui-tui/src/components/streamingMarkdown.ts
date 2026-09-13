// SPDX-License-Identifier: MIT
// Block-boundary rules adapted from OpenDDE's StreamingMd (MIT).
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See LICENSES/README.md and LICENSES/MIT-hermes-agent.txt.
//
// pi-tui's `Markdown` re-lexes its entire text on every `setText`, so a long
// streamed reply pays for the whole message on each delta. This splits the
// stream at stable top-level block boundaries: a completed block becomes its
// own `Markdown` child, which never re-parses again, and only the open tail
// block is re-parsed per delta.
//
// Output is byte-identical to a whole-text `Markdown` at every prefix. Blocks
// are independent of each other, which holds for every Markdown construct but
// two: a link reference definition (`[ref]: url`) resolves document-wide, in
// both directions, so `[label][ref]` in one block can depend on a definition
// in another; and an HTML comment swallows the blank lines that would
// otherwise end a block. Either one anywhere in the reply turns the splitting
// off for that reply and the whole text is rendered by a single `Markdown` —
// the same output pi would produce, at pi's cost. See the tests.
//
// The theme's `highlightCode` is passed through untouched, so it is still
// called once per fenced block and a highlighter that tracks state across
// lines (an open `/* … */`) sees the block it is highlighting.

import type { DefaultTextStyle, MarkdownOptions, MarkdownTheme } from '@earendil-works/pi-tui'

import { Container, Markdown } from '@earendil-works/pi-tui'

/**
 * Constructs whose blocks are not independent of each other.
 *
 * - A link reference definition (`[ref]: url`) resolves across the whole
 *   document, in both directions, so a use in one block can depend on a
 *   definition in another. Line-anchored.
 * - An HTML comment runs until `-->`, blank lines and all, so what looks like
 *   a top-level boundary inside one is not a boundary at all.
 *
 * A false positive costs the splitting for one reply and never costs
 * correctness: the whole text is rendered by a single `Markdown` instead.
 */
const NONLOCAL_CONSTRUCT = /^ {0,3}\[[^\]\n]*\]:|<!--/m

/**
 * Offsets in `text` where a top-level block provably ends.
 *
 * Only the open tail is inspected. A blank line inside a fence, display math
 * or a loose list is not a boundary, and a boundary is only declared once the
 * *following* line has arrived in full — a partial indent or list marker would
 * otherwise look like the start of a new top-level block.
 */
function stableEnds(text: string): number[] {
  const ends: number[] = []

  let code: { length: number; marker: string } | undefined
  let math: '$$' | '\\[' | undefined
  let list = false
  let quote = false
  let trailingMath = false
  let start = 0

  for (let i = 0; i < text.length;) {
    const nl = text.indexOf('\n', i)

    if (nl < 0) {
      break
    }

    const line = text.slice(i, nl).replace(/\r$/, '')
    const trimmed = line.trim()

    if (trimmed) {
      trailingMath = false
    }

    const fence = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line)

    if (code) {
      if (fence && fence[1][0] === code.marker && fence[1].length >= code.length && !fence[2].trim()) {
        code = undefined
      }
    } else if (math) {
      if (math === '$$' ? trimmed.endsWith('$$') : trimmed.endsWith('\\]')) {
        math = undefined
        trailingMath = true
      }
    } else if (fence && !(fence[1][0] === '`' && fence[2].includes('`'))) {
      code = { length: fence[1].length, marker: fence[1][0] }
    } else if (trimmed.startsWith('$$') && !(trimmed.length >= 4 && trimmed.endsWith('$$'))) {
      math = '$$'
    } else if (trimmed.startsWith('\\[') && !trimmed.endsWith('\\]')) {
      math = '\\['
    }

    if (!code && !math) {
      trailingMath ||= /^(?:\$\$.*\$\$|\\\[.*\\\])$/.test(trimmed)
      list ||= /^ {0,3}(?:[-+*]|\d+[.)])(?:\s|$)/.test(line)
      quote ||= /^ {0,3}>/.test(line)
    }

    i = nl + 1

    // pi consumes blank lines after display math; keep the following block with
    // it so its next-token-dependent spacing stays correct.
    if (trimmed || code || math || trailingMath) {
      continue
    }

    // A run of whitespace-only lines belongs to the block before it.
    const following = /^(?:[ \t]*\r?\n)*([^\n]+)\n/.exec(text.slice(i))

    if (!following) {
      continue
    }

    const next = following[1].replace(/\r$/, '')

    if (!next.trim() || /^[ \t]/.test(next)) {
      continue
    }

    if (list && /^(?:[-+*]|\d+[.)])(?:\s|$)/.test(next)) {
      continue
    }

    if (quote && next.startsWith('>')) {
      continue
    }

    const end = i + following[0].length - following[1].length - 1

    if (text.slice(start, end).trim()) {
      ends.push(end)
      start = end
      list = quote = false
    }

    i = end
  }

  return ends
}

/**
 * A `Markdown` that grows by appending. Completed top-level blocks are frozen
 * as their own children; the tail is the only one that re-parses.
 *
 * `setText` must be called with a superset of what came before. Anything else
 * (including any call after `finish`) rebuilds the component from scratch, so
 * replacing the text is allowed, just not free.
 *
 * A reply that defines a link reference, or opens an HTML comment, stops being
 * split from that delta on and renders as one `Markdown` instead, because its
 * blocks are no longer independent of one another.
 */
export class StreamingMarkdown extends Container {
  private committed = 0
  private finished = false
  private readonly options: MarkdownOptions
  /** How much of the text has already been examined for a reference
   *  definition. Always a line start, so only what arrived since the last
   *  delta — plus the line still being written — is ever rescanned. */
  private scanned = 0
  private tail: Markdown
  private tailText = ''
  private text = ''
  /** A construct with a document-wide dependency turned the splitting off. */
  private whole = false

  constructor(
    text: string,
    private readonly paddingX: number,
    private readonly theme: MarkdownTheme,
    private readonly defaultTextStyle?: DefaultTextStyle,
    options?: MarkdownOptions
  ) {
    super()

    this.options = { ...options }
    this.tail = this.makeMarkdown()
    this.addChild(this.tail)
    this.setText(text)
  }

  setText(text: string): void {
    if (text === this.text) {
      return
    }

    if (this.finished || !text.startsWith(this.text)) {
      this.reset()
    }

    this.text = text

    if (!this.whole && this.opensNonlocalConstruct(text)) {
      this.renderWhole()
    }

    const pending = text.slice(this.committed)
    // A caller-supplied transform can depend on the whole reply, and so can a
    // reference definition, so neither allows anything to be frozen ahead of
    // it.
    const splits = this.whole || this.options.transform ? [] : stableEnds(pending)
    let previous = 0

    for (const end of splits) {
      this.tail.setText(pending.slice(previous, end))
      this.tail = this.makeMarkdown()
      this.addChild(this.tail)
      this.tailText = ''
      previous = end
    }

    this.committed += previous

    const tail = pending.slice(previous)

    if (tail !== this.tailText) {
      this.tail.setText(tail)
      this.tailText = tail
    }
  }

  /**
   * Stop accepting appends. Deliberately does NOT re-parse the whole text: the
   * children already render byte-for-byte what a whole-text `Markdown` would,
   * and re-parsing a long reply here is a 30-90 ms stall at end of turn.
   */
  finish(): void {
    this.finished = true
  }

  /**
   * Whether the text opens a construct whose blocks depend on each other,
   * examining only what has not been examined before. The line still being
   * written is left unscanned so the next delta sees it whole.
   */
  private opensNonlocalConstruct(text: string): boolean {
    const from = this.scanned
    const region = text.slice(from)
    const lastBreak = region.lastIndexOf('\n')

    this.scanned = lastBreak < 0 ? from : from + lastBreak + 1

    return NONLOCAL_CONSTRUCT.test(region)
  }

  private makeMarkdown(): Markdown {
    return new Markdown('', this.paddingX, 0, this.theme, this.defaultTextStyle, this.options)
  }

  /** Give up on splitting and hand the whole text to one `Markdown`: a block
   *  in it depends on another, so only the whole document renders correctly. */
  private renderWhole(): void {
    this.whole = true
    this.clear()
    this.committed = 0
    this.tailText = ''
    this.tail = this.makeMarkdown()
    this.addChild(this.tail)
  }

  private reset(): void {
    this.clear()
    this.committed = 0
    this.finished = false
    this.scanned = 0
    this.text = ''
    this.tailText = ''
    this.whole = false
    this.tail = this.makeMarkdown()
    this.addChild(this.tail)
  }
}
