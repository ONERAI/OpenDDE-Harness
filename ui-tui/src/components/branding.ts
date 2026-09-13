// The brand lockup the first screen opens with: the hero mark on the left, the
// stacked OPENDDE / HARNESS wordmark on the right, and the product name under
// both. Ported from the Ink TUI's `banner.ts` + `components/branding.tsx`; the
// art is byte-for-byte the one that ships today.
//
// Two rules the old version also followed. The art is a bitmap, not a picture:
// every cell is either filled or not, and a scale factor downsamples it through
// half-block characters, so the same source draws at three sizes and the hero
// and the wordmark always shrink together. And no color is baked in — the
// wordmark's gradient is bands of the theme's brand ramp and the hero is one
// theme token, so a 16-color terminal and a light background both get something
// the palette chose.

import type { Component } from '@earendil-works/pi-tui'

import { truncateToWidth, visibleWidth } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

import { BRAND } from '../content/brand.js'

/** Bitmap rows per letter, one character per cell. */
const BLOCK_GLYPHS: Record<string, readonly string[]> = {
  A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
  D: ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
  E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
  H: ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
  N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
  O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
  P: ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
  R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
  S: ['01111', '10000', '10000', '01110', '00001', '00001', '11110']
}

/** The product name, one word per line: the single-line form would need ~130
 *  columns before any downsampling, which no terminal we target has. */
const WORDMARK_WORDS = ['OPENDDE', 'HARNESS'] as const

/** Each bitmap cell is drawn two columns wide, letters one cell apart. */
const CELL_BLOCK = '██'
const CELL_EMPTY = '  '
const LETTER_GAP = '  '

/** Columns between the hero and the wordmark. */
const LOCKUP_GAP = 4

/** Vertical bands the wordmark's gradient is split into. The theme's ramp has
 *  one violet per band, lightest first, already tuned to the ground it is drawn
 *  on, so the bands are read straight off it. */
const WORDMARK_BANDS = 4

/** Downsampling factors, largest art first. */
const ART_SCALES = [1, 2, 4] as const

export type ArtScale = (typeof ART_SCALES)[number]

/** Below this the art is dropped whatever the scale: the smallest lockup would
 *  fit, but a banner that fills a narrow window is in the way rather than
 *  welcoming. The one-line lockup stands in for it. */
const MIN_ART_COLUMNS = 60

/** Rows the rest of the first screen needs under the banner: the panel's facts,
 *  the editor and the footer. The art is only allowed what is left. */
const CHROME_ROWS = 10

/** Terminal height to assume when nothing reports one (a pipe, a test). */
const ASSUMED_ROWS = 24

function renderWord(word: string): string[] {
  const rows = Array.from({ length: 7 }, (_, row) =>
    [...word]
      .map(letter =>
        (BLOCK_GLYPHS[letter] ?? BLOCK_GLYPHS.E)![row]!.replaceAll('1', CELL_BLOCK).replaceAll('0', CELL_EMPTY)
      )
      .join(LETTER_GAP)
  )

  // The blank row keeps the two words apart and the height a multiple of the
  // scale factors, so downsampling never merges them.
  return [...rows, '']
}

/** The wordmark at its drawn size, which is the only size it is ever drawn at.
 *  Exported so a test can assert that what reaches the screen is this and not a
 *  downsampling of it. */
export const WORDMARK_ART: readonly string[] = WORDMARK_WORDS.flatMap(word => renderWord(word))

const HERO_ART: readonly string[] = [
  '                                      ▄████',
  '                                  ▄███▀ ███',
  '                              ▄███▀     ███',
  '                          ▄███▀          ███',
  '                       ▄██▀              ███',
  '       ████████                     ████████',
  '        ████████                   ████████ ',
  '         ████████                 ████████  ',
  '          ████████               ████████   ',
  '  ▄██▄     ████████             ████████    ',
  ' ██▀        ████████           ████████     ',
  '██▀          ████████         ████████      ',
  '██            ████████       ████████       ',
  '██             ████████     ████████        ',
  '██              ████████████████████         ',
  '██                ████████████████           ',
  ' ██▄                ████████████             ',
  '  ▀██▄                 ██████                ',
  '    ▀███▄               ██████               ',
  '       ▀████▄▄           ██████               ',
  '            ▀████▄▄      ██████               ',
  '                 ▀█████████████▄▄▄▄▄          ',
  '                       ▀▀▀▀▀▀▀██████▀          ',
  ''
].map(row => row.trimEnd())

function artWidth(rows: readonly string[]): number {
  return rows.reduce((width, row) => Math.max(width, [...row].length), 0)
}

const HERO_COLUMNS = artWidth(HERO_ART)
const WORDMARK_COLUMNS = artWidth(WORDMARK_ART)

const isFilled = (row: readonly string[], col: number) => {
  const ch = row[col]

  return ch !== undefined && ch !== ' '
}

/**
 * Downsample block art by `factor`: one output cell covers a factor-wide,
 * factor-tall block of the bitmap, and the upper and lower halves of that block
 * become the two halves of one terminal cell. Factor 2 therefore turns two
 * source rows into one row of half blocks, which is how the same mark draws at
 * three sizes without a second drawing.
 *
 * A half of the block is drawn when any of it is filled. That dilates — it
 * grows every stroke by up to a whole output cell and closes the gaps between
 * them — which is what turned the wordmark's letterforms into a smudge. The
 * wordmark is no longer downsampled at all, and for the hero dilation is the
 * right rule: a coverage threshold of half was tried and it breaks the mark at
 * factor 4, where the swoosh is a one-cell diagonal that covers well under half
 * of each block and disappears into a scatter of dots. Factor 2 is unaffected
 * either way (a half-block there is two source cells, so half of it is any of
 * it), so the choice only decides whether the smallest hero reads at all.
 */
export function scaleArt(rows: readonly string[], factor: number): string[] {
  if (factor <= 1) {
    return rows.map(row => row.trimEnd())
  }

  const bitmap = rows.map(row => [...row])
  const width = artWidth(rows)
  const half = Math.max(1, Math.floor(factor / 2))
  const anyFilled = (rowStart: number, rowCount: number, colStart: number) => {
    for (let r = rowStart; r < rowStart + rowCount; r++) {
      const row = bitmap[r]

      if (!row) {
        continue
      }

      for (let c = colStart; c < colStart + factor; c++) {
        if (isFilled(row, c)) {
          return true
        }
      }
    }

    return false
  }

  const out: string[] = []

  for (let i = 0; i < Math.ceil(rows.length / factor); i++) {
    let line = ''

    for (let j = 0; j < Math.ceil(width / factor); j++) {
      const top = anyFilled(i * factor, half, j * factor)
      const bottom = anyFilled(i * factor + half, factor - half, j * factor)

      line += top && bottom ? '█' : top ? '▀' : bottom ? '▄' : ' '
    }

    out.push(line.trimEnd())
  }

  return out
}

export interface LockupSize {
  cols: number
  heroCols: number
  rows: number
}

/**
 * The smallest the wordmark is drawn at.
 *
 * Halving it keeps the letters: a stroke is two cells thick in the drawing and
 * one after, and the counters — the holes in O, P, D, R — survive as single
 * cells. Quartering does not: strokes and counters both fall below one cell and
 * the dilation the downsampler applies closes them, so the word becomes a
 * block. Below this the wordmark is dropped and the hero shrinks alone.
 */
const WORDMARK_MIN_SCALE = 2

/**
 * Footprint of the lockup at a scale factor.
 *
 * Room is reserved for the wordmark only where the wordmark is drawn; at every
 * smaller size the hero stands alone.
 */
export function lockupSize(factor: number): LockupSize {
  const heroCols = Math.ceil(HERO_COLUMNS / factor)
  const heroRows = Math.ceil(HERO_ART.length / factor)

  if (factor > WORDMARK_MIN_SCALE) {
    return { cols: heroCols, heroCols, rows: heroRows }
  }

  const wordmarkCols = Math.ceil(WORDMARK_COLUMNS / factor)

  return {
    cols: heroCols + LOCKUP_GAP + wordmarkCols,
    heroCols,
    rows: Math.max(heroRows, Math.ceil(WORDMARK_ART.length / factor))
  }
}

/**
 * The largest lockup that fits the budget, or 0 for "no room — say the name".
 *
 * Factors 1 and 2 carry the wordmark; below them it is gone and what shrinks is
 * the hero alone. The hero is a mark and downsamples into a smaller mark, while
 * the wordmark is text, and at a quarter size its letterforms have no
 * resolution left to lose. The name line under the art carries the words at
 * every width, so dropping the block letters loses nothing but the drawing.
 */
export function resolveLockupScale(budgetRows: number, budgetCols: number): ArtScale | 0 {
  for (const factor of ART_SCALES) {
    const size = lockupSize(factor)

    if (size.rows <= budgetRows && size.cols <= budgetCols) {
      return factor
    }
  }

  return 0
}

export interface BrandingOptions {
  /** When that version shipped, from the gateway. Shown beside it, as the old
   *  welcome panel did; absent when the install records no date. */
  releaseDate?: null | string
  /** Terminal height. The art takes what the rest of the screen does not need. */
  rows?: () => number
  /** Shown after the product name once the gateway has said which build this is. */
  version?: string
}

/**
 * The lockup, as many rows as the terminal can spare.
 *
 * Always ends with the product name, so the brand is on screen at every width;
 * the art above it appears only when there is room for it.
 */
export class Branding implements Component {
  private cachedKey?: string
  private cachedLines?: string[]

  constructor(
    private readonly theme: Theme,
    private options: BrandingOptions = {}
  ) {}

  setVersion(version: string | undefined, releaseDate?: null | string): void {
    if (version === this.options.version && releaseDate === this.options.releaseDate) {
      return
    }

    this.options = { ...this.options, releaseDate, version }
    this.invalidate()
  }

  invalidate(): void {
    this.cachedKey = undefined
    this.cachedLines = undefined
  }

  render(width: number): string[] {
    const rows = this.options.rows?.() ?? process.stdout.rows ?? ASSUMED_ROWS
    // The scheme is part of the key, not an afterthought: this art is painted
    // once and held, and the palette can still move under it — the terminal's
    // background reply lands after the app is built, and `tui.theme` lands
    // after that. Without it the lockup keeps whatever palette the boot
    // guessed while everything painted later is correct.
    const key = [width, rows, this.theme.scheme, this.options.version ?? '', this.options.releaseDate ?? ''].join(':')

    if (this.cachedLines && this.cachedKey === key) {
      return this.cachedLines
    }

    const lines = [...this.art(width, rows), this.nameLine()]

    this.cachedKey = key
    this.cachedLines = lines.map(line => truncateToWidth(line, width))

    return this.cachedLines
  }

  /** `ϒ  OpenDDE Harness v0.4.2 (2026-09-10)`. */
  private nameLine(): string {
    const t = this.theme
    const version = this.options.version ? ` v${this.options.version}` : ''
    // The date belongs to the version: without one there is nothing to date.
    const released = version && this.options.releaseDate ? ` (${this.options.releaseDate})` : ''

    return (
      t.bold(t.fg('accent', BRAND.icon)) +
      t.bold(t.fg('primary', `  ${BRAND.name}${version}`)) +
      (released ? t.fg('muted', released) : '')
    )
  }

  private art(width: number, rows: number): string[] {
    if (width < MIN_ART_COLUMNS) {
      return []
    }

    const factor = resolveLockupScale(rows - CHROME_ROWS, width)

    if (factor === 0) {
      return []
    }

    const size = lockupSize(factor)
    const hero = scaleArt(HERO_ART, factor)
    const wordmark = factor <= WORDMARK_MIN_SCALE ? scaleArt(WORDMARK_ART, factor) : []

    // Centred against each other, the way the Ink row did it.
    const heroOffset = Math.max(0, Math.floor((size.rows - hero.length) / 2))
    const markOffset = Math.max(0, Math.floor((size.rows - wordmark.length) / 2))
    const rowsPerBand = Math.max(1, Math.ceil(wordmark.length / WORDMARK_BANDS))
    const out: string[] = []

    for (let i = 0; i < size.rows; i++) {
      const heroRow = hero[i - heroOffset] ?? ''
      const markRow = wordmark[i - markOffset] ?? ''
      const gap = ' '.repeat(size.heroCols - visibleWidth(heroRow) + LOCKUP_GAP)
      const band = Math.min(WORDMARK_BANDS - 1, Math.floor((i - markOffset) / rowsPerBand))

      const left = heroRow ? this.theme.fg('primary', heroRow) : ''
      const right = markRow ? this.theme.ramp(Math.max(0, band), markRow) : ''

      // The gap only exists to put the wordmark where it belongs; a row with
      // nothing on the right ends at the hero rather than in trailing blanks.
      out.push(right ? left + gap + right : left)
    }

    // A trailing blank row so the facts below do not sit against the art.
    out.push('')

    return out
  }
}
