// Text that takes its colour from the theme when it renders, not when it is
// built.
//
// A component that stores an already-painted string freezes the palette that
// was current when it was written. The OSC 11 background reply lands up to
// 200 ms after boot and swaps the whole palette (see entry.ts), and
// `tui.invalidate()` cannot repaint a string that no longer knows which token
// it came from — so the transcript keeps dark-theme text on a light terminal.

import type { Component } from '@earendil-works/pi-tui'

import { Text, truncateToWidth } from '@earendil-works/pi-tui'

/** Paints raw text with the palette in force *now*. Closes over the mutable
 *  `Theme`, so it answers differently after `setScheme`. */
export type Paint = (text: string) => string

/** Wrapped text (pi's `Text`), repainted whenever the palette has moved. */
export class ThemedText extends Text {
  private painted?: string

  constructor(
    private readonly paint: Paint,
    private raw: string,
    paddingX = 0,
    paddingY = 0
  ) {
    super('', paddingX, paddingY)
  }

  setRaw(raw: string): void {
    this.raw = raw
  }

  override render(width: number): string[] {
    const painted = this.paint(this.raw)

    // `setText` drops pi's wrap cache, so only touch it when the paint moved.
    if (painted !== this.painted) {
      this.painted = painted
      this.setText(painted)
    }

    return super.render(width)
  }
}

/** One line, truncated rather than wrapped, repainted the same way. Cached on
 *  the painted text and the width, so a palette swap invalidates it by itself. */
export class ThemedLine implements Component {
  private cachedLines?: string[]
  private cachedText?: string
  private cachedWidth?: number

  constructor(
    private readonly paint: Paint,
    private readonly raw: string,
    private readonly paddingX = 0
  ) {}

  invalidate(): void {
    this.cachedLines = undefined
    this.cachedText = undefined
    this.cachedWidth = undefined
  }

  render(width: number): string[] {
    const painted = this.paint(this.raw)

    if (this.cachedLines && this.cachedText === painted && this.cachedWidth === width) {
      return this.cachedLines
    }

    const inset = ' '.repeat(this.paddingX)

    this.cachedText = painted
    this.cachedWidth = width
    this.cachedLines = [inset + truncateToWidth(painted, Math.max(1, width - this.paddingX * 2))]

    return this.cachedLines
  }
}
