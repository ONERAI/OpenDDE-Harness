// What the UI says while it is busy: a face, the turn's activity verb, how long
// it has been going, and how to stop it.
//
//   (◔_◔)  grafting CDRs… (12s · esc to interrupt)
//
// It draws on its own line above the editor, the way Claude Code's does, rather
// than inside the editor's border — the owner's call. `StatusIndicator` is the
// content; `StatusLine` is the slot it lives in, which is zero rows high
// whenever nothing is running.
//
// The faces are the kaomoji the Ink UI wore before this one, and
// they are the only ones: the owner's call is that this is how the product
// looks while it works, not a setting.

import type { Component, LoaderIndicatorOptions, TUI } from '@earendil-works/pi-tui'

import { Loader, truncateToWidth, visibleWidth } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'
import type { CountdownClock } from './countdownTimer.js'

import { FACES } from '../content/faces.js'

export type StatusKind = 'retry' | 'working'

/** A face that flickers is unreadable, so they change on their own slow beat
 *  rather than on a spinner's. */
const FACE_INTERVAL_MS = 2500

/**
 * The shimmer, as Claude Code draws it: a band of {@link SHIMMER_BAND}
 * characters in a lighter tint of the text's own colour sweeps the face and
 * the verb one character every {@link SHIMMER_MS} (Claude Code steps every
 * 50 ms; the owner asked for half that pace). The sweep starts
 * {@link SHIMMER_LEAD} characters before the run and ends as far past it, so
 * for most of each cycle the band is off the text and the line rests: that
 * pause, not the step, is what makes it read as gentle. The rest of the run
 * keeps its colour. The line redraws on this beat, which also keeps its clock
 * honest; a kaomoji itself only changes every 2.5 s.
 */
export const SHIMMER_MS = 100
export const SHIMMER_BAND = 3
export const SHIMMER_LEAD = 10

/** Which character the band is centred on at `tick`, over a run of `length`
 *  characters: from {@link SHIMMER_LEAD} before the run to as far past it. */
export function glimmerAt(tick: number, length: number): number {
  const cycle = Math.max(1, length + 2 * SHIMMER_LEAD)

  return (tick % cycle) - SHIMMER_LEAD
}

/** Whether character `index` lies in the band centred on `glimmer`. */
export function inBand(index: number, glimmer: number): boolean {
  return Math.abs(index - glimmer) <= (SHIMMER_BAND - 1) / 2
}

/** `45s`, `2m 05s`, `1h 12m`. */
export function formatElapsed(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(whole / 3600)
  const minutes = Math.floor((whole % 3600) / 60)
  const rest = whole % 60

  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }

  return minutes > 0 ? `${minutes}m ${String(rest).padStart(2, '0')}s` : `${rest}s`
}

/**
 * The faces, and how fast they change.
 *
 * Every frame is padded to the width of the widest one: kaomoji run from five
 * to seven columns, and an unpadded set would shove the verb sideways each time
 * the face changed.
 */
export function indicatorOptions(): LoaderIndicatorOptions {
  const width = FACES.reduce((max, frame) => Math.max(max, visibleWidth(frame)), 1)

  return {
    frames: FACES.map(frame => frame + ' '.repeat(Math.max(0, width - visibleWidth(frame)))),
    intervalMs: FACE_INTERVAL_MS
  }
}

export interface StatusIndicatorOptions {
  /** How to stop what is running, e.g. `esc to interrupt`. Read per render so a
   *  rebound key is never described wrongly. */
  hint?: () => string
  kind?: StatusKind
  /** What is being waited on: the turn's activity verb, or a retry's reason. */
  message: string
  /** Epoch milliseconds; tests inject one so the elapsed count is exact. */
  now?: () => number
}

/**
 * The content of the status line.
 *
 * A `Loader` subclass for its frame animation only: the timer it runs is what
 * advances the face and asks for a render. The line itself is composed per
 * render rather than cached, so a palette swapped under a running turn repaints
 * with everything else.
 */
export class StatusIndicator extends Loader {
  readonly kind: StatusKind

  private readonly activity: string
  private readonly hint: (() => string) | undefined
  private readonly now: () => number
  private paint?: (text: string) => string
  private readonly startedAt: number
  private readonly theme: Theme

  constructor(tui: TUI, theme: Theme, options: StatusIndicatorOptions) {
    const kind = options.kind ?? 'working'

    super(
      tui,
      text => theme.fg(kind === 'retry' ? 'warn' : 'accent', text),
      text => text,
      options.message,
      indicatorOptions()
    )
    this.activity = options.message
    this.hint = options.hint
    this.kind = kind
    this.now = options.now ?? Date.now
    this.paint = text => theme.fg(kind === 'retry' ? 'warn' : 'accent', text)
    this.startedAt = this.now()
    this.theme = theme
  }

  /** How long this indicator has been up, in whole seconds. One indicator lasts
   *  a whole turn — a retry inside it does not restart the count. */
  get elapsedSeconds(): number {
    return Math.max(0, Math.floor((this.now() - this.startedAt) / 1000))
  }

  /** The shimmer tick this render falls in, read off the clock so a test can
   *  move it. */
  get tick(): number {
    return Math.max(0, Math.floor((this.now() - this.startedAt) / SHIMMER_MS))
  }

  /**
   * The face and the verb as one run of characters, with the band across it.
   *
   * Painted per character, as Claude Code paints its verb: a character in the
   * band gets the accent's lighter step, the rest the run's own colour. A
   * retry's run is amber, and its band is the same amber in bold.
   */
  private shimmered(run: string): string {
    const theme = this.theme
    const chars = Array.from(run)
    const glimmer = glimmerAt(this.tick, chars.length)
    const base = (text: string) => theme.fg(this.kind === 'retry' ? 'warn' : 'accent', text)
    const band = (text: string) => (this.kind === 'retry' ? theme.bold(base(text)) : theme.fg('shimmer', text))

    return chars.map((char, index) => (inBand(index, glimmer) ? band(char) : base(char))).join('')
  }

  /** The whole line, clipped to the width it was given. */
  renderLine(width: number): string {
    if (width <= 0) {
      return ''
    }

    const theme = this.theme
    const label = this.activity
    const details = [formatElapsed(this.elapsedSeconds), this.hint?.()].filter(Boolean).join(' · ')
    const text = [
      this.shimmered([this.frameText(), label].filter(Boolean).join(' ')),
      details ? theme.fg('muted', `(${details})`) : ''
    ]
      .filter(Boolean)
      .join(' ')

    return truncateToWidth(text, width, '')
  }

  /** Stop the animation timer. Nothing else holds the process open. */
  dispose(): void {
    this.stop()
  }

  /** The current face, unpainted: the shimmer paints it character by
   *  character with the verb. */
  private frameText(): string {
    return super.getRenderedIndicator()
  }

  /**
   * Paint the face in the theme's own color, for the base class's own render
   * path. Guarded because the base constructor renders once before this
   * subclass's fields exist.
   */
  protected override getRenderedIndicator(): string {
    return this.paint?.(super.getRenderedIndicator()) ?? super.getRenderedIndicator()
  }
}

export interface StatusLineOptions {
  /** Drives the redraw on every shimmer step. Tests pass a manual one. */
  clock: CountdownClock
  tui: TUI
}

/**
 * The row between the queued messages and the editor.
 *
 * Two rows high, always, whatever the width and whether or not anything is
 * running: pi's own shape, where the working indicator is a blank row and a
 * line (`Loader.render`) and an idle screen puts pi's `IdleStatus` -- two
 * blank rows -- in its place. The height never changes, so the editor and
 * everything under it never shift when a turn starts or ends, and the reply
 * above never sits directly on the editor's rule.
 */
export class StatusLine implements Component {
  private indicator: StatusIndicator | undefined
  /** A painted one-line note to show while nothing is running. */
  private notice: string | undefined
  private readonly options: StatusLineOptions
  private stopClock: (() => void) | undefined
  private stopNotice: (() => void) | undefined

  constructor(options: StatusLineOptions) {
    this.options = options
  }

  get active(): boolean {
    return this.indicator !== undefined
  }

  /** Show this indicator, or clear the line with `undefined`. */
  set(indicator: StatusIndicator | undefined): void {
    if (indicator === this.indicator) {
      return
    }

    this.indicator = indicator
    this.stopClock?.()
    this.stopClock = indicator
      ? this.options.clock.every(SHIMMER_MS, () => this.options.tui.requestRender())
      : undefined
    this.options.tui.requestRender()
  }

  /**
   * A note in the row for as long as nothing is running.
   *
   * Shown for `ms` and then gone by itself. The armed-quit hint is what this
   * exists for: it has to disappear when the arm does, and an idle screen
   * redraws for nothing else, so the row would otherwise keep offering a
   * second press that no longer means anything.
   *
   * The text arrives painted. This slot holds no theme, and a note that lives
   * two seconds is not worth giving it one.
   */
  setNotice(text: null | string, ms?: number): void {
    this.stopNotice?.()
    this.stopNotice = undefined
    this.notice = text ?? undefined

    if (this.notice !== undefined && ms !== undefined && ms > 0) {
      this.stopNotice = this.options.clock.every(ms, () => this.setNotice(null))
    }

    this.options.tui.requestRender()
  }

  /** Let go of the row and the once-a-second redraw behind it. The animation
   *  belongs to the indicator; this owns the clock, and both have to stop. */
  dispose(): void {
    this.indicator?.dispose()
    this.setNotice(null)
    this.set(undefined)
  }

  invalidate(): void {
    this.indicator?.invalidate()
  }

  render(width: number): string[] {
    // The note outranks the activity, for the two seconds it lives.
    //
    // It is the other half of a key that is about to exit the application, and
    // a promise that it has been offered first. A gateway command borrows this
    // row without being a turn, so the ladder reads an idle prompt and arms;
    // with the activity on top the offer was made invisibly and the next press
    // exited with nothing ever having been shown. Whatever is running is still
    // running and takes the row back when the offer lapses.
    const line = this.notice === undefined ? this.indicator?.renderLine(width) : truncateToWidth(this.notice, width)

    // Idle: pi's `IdleStatus`, two rows of spaces. Spaces rather than empty
    // strings because these rows are standing in for something that was drawn
    // there a moment ago, which is what they are for.
    if (line === undefined) {
      const blank = ' '.repeat(width)

      return [blank, blank]
    }

    // Busy: pi-tui's `Loader` shape, a blank row and the line. The blank keeps
    // the last line of the reply off the row; the row below it is the
    // editor's own rule, as it is in pi.
    return ['', line]
  }
}
