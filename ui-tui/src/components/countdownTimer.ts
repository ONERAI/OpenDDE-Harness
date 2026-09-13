// The one timer the prompts use.
//
// pi's CountdownTimer decrements a counter once a second, which drifts: a
// suspended process (an external editor, a stopped terminal) resumes with the same
// number of seconds left it had when it went away, so the prompt outlives the
// broker's deadline it was drawn from. This one holds the absolute deadline and
// recomputes from the clock, and the clock is injected so tests never sleep.

import type { TUI } from '@earendil-works/pi-tui'

/** Epoch milliseconds and a repeating callback. `every` returns a cancel that
 *  is safe to call more than once. */
export interface CountdownClock {
  every(ms: number, callback: () => void): () => void
  nowMs(): number
}

/** Date.now plus setInterval, for the app. */
export function systemClock(): CountdownClock {
  return {
    every(ms, callback) {
      const handle = setInterval(callback, ms)

      handle.unref?.()

      let cleared = false

      return () => {
        if (cleared) {
          return
        }

        cleared = true
        clearInterval(handle)
      }
    },
    nowMs: () => Date.now()
  }
}

const TICK_MS = 1000

export class CountdownTimer {
  private cancel: (() => void) | undefined
  private expired = false

  constructor(
    private readonly deadlineMs: number,
    private readonly clock: CountdownClock,
    private readonly tui: TUI | undefined,
    private readonly onTick: (seconds: number) => void,
    private readonly onExpire: () => void
  ) {
    this.cancel = this.clock.every(TICK_MS, () => this.checkNow())
    this.checkNow()
  }

  /**
   * Whole seconds left, floored and never negative.
   *
   * Floored rather than rounded up: the broker's `expires_at` is a fractional
   * `time.time()` reading, so a 30-second window measured against this process's
   * millisecond clock is a hair over 30 and `ceil` displayed 31. A countdown
   * that claims more time than the backend will honour is the one error worth
   * ruling out, so this never shows more than is actually left.
   */
  remaining(): number {
    return Math.max(0, Math.floor((this.deadlineMs - this.clock.nowMs()) / 1000))
  }

  /**
   * Recompute now rather than at the next tick. Call it on resume and
   * immediately before accepting an answer: the second between ticks is
   * exactly where a late acceptance would otherwise slip through.
   */
  checkNow(): void {
    if (this.expired) {
      return
    }

    const seconds = this.remaining()

    this.onTick(seconds)
    this.tui?.requestRender()

    if (this.clock.nowMs() >= this.deadlineMs) {
      this.expired = true
      this.dispose()
      this.onExpire()
    }
  }

  get isExpired(): boolean {
    return this.expired || this.clock.nowMs() >= this.deadlineMs
  }

  dispose(): void {
    this.cancel?.()
    this.cancel = undefined
  }
}
