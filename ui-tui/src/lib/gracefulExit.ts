// Ported from ui-tui/src/lib/gracefulExit.ts.

type HeldSignal = 'SIGHUP' | 'SIGINT' | 'SIGTERM'

interface SetupOptions {
  cleanups?: (() => Promise<void> | void)[]
  /** How the process ends, after the cleanups or after the failsafe. Injected
   *  by the tests, which cannot let the real one run. */
  exit?: (code: number) => void
  failsafeMs?: number
  onError?: (scope: 'uncaughtException' | 'unhandledRejection', err: unknown) => void
  onSignal?: (signal: NodeJS.Signals) => void
}

const SIGNAL_EXIT_CODE: Record<HeldSignal, number> = {
  SIGHUP: 129,
  SIGINT: 130,
  SIGTERM: 143
}

let wired = false
let deferrals = 0
let sigintOwners = 0
let pendingSignal: HeldSignal | undefined
let replaySignal: ((code: number, signal: NodeJS.Signals) => void) | undefined

/** What a signal means right now.
 *
 *  - `exit`: nothing is deferred, so it is this process's to act on.
 *  - `drop`: a child owns the terminal and Ctrl+C was aimed at it. The child
 *    is in this process group and got its own copy; this one is not ours and
 *    is forgotten, not remembered. Replaying it when the child returns would
 *    end the session the user was only stepping out of.
 *  - `hold`: a deliberate shutdown that arrived mid-handoff. Remembered and
 *    honoured once the deferral clears.
 */
export function classifySignal(signal: HeldSignal): 'drop' | 'exit' | 'hold' {
  if (deferrals === 0) {
    return 'exit'
  }

  return signal === 'SIGINT' && sigintOwners > 0 ? 'drop' : 'hold'
}

export interface DeferSignalOptions {
  /**
   * The child, not this process, is what Ctrl+C means for the duration.
   *
   * Without this, a SIGINT that arrives while a child owns the terminal is
   * remembered and replayed as an app exit the moment the deferral clears —
   * so cancelling a login would also end the session. With it, SIGINT is
   * consumed (the child is in the same process group and got its own copy),
   * while SIGTERM and SIGHUP are still remembered and honoured afterwards.
   */
  childOwnsSigint?: boolean
}

/**
 * Stop signals from exiting this process until the returned callback runs.
 *
 * For the window where a child process owns the terminal: it is in the same
 * process group, so it receives the same Ctrl-C, and exiting here would take
 * the session down with the thing the user was interrupting.
 */
export function deferSignalExit(options: DeferSignalOptions = {}): () => void {
  const ownsSigint = options.childOwnsSigint === true

  deferrals += 1

  if (ownsSigint) {
    sigintOwners += 1
  }

  let released = false

  return () => {
    if (released) {
      return
    }

    released = true
    deferrals = Math.max(0, deferrals - 1)

    if (ownsSigint) {
      sigintOwners = Math.max(0, sigintOwners - 1)
    }

    if (deferrals === 0 && pendingSignal) {
      const signal = pendingSignal
      pendingSignal = undefined
      replaySignal?.(SIGNAL_EXIT_CODE[signal], signal)
    }
  }
}

export function setupGracefulExit({
  cleanups = [],
  exit: end = code => process.exit(code),
  failsafeMs = 4000,
  onError,
  onSignal
}: SetupOptions = {}) {
  if (wired) {
    return
  }

  wired = true

  let shuttingDown = false

  const exit = (code: number, signal?: NodeJS.Signals) => {
    if (shuttingDown) {
      return
    }

    shuttingDown = true

    if (signal) {
      onSignal?.(signal)
    }

    setTimeout(() => end(code), failsafeMs).unref?.()

    void Promise.allSettled(cleanups.map(fn => Promise.resolve().then(fn))).finally(() => end(code))
  }

  replaySignal = exit

  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const) {
    process.on(sig, () => {
      const decision = classifySignal(sig)

      if (decision === 'drop') {
        return
      }

      if (decision === 'hold') {
        // Keep the first signal: whichever arrived first is the intent to
        // honour once the deferral clears; later ones during the same handoff
        // carry no extra information.
        pendingSignal ??= sig

        return
      }

      exit(SIGNAL_EXIT_CODE[sig], sig)
    })
  }

  process.on('uncaughtException', err => onError?.('uncaughtException', err))
  process.on('unhandledRejection', reason => onError?.('unhandledRejection', reason))
}
