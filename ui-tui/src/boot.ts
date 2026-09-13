// The guarded startup: from "we have a socket target" to "the app is on
// screen". Everything that can fail in between exits 3, the code
// `opendde_harness/cli/tui_commands.py` documents for a boot failure.
//
// Construction is inside the guard, not before it: `net.createConnection`
// throws synchronously on a malformed target (`127.0.0.1:99999`), and outside
// the guard that left through Node's uncaught-exception path as exit 1.

import type { Terminal, TUI } from '@earendil-works/pi-tui'
import type { RgbColor } from '@earendil-works/pi-tui'

import { ProcessTerminal } from '@earendil-works/pi-tui'

import type { GatewayNotification } from './gateway.js'
import type { Theme } from './theme.js'

import { HarnessApp } from './app.js'
import { Gateway } from './gateway.js'
import { writeClipboardText, writeOsc52Clipboard } from './lib/clipboard.js'
import { historyFilePath, InputHistory } from './lib/history.js'
import { initialRendererMode, RendererOwner } from './renderer.js'
import { RpcClient } from './rpc/index.js'
import { createTheme, schemeFromBackground } from './theme.js'

/**
 * How long to wait for the terminal to say what color it is.
 *
 * The reply is one round trip over the pty, which is sub-millisecond locally
 * and a network hop over ssh. This is the ceiling, not the cost: a terminal
 * that answers is not waited on a moment longer, and one that does not answer
 * at all — the common case, and every terminal under the test harness — delays
 * the first paint by this much and then paints the guess. Long enough to be
 * worth asking, short enough that nobody sees it.
 */
const SCHEME_QUERY_TIMEOUT_MS = 60

/**
 * How much longer the answer is still worth having, once the screen is up.
 *
 * Bounding the first paint is not the same as giving up on the question. A
 * terminal at the far end of an ssh hop can be slower than the paint budget,
 * and abandoning detection there leaves it on the guess for the rest of the
 * session — the same permanent wrong palette that waiting for the answer was
 * meant to prevent. Everything that holds painted lines keys them on the
 * palette, so an answer that lands inside this window repaints rather than
 * flashes.
 */
const SCHEME_LATE_TIMEOUT_MS = 2000

/** Filled in as boot constructs each piece, so a cleanup wired *before* the
 *  boot can close whatever exists. On the malformed-target path that is
 *  nothing at all: the client throws from its own constructor. */
export interface TuiSession {
  client?: RpcClient
  /** Leave the alternate screen and release the terminal. Filled in the moment
   *  a renderer exists, which is well before the boot that may fail after it
   *  has already taken the screen. */
  stopRenderer?: () => void
}

export interface BootOptions {
  /** The environment the prompt history is located from. Defaults to this
   *  process's; the tests point it at a temporary home. */
  env?: NodeJS.ProcessEnv
  /** How the process ends: a boot failure calls it with 3, and the app calls
   *  it when the user quits. It owns the teardown, so it has to tolerate a
   *  session whose client was never constructed. */
  exit: (code: number) => void
  socketPath: string
  stderr?: (text: string) => void
  /** The screen to drive. Defaults to this process's; the tests pass one that
   *  records bytes instead of owning a tty. */
  terminal?: Terminal
}

/** What entry needs after a successful boot to keep driving the screen. */
export interface BootedTui {
  /** The wired app. Entry does not need it; a test that wants to submit a
   *  prompt through the real composition does. */
  app: HarnessApp
  /** Owns the renderer on screen and the switch between the two. */
  renderer: RendererOwner
  theme: Theme
  /** The stable reference, not a concrete renderer: `/fullscreen` replaces the
   *  renderer under it and everything holding this keeps working. */
  tui: TUI
}

export async function bootTui(session: TuiSession, opts: BootOptions): Promise<BootedTui | undefined> {
  const write = opts.stderr ?? ((text: string) => void process.stderr.write(text))

  try {
    // The client is constructed before the app that consumes its
    // notifications, so both sinks are filled in once the app exists.
    const notifications: { sink?: (method: string, params: unknown) => void } = {}
    const lifecycle: { close?: () => void } = {}
    const client = new RpcClient({
      socketPath: opts.socketPath,
      onClose: () => lifecycle.close?.(),
      onNotification: (method, params) => notifications.sink?.(method, params)
    })

    session.client = client

    const terminal = opts.terminal ?? new ProcessTerminal()
    const env = opts.env ?? process.env
    const theme = createTheme(env)
    const renderer = new RendererOwner({
      // A terminal that took the OSC 52 sequence has not told us the desktop
      // clipboard changed, so the native helper answers first and OSC 52 is
      // reported as sent rather than as done.
      copySelection: async text => {
        if (await writeClipboardText(text)) {
          return true
        }

        writeOsc52Clipboard(text)

        return true
      },
      env,
      mode: initialRendererMode(env),
      terminal,
      theme
    })
    // Registered here, not after the boot resolves: `tui.start()` below takes
    // the terminal — the alternate screen, mouse reporting, autowrap — and
    // everything after it can still fail. A cleanup installed only on success
    // is a cleanup that never runs on the one path that needs it.
    session.stopRenderer = () => renderer.dispose()

    const tui = renderer.reference
    // Prompt history is composed here, not in the app: this is the only place
    // that knows there is a filesystem to persist it to. Without it a launched
    // UI kept the current process's prompts and forgot them on exit.
    const history = new InputHistory(historyFilePath(env))
    const gateway = new Gateway(client)

    await client.ready()
    tui.start()
    // Settled before the app exists, not merely before it boots: building the
    // app mounts the welcome lockup, and mounting it composes it — against
    // whatever palette is in force at that moment. Asking the terminal first
    // means the composition and the first paint both happen in the palette the
    // terminal actually has, so a light terminal never shows the dark lockup.
    //
    // This is the only reason the app is constructed here rather than beside
    // the client: it costs one terminal round trip, and nothing is on screen
    // to see it.
    await settleScheme(theme, tui, {
      // An answer that misses the first paint still arrives to a screen that
      // is up, so it repaints rather than being dropped. Every component that
      // holds painted lines keys them on the palette, which is what makes this
      // one clean repaint instead of a half-recolored screen.
      onLateChange: () => {
        tui.invalidate()
        tui.requestRender(true)
      }
    })

    const app = new HarnessApp({ env, gateway, history, onExit: opts.exit, renderer, theme, tui })

    // Filled in the moment the app exists. The gateway pushes nothing before
    // `system.hello`, which `app.boot()` below is what sends, so the window
    // between the connection being ready and these being wired carries no
    // traffic.
    notifications.sink = (method, params) =>
      app.handleNotification(method as GatewayNotification['method'], params as GatewayNotification['params'])
    lifecycle.close = () => app.onDisconnect()

    await app.boot()

    return { app, renderer, theme, tui }
  } catch (err) {
    write(`opendde-tui: ${err instanceof Error ? err.message : String(err)}\n`)
    // Unwound here as well as by `exit`, so a caller whose exit does not tear
    // the session down still gets the screen back. Disposal is idempotent.
    session.stopRenderer?.()
    opts.exit(3)

    return undefined
  }
}

/** What `settleScheme` needs of a TUI. Narrower than the real one so a test can
 *  answer the question without owning a terminal. */
export interface BackgroundQuery {
  queryTerminalBackgroundColor(options: { timeoutMs: number }): Promise<RgbColor | undefined>
}

export interface SettleSchemeOptions {
  /** How long an answer is still applied after the first paint. Zero asks
   *  once and accepts the guess, which is what a test that wants only the
   *  bounded behavior passes. */
  lateTimeoutMs?: number
  /** Called when an answer arriving after the first paint moved the palette.
   *  The screen is already up by then, so the caller has to repaint it. */
  onLateChange?: () => void
  /** How long the first paint waits. */
  timeoutMs?: number
}

/**
 * Settle the palette before the first paint.
 *
 * A theme the environment pinned is not up for discussion, and asking would
 * cost a round trip to ignore the answer. Anything else asks the terminal what
 * its background actually is, because the alternative is guessing from
 * `COLORFGBG` and a default, and on a light terminal the guess is wrong.
 *
 * The answer is applied, not pinned: `tui.theme` still outranks a measurement.
 * Returns whether the palette moved.
 */
export async function settleScheme(
  theme: Theme,
  tui: BackgroundQuery,
  options: SettleSchemeOptions = {}
): Promise<boolean> {
  const { lateTimeoutMs = SCHEME_LATE_TIMEOUT_MS, onLateChange, timeoutMs = SCHEME_QUERY_TIMEOUT_MS } = options

  if (theme.pinned) {
    return false
  }

  const answer = askUntilAnswered(tui, timeoutMs + lateTimeoutMs)
  const early = await Promise.race<RgbColor | typeof STILL_WAITING | undefined>([
    answer,
    sleep(timeoutMs).then(() => STILL_WAITING)
  ])

  if (early !== STILL_WAITING) {
    return early ? theme.setScheme(schemeFromBackground(early)) : false
  }

  // Not in time for the first paint. The question stays open, and an answer
  // that arrives inside the late window still corrects the palette; the caller
  // repaints what is already on screen.
  void answer.then(background => {
    if (background && theme.setScheme(schemeFromBackground(background))) {
      onLateChange?.()
    }
  })

  return false
}

/** The race lost, not the question: the first paint goes ahead without it. */
const STILL_WAITING: unique symbol = Symbol('still waiting')

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

/** How many times the question is put, at most.
 *
 *  The deadline alone does not bound the asking: a query consumed the instant
 *  it is made costs no wall-clock time, so a terminal replying with garbage as
 *  fast as it is asked would spin here. Three is enough for the case this
 *  exists for, one garbled reply followed by a real one, and small enough that
 *  a flood costs three questions rather than thousands.
 */
const MAX_SCHEME_ASKS = 3

/**
 * Ask until the terminal gives a color or the deadline passes.
 *
 * One ask is not enough. pi settles the pending query on any reply shaped like
 * an answer, including one whose color it cannot parse, so a garbled reply
 * consumes the question without answering it. That is indistinguishable from
 * silence unless we ask again — and re-asking is safe only here, because a
 * query that was consumed leaves nothing pending, while one that timed out
 * would. The deadline bounds the loop; each further pass costs one reply
 * arriving and a seven-byte question going back.
 */
async function askUntilAnswered(tui: BackgroundQuery, budgetMs: number): Promise<RgbColor | undefined> {
  // The first ask gets the whole budget as given; only a re-ask reads the
  // clock. Read twice, the clock could tick between the caller's reading and
  // this one, and the terminal was asked for one millisecond less than the
  // budget said.
  const deadline = Date.now() + budgetMs

  for (let ask = 0, left = budgetMs; ask < MAX_SCHEME_ASKS && left > 0; ask += 1, left = deadline - Date.now()) {
    const background = await tui.queryTerminalBackgroundColor({ timeoutMs: left }).catch(() => undefined)

    if (background) {
      return background
    }
  }

  return undefined
}
