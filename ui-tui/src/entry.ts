// Process entry for the pi-tui TUI. The env contract and exit codes are the
// ones `opendde_harness/cli/tui_commands.py` already speaks:
//
//   no TTY                          → print a note, exit 0
//   OPENDDE_HARNESS_TUI_CHECK=1     → exit 0 (import smoke; needs no socket)
//   no OPENDDE_HARNESS_RPC_SOCKET   → print how to launch, exit 2
//   handshake / boot failure        → exit 3 (see boot.ts)
//   SIGHUP / SIGINT / SIGTERM       → 129 / 130 / 143, after cleanups
//
// The terminal is reset at boot as well as on exit: a previous TUI that was
// killed with -9 can leave mouse, focus and paste modes on in the tab.

import type { TuiSession } from './boot.js'

import { bootTui } from './boot.js'
import { setupGracefulExit } from './lib/gracefulExit.js'
import { resetTerminalModes } from './lib/terminalModes.js'

if (!process.stdin.isTTY) {
  console.log('opendde-tui: no TTY')
  process.exit(0)
}

resetTerminalModes()

if (process.env.OPENDDE_HARNESS_TUI_CHECK === '1') {
  process.exit(0)
}

const socketPath = process.env.OPENDDE_HARNESS_RPC_SOCKET

if (!socketPath) {
  process.stderr.write('opendde-tui: OPENDDE_HARNESS_RPC_SOCKET env var required; spawn via `ddeharness tui` parent\n')
  process.exit(2)
}

// Boot fills this in as it constructs; the teardown below runs whether or not
// it got that far.
const session: TuiSession = {}

let exiting = false

const shutdown = (code: number) => {
  if (exiting) {
    return
  }

  exiting = true
  // Stop the renderer before the terminal is reset: the alternate screen has
  // to be left, and its mouse and focus modes turned off, by the thing that
  // turned them on.
  session.stopRenderer?.()
  resetTerminalModes()
  session.client?.close()
  process.exit(code)
}

setupGracefulExit({
  cleanups: [
    () => {
      session.stopRenderer?.()
      resetTerminalModes()
      session.client?.close()
    }
  ],
  onError: (scope, err) => {
    const message = err instanceof Error ? `${err.name}: ${err.message}` : String(err)

    process.stderr.write(`opendde-tui ${scope}: ${message.slice(0, 2000)}\n`)
  },
  onSignal: signal => {
    resetTerminalModes()
    process.stderr.write(`opendde-tui: received ${signal}\n`)
  }
})

// Boot fills in `session.stopRenderer` as soon as it has a renderer, so the
// cleanups above cover a boot that fails after the screen was taken as well as
// a signal that arrives while it is still in flight. It also settles the
// palette before it draws anything, so nothing here has to correct it after.
await bootTui(session, { exit: shutdown, socketPath })
