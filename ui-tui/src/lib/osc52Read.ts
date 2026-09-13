// Asking the terminal for its clipboard.
//
// Over ssh there is no local clipboard tool to run: `xclip` on the remote host
// reads the remote X server, which is not where the text the user copied is.
// OSC 52 asks the terminal emulator itself, which is. Most terminals need the
// read half enabled explicitly and many never answer at all, so this is always
// a best effort behind a timeout — a terminal that stays quiet is the common
// case, not an error.

import type { TUI } from '@earendil-works/pi-tui'

const ESC = '\x1b'
const BEL = '\x07'
const ST = `${ESC}\\`

/** How long a silent terminal is waited for. */
export const OSC52_READ_TIMEOUT_MS = 400

/** Wrap a sequence so tmux and screen pass it through to the real terminal. */
export function wrapForMultiplexer(sequence: string, env: NodeJS.ProcessEnv = process.env): string {
  if (env.TMUX) {
    return `${ESC}Ptmux;${sequence.split(ESC).join(ESC + ESC)}${ST}`
  }

  if (env.STY) {
    return `${ESC}P${sequence}${ST}`
  }

  return sequence
}

export function osc52ReadQuery(env: NodeJS.ProcessEnv = process.env): string {
  return wrapForMultiplexer(`${ESC}]52;c;?${BEL}`, env)
}

// eslint-disable-next-line no-control-regex
const REPLY = /\x1b\]52;([^;]*);([A-Za-z0-9+/=]*)(?:\x07|\x1b\\)/

/** The clipboard text in an OSC 52 reply, or null when this is not one. */
export function parseOsc52Reply(data: string): null | string {
  const match = REPLY.exec(data)

  if (!match) {
    return null
  }

  const [, selection = '', payload = ''] = match

  if ((selection !== 'c' && selection !== 'p' && selection !== '') || !payload) {
    return null
  }

  try {
    return Buffer.from(payload, 'base64').toString('utf8')
  } catch {
    return null
  }
}

/** Everything except the reply, so the rest of the burst still reaches the UI. */
export function stripOsc52Reply(data: string): string {
  return data.replace(REPLY, '')
}

/**
 * Query the terminal's clipboard.
 *
 * The reply arrives on stdin among ordinary keystrokes, so the listener takes
 * the reply out of the data and passes whatever else was in the same read
 * through. Resolves to null on a timeout.
 */
export async function readOsc52Clipboard(
  tui: TUI,
  env: NodeJS.ProcessEnv = process.env,
  timeoutMs = OSC52_READ_TIMEOUT_MS
): Promise<null | string> {
  return new Promise<null | string>(resolve => {
    // Filled in once the listener is registered. `finish` can run from inside
    // that very call, so it reads the disposer rather than closing over one.
    const listener: { settled: boolean; stop?: () => void } = { settled: false }

    const finish = (text: null | string) => {
      if (listener.settled) {
        return
      }

      listener.settled = true
      clearTimeout(timer)
      listener.stop?.()
      resolve(text)
    }

    const timer = setTimeout(() => finish(null), timeoutMs)

    timer.unref?.()

    listener.stop = tui.addInputListener(data => {
      const text = parseOsc52Reply(data)

      if (text === null) {
        return undefined
      }

      finish(text)

      const rest = stripOsc52Reply(data)

      return rest ? { data: rest } : { consume: true }
    })

    if (listener.settled) {
      listener.stop()
    }

    tui.terminal.write(osc52ReadQuery(env))
  })
}
