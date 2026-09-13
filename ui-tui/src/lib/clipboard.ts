// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See LICENSES/README.md and LICENSES/MIT-hermes-agent.txt.
//
// Writing to the system clipboard. Native helpers first, OSC 52 as the fallback
// that also works over ssh.

import type { spawn as SpawnFn } from 'node:child_process'

import { spawn } from 'node:child_process'

interface ClipboardCommand {
  args: readonly string[]
  cmd: string
}

const SET_CLIPBOARD = ['-NoProfile', '-NonInteractive', '-Command', 'Set-Clipboard -Value $input'] as const

function writers(platform: NodeJS.Platform, env: NodeJS.ProcessEnv): ClipboardCommand[] {
  if (platform === 'darwin') {
    return [{ args: [], cmd: 'pbcopy' }]
  }

  if (platform === 'win32') {
    return [{ args: SET_CLIPBOARD, cmd: 'powershell' }]
  }

  const attempts: ClipboardCommand[] = []

  if (env.WSL_INTEROP || env.WSL_DISTRO_NAME) {
    attempts.push({ args: SET_CLIPBOARD, cmd: 'powershell.exe' })
  }

  if (env.WAYLAND_DISPLAY) {
    attempts.push({ args: ['--type', 'text/plain'], cmd: 'wl-copy' })
  }

  attempts.push({ args: ['-selection', 'clipboard', '-in'], cmd: 'xclip' })
  attempts.push({ args: ['--clipboard', '--input'], cmd: 'xsel' })

  return attempts
}

/** True when a native clipboard tool took the text. Callers fall back to OSC 52. */
export async function writeClipboardText(
  text: string,
  platform: NodeJS.Platform = process.platform,
  start: typeof SpawnFn = spawn,
  env: NodeJS.ProcessEnv = process.env
): Promise<boolean> {
  for (const { args, cmd } of writers(platform, env)) {
    try {
      const ok = await new Promise<boolean>(resolve => {
        const child = start(cmd, [...args], { stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true })

        child.once('error', () => resolve(false))
        child.once('close', code => resolve(code === 0))
        child.stdin?.end(text)
      })

      if (ok) {
        return true
      }
    } catch {
      // Try the next backend.
    }
  }

  return false
}

function readers(platform: NodeJS.Platform, env: NodeJS.ProcessEnv): ClipboardCommand[] {
  if (platform === 'darwin') {
    return [{ args: [], cmd: 'pbpaste' }]
  }

  if (platform === 'win32') {
    return [{ args: ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard'], cmd: 'powershell' }]
  }

  const attempts: ClipboardCommand[] = []

  if (env.WSL_INTEROP || env.WSL_DISTRO_NAME) {
    attempts.push({ args: ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard'], cmd: 'powershell.exe' })
  }

  if (env.WAYLAND_DISPLAY) {
    attempts.push({ args: ['--no-newline'], cmd: 'wl-paste' })
  }

  attempts.push({ args: ['-selection', 'clipboard', '-out'], cmd: 'xclip' })
  attempts.push({ args: ['--clipboard', '--output'], cmd: 'xsel' })

  return attempts
}

/**
 * How long one clipboard helper gets to answer.
 *
 * `xclip` in particular is famous for hanging: it can start, own the X
 * selection and never exit, in which case it emits neither `error` nor
 * `close`. Paste is a keystroke, so a helper that has not answered by now has
 * already failed as far as the user is concerned.
 */
export const NATIVE_CLIPBOARD_TIMEOUT_MS = 600

/**
 * What a native clipboard tool holds, or null when none of them answered.
 *
 * Null is the normal answer over ssh with no forwarded display: the clipboard
 * the user copied into belongs to their terminal, not to this host. That is
 * what the OSC 52 read is for, and it is the reason every attempt here is on a
 * deadline — a helper that hangs would otherwise take the fallback down with
 * it and paste would simply do nothing, forever.
 */
export async function readClipboardText(
  platform: NodeJS.Platform = process.platform,
  start: typeof SpawnFn = spawn,
  env: NodeJS.ProcessEnv = process.env,
  timeoutMs: number = NATIVE_CLIPBOARD_TIMEOUT_MS
): Promise<null | string> {
  for (const { args, cmd } of readers(platform, env)) {
    try {
      const text = await new Promise<null | string>(resolve => {
        const child = start(cmd, [...args], { stdio: ['ignore', 'pipe', 'ignore'], windowsHide: true })
        let out = ''
        let done = false

        const settle = (value: null | string) => {
          if (done) {
            return
          }

          done = true
          clearTimeout(timer)
          resolve(value)
        }

        const timer = setTimeout(() => {
          // Take the listeners off before killing it: the `close` this
          // provokes belongs to a read nobody is waiting for any more.
          child.stdout?.removeAllListeners('data')
          child.removeAllListeners('error')
          child.removeAllListeners('close')
          child.once('error', () => {})

          try {
            child.kill('SIGKILL')
          } catch {
            // It is already gone, which is the outcome we wanted.
          }

          settle(null)
        }, timeoutMs)

        timer.unref?.()

        child.stdout?.setEncoding('utf8')
        child.stdout?.on('data', (chunk: string) => {
          out += chunk
        })
        child.once('error', () => settle(null))
        child.once('close', code => settle(code === 0 ? out : null))
      })

      if (text !== null && text !== '') {
        return text
      }
    } catch {
      // Try the next backend.
    }
  }

  return null
}

/** The OSC 52 copy sequence, wrapped for tmux/screen when we are inside one. */
export function osc52Sequence(text: string, env: NodeJS.ProcessEnv = process.env): string {
  const sequence = `\x1b]52;c;${Buffer.from(text, 'utf8').toString('base64')}\x07`

  if (env.TMUX) {
    return `\x1bPtmux;${sequence.split('\x1b').join('\x1b\x1b')}\x1b\\`
  }

  if (env.STY) {
    return `\x1bP${sequence}\x1b\\`
  }

  return sequence
}

/** Ask the terminal itself to hold the text. Works over ssh; not every terminal obeys. */
export function writeOsc52Clipboard(text: string, out: { write(data: string): unknown } = process.stdout): void {
  out.write(osc52Sequence(text))
}
