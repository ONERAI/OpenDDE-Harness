// Hand the prompt to $VISUAL / $EDITOR.
//
// Two things have to happen around the child: the TUI must stop writing to the
// terminal (`preserveScreen`, so the renderer's output stays put for the editor
// to draw over and for us to resume onto), and this process must stop treating
// Ctrl+C as its own. The child shares our process group, so the interrupt the
// user aims at vim arrives here too, and exiting on it would take the session
// down with the thing they were interrupting.

import { spawn } from 'node:child_process'
import { accessSync, constants, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'

import { deferSignalExit } from './gracefulExit.js'

/**
 * Editor fallback chain when neither $VISUAL nor $EDITOR is set. Mirrors
 * prompt_toolkit's `Buffer.open_in_editor()` picker so the classic CLI and the
 * TUI launch the same editor on a given box.
 */
const FALLBACKS = ['editor', 'nano', 'pico', 'vi', 'emacs']

function isExecutable(path: string): boolean {
  try {
    accessSync(path, constants.X_OK)

    return true
  } catch {
    return false
  }
}

/**
 * A command line split into argv the way a POSIX shell would, minus everything
 * a shell does beyond splitting: no globbing, no variable expansion, no
 * operators. Quotes group, and are removed.
 *
 * `EDITOR="code --wait"` has to become two arguments, and
 * `VISUAL='/opt/my editor/bin/edit'` has to stay one — splitting on
 * whitespace turns the second into three arguments that still carry their
 * quote characters, and the editor never launches.
 */
export function tokenizeCommand(command: string): string[] {
  const tokens: string[] = []

  let token = ''
  let open = false
  let quote: '"' | "'" | null = null

  for (let i = 0; i < command.length; i += 1) {
    const char = command[i]!

    if (quote) {
      if (char === quote) {
        quote = null
      } else if (quote === '"' && char === '\\' && i + 1 < command.length && '"$\\`'.includes(command[i + 1]!)) {
        // Inside double quotes a backslash escapes only these four.
        i += 1
        token += command[i]!
      } else {
        token += char
      }

      continue
    }

    if (char === '\\' && i + 1 < command.length) {
      i += 1
      token += command[i]!
      open = true
    } else if (char === '"' || char === "'") {
      quote = char
      open = true
    } else if (/\s/.test(char)) {
      if (open) {
        tokens.push(token)
        token = ''
        open = false
      }
    } else {
      token += char
      open = true
    }
  }

  // An unterminated quote is a typo in the variable, not a reason to lose the
  // path it was trying to spell.
  if (open) {
    tokens.push(token)
  }

  return tokens
}

/**
 * The editor invocation argv, without the file argument.
 *
 *   1. the first of $VISUAL / $EDITOR that holds something, shell-tokenized
 *      so `EDITOR="code --wait"` works and a quoted path with a space survives
 *   2. on POSIX: the first FALLBACKS entry resolvable on $PATH
 *   3. on Windows: `notepad.exe`
 *   4. literal `['vi']` as the last-resort POSIX floor
 */
export function resolveEditor(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform
): string[] {
  // An empty VISUAL is not a choice of editor, so it does not mask EDITOR.
  for (const value of [env.VISUAL, env.EDITOR]) {
    const argv = tokenizeCommand(value?.trim() ?? '')

    if (argv.length > 0) {
      return argv
    }
  }

  if (platform === 'win32') {
    return ['notepad.exe']
  }

  const dirs = (env.PATH ?? '').split(delimiter).filter(Boolean)
  const found = FALLBACKS.flatMap(name => dirs.map(dir => join(dir, name))).find(isExecutable)

  return [found ?? 'vi']
}

export type ExternalEditResult = { content: string; status: 'complete' } | { status: 'failed' }

/** Run the editor on `content` and return what was saved. */
export async function editExternally(content: string, argv: string[] = resolveEditor()): Promise<ExternalEditResult> {
  const dir = mkdtempSync(join(tmpdir(), 'opendde-prompt-'))
  const file = join(dir, 'prompt.md')
  // The editor is the foreground process for as long as it runs, so Ctrl+C in
  // it is aimed at it and not at this session. Held rather than remembered:
  // replaying it when the editor returns would quit the session the user was
  // only stepping out of.
  const release = deferSignalExit({ childOwnsSigint: true })

  try {
    writeFileSync(file, content, 'utf8')

    const [command, ...args] = argv

    if (!command) {
      return { status: 'failed' }
    }

    const code = await new Promise<null | number>(resolve => {
      // Not spawnSync: a synchronous child keeps libuv's stdin read active
      // after the parent pauses stdin, and it races vim for the input buffer.
      const child = spawn(command, [...args, file], {
        stdio: 'inherit',
        shell: process.platform === 'win32'
      })

      child.on('error', () => resolve(null))
      child.on('close', exit => resolve(exit))
    })

    if (code !== 0) {
      return { status: 'failed' }
    }

    return {
      content: readFileSync(file, 'utf8')
        .replace(/^\uFEFF/, '')
        .replace(/\n$/, ''),
      status: 'complete'
    }
  } catch {
    return { status: 'failed' }
  } finally {
    release()

    try {
      rmSync(dir, { force: true, recursive: true })
    } catch {
      // Cleanup is best effort.
    }
  }
}
