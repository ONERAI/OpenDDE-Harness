import { chmodSync, mkdirSync, mkdtempSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { resolveFdPath } from '../components/autocomplete.js'
import { osc52Sequence } from '../lib/clipboard.js'
import { resolveEditor, tokenizeCommand } from '../lib/externalEditor.js'
import { findHeadFile, GitBranch, parseHead } from '../lib/gitBranch.js'
import { rawKeyHint } from '../lib/keybindings.js'
import { setTerminalTitle } from '../lib/terminal.js'
import { Theme } from '../theme.js'

const HELD_SIGNALS = ['SIGHUP', 'SIGINT', 'SIGTERM'] as const

const settle = (ms = 30) => new Promise(resolve => setTimeout(resolve, ms))

/**
 * Wait until `done` holds, or give up after `timeoutMs` and say what was seen.
 *
 * `fs.watch` delivery is the kernel's to schedule: an event is queued, not
 * dropped, but on a loaded host it can arrive well after a fixed sleep would
 * have given up. Waiting for the condition keeps the assertion about the
 * watcher rather than about how busy the machine is, and a timeout still
 * fails — a watcher that genuinely misses the change is not hidden by this.
 */
/** How long a watcher callback may take to arrive.
 *
 *  Generous on purpose: these tests wait on the operating system's file
 *  watcher, and this repository is built and tested by several agents at once.
 *  Measured on a host at load 38, the callback took longer than vitest's
 *  five-second default and the suite failed with a bare "Test timed out" while
 *  the same test passed alone. The budget is a ceiling for a watcher that
 *  never fires, not a claim about how fast one should be; it stays below the
 *  per-test timeout so the message says what was being waited for. */
const WATCH_BUDGET_MS = 15_000

async function until(done: () => boolean, describe: () => string, timeoutMs = WATCH_BUDGET_MS): Promise<void> {
  const deadline = Date.now() + timeoutMs

  while (!done()) {
    if (Date.now() > deadline) {
      throw new Error(`timed out after ${timeoutMs}ms waiting for ${describe()}`)
    }

    await settle(5)
  }
}

describe('resolveEditor', () => {
  let dir: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'harness-bin-'))
  })

  afterEach(() => {
    rmSync(dir, { force: true, recursive: true })
  })

  it('prefers $VISUAL, then $EDITOR, and keeps their arguments', () => {
    expect(resolveEditor({ EDITOR: 'vim', VISUAL: 'code --wait' }, 'linux')).toEqual(['code', '--wait'])
    expect(resolveEditor({ EDITOR: 'vim' }, 'linux')).toEqual(['vim'])
  })

  it('keeps a quoted executable path in one piece', () => {
    expect(resolveEditor({ VISUAL: '"/opt/my editor/bin/edit" --wait' }, 'linux')).toEqual([
      '/opt/my editor/bin/edit',
      '--wait'
    ])
    expect(resolveEditor({ VISUAL: "'/opt/my editor/bin/edit'" }, 'linux')).toEqual(['/opt/my editor/bin/edit'])
    expect(resolveEditor({ VISUAL: '/opt/my\\ editor/edit' }, 'linux')).toEqual(['/opt/my editor/edit'])
  })

  it('does not let an empty $VISUAL mask $EDITOR', () => {
    expect(resolveEditor({ EDITOR: 'vim', VISUAL: '' }, 'linux')).toEqual(['vim'])
    expect(resolveEditor({ EDITOR: 'vim', VISUAL: '   ' }, 'linux')).toEqual(['vim'])
    expect(resolveEditor({ EDITOR: '', PATH: dir, VISUAL: '' }, 'linux')).toEqual(['vi'])
  })

  it('falls back to the first editor on PATH', () => {
    const nano = join(dir, 'nano')

    writeFileSync(nano, '')
    chmodSync(nano, 0o755)

    expect(resolveEditor({ PATH: dir }, 'linux')).toEqual([nano])
  })

  it('falls back to vi with nothing on PATH, and to notepad on Windows', () => {
    expect(resolveEditor({ PATH: dir }, 'linux')).toEqual(['vi'])
    expect(resolveEditor({}, 'win32')).toEqual(['notepad.exe'])
  })
})

describe('tokenizeCommand', () => {
  it('splits on whitespace and keeps quoted runs together', () => {
    expect(tokenizeCommand('code --wait')).toEqual(['code', '--wait'])
    expect(tokenizeCommand('  code   --wait  ')).toEqual(['code', '--wait'])
    expect(tokenizeCommand('code "--file name.txt"')).toEqual(['code', '--file name.txt'])
    expect(tokenizeCommand('emacs \'--eval=(message "hi")\'')).toEqual(['emacs', '--eval=(message "hi")'])
  })

  it('honours backslash escapes outside quotes and the four inside them', () => {
    expect(tokenizeCommand('my\\ editor')).toEqual(['my editor'])
    expect(tokenizeCommand('vim "a\\"b"')).toEqual(['vim', 'a"b'])
    // A backslash before anything else inside double quotes is literal.
    expect(tokenizeCommand('vim "a\\nb"')).toEqual(['vim', 'a\\nb'])
    // Single quotes take everything literally.
    expect(tokenizeCommand("vim 'a\\nb'")).toEqual(['vim', 'a\\nb'])
  })

  it('keeps an empty argument and recovers from an unterminated quote', () => {
    expect(tokenizeCommand('vim ""')).toEqual(['vim', ''])
    expect(tokenizeCommand('"/opt/my editor/edit')).toEqual(['/opt/my editor/edit'])
    expect(tokenizeCommand('   ')).toEqual([])
    expect(tokenizeCommand('')).toEqual([])
  })
})

describe('resolveFdPath', () => {
  let dir: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'harness-fd-'))
  })

  afterEach(() => {
    rmSync(dir, { force: true, recursive: true })
  })

  it('finds fd or the Debian fdfind', () => {
    const fdfind = join(dir, 'fdfind')

    writeFileSync(fdfind, '')
    chmodSync(fdfind, 0o755)

    expect(resolveFdPath({ PATH: dir }, 'linux')).toBe(fdfind)
  })

  it('answers null when it is not installed', () => {
    expect(resolveFdPath({ PATH: dir }, 'linux')).toBeNull()
    expect(resolveFdPath({}, 'linux')).toBeNull()
  })
})

describe('parseHead', () => {
  it('reads a branch name', () => {
    expect(parseHead('ref: refs/heads/tui/pi-tui\n')).toBe('tui/pi-tui')
  })

  it('calls a raw sha detached', () => {
    expect(parseHead('9b1d4e0c5f2a\n')).toBe('detached')
  })

  it('answers null for an empty file', () => {
    expect(parseHead('  \n')).toBeNull()
  })
})

describe('GitBranch', () => {
  let dir: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'harness-git-'))
  })

  afterEach(() => {
    rmSync(dir, { force: true, recursive: true })
  })

  it('finds HEAD in a plain repository, from a subdirectory', () => {
    mkdirSync(join(dir, '.git'))
    writeFileSync(join(dir, '.git', 'HEAD'), 'ref: refs/heads/main\n')
    mkdirSync(join(dir, 'src', 'deep'), { recursive: true })

    expect(findHeadFile(join(dir, 'src', 'deep'))).toBe(join(dir, '.git', 'HEAD'))

    const branch = new GitBranch(join(dir, 'src', 'deep'))

    expect(branch.get()).toBe('main')
    branch.dispose()
  })

  it('follows a worktree .git file', () => {
    const gitDir = join(dir, 'store', 'worktrees', 'feature')

    mkdirSync(gitDir, { recursive: true })
    writeFileSync(join(gitDir, 'HEAD'), 'ref: refs/heads/feature\n')

    const work = join(dir, 'work')

    mkdirSync(work)
    writeFileSync(join(work, '.git'), `gitdir: ${gitDir}\n`)

    const branch = new GitBranch(work)

    expect(branch.get()).toBe('feature')
    branch.dispose()
  })

  it('answers null outside a repository', () => {
    const branch = new GitBranch(dir)

    expect(branch.get()).toBeNull()
    branch.dispose()
  })

  it(
    'follows HEAD through repeated atomic replacement',
    async () => {
      const git = join(dir, '.git')
      const head = join(git, 'HEAD')

      mkdirSync(git)
      writeFileSync(head, 'ref: refs/heads/a\n')

      const seen: (null | string)[] = []
      // A checkout writes a new HEAD and renames it over the old one. Watching
      // the file watched the inode that rename unlinked, so the second
      // replacement was never noticed.
      const replace = (branch: string) => {
        const staged = join(git, 'HEAD.new')

        writeFileSync(staged, `ref: refs/heads/${branch}\n`)
        renameSync(staged, head)
      }

      const watcher = new GitBranch(dir, () => seen.push(watcher.get()), 5)

      expect(watcher.get()).toBe('a')

      try {
        replace('b')
        // Waited for rather than slept past, and waited for before the second
        // replacement, so the two changes cannot coalesce into one re-read.
        await until(
          () => watcher.get() === 'b',
          () => `the first replacement; branch is ${String(watcher.get())}`
        )

        replace('c')
        await until(
          () => watcher.get() === 'c',
          () => `the second replacement; branch is ${String(watcher.get())}`
        )

        expect(seen).toEqual(['b', 'c'])
      } finally {
        watcher.dispose()
      }
      // Above the wait budget, so a slow watcher fails with the message saying
      // what it was waiting for rather than with vitest's bare timeout.
    },
    WATCH_BUDGET_MS + 5_000
  )

  it('ignores the other files a repository writes beside HEAD', async () => {
    const git = join(dir, '.git')

    mkdirSync(git)
    writeFileSync(join(git, 'HEAD'), 'ref: refs/heads/a\n')

    let changes = 0

    const watcher = new GitBranch(dir, () => changes++, 5)

    try {
      writeFileSync(join(git, 'index'), 'not HEAD')
      // A negative assertion, so this one is a real wait: long enough that a
      // watcher which did re-read would have done so by now.
      await settle(250)

      expect(changes).toBe(0)
      expect(watcher.get()).toBe('a')
    } finally {
      watcher.dispose()
    }
  })
})

describe('deferSignalExit', () => {
  /** A fresh module instance per test: the deferral counters and the wiring
   *  are module state, and `setupGracefulExit` wires the process once. */
  async function wire() {
    vi.resetModules()

    const graceful = await import('../lib/gracefulExit.js')
    const external = await import('../lib/externalEditor.js')
    const exits: number[] = []
    const signals: NodeJS.Signals[] = []

    graceful.setupGracefulExit({
      exit: code => exits.push(code),
      // Long enough never to fire during a test; unref'd, so it holds nothing.
      failsafeMs: 60_000,
      onSignal: signal => signals.push(signal)
    })

    const installed = Object.fromEntries(
      HELD_SIGNALS.map(signal => [signal, process.listeners(signal).at(-1)!])
    ) as Record<(typeof HELD_SIGNALS)[number], (signal: NodeJS.Signals) => void>

    return {
      dispose: () => {
        for (const signal of HELD_SIGNALS) {
          process.off(signal, installed[signal])
        }
      },
      exits,
      external,
      fire: (signal: (typeof HELD_SIGNALS)[number]) => installed[signal](signal),
      graceful,
      signals
    }
  }

  it('exits on a signal when nothing is deferred', async () => {
    const held = await wire()

    try {
      held.fire('SIGINT')
      await settle(1)

      expect(held.exits).toEqual([130])
      expect(held.signals).toEqual(['SIGINT'])
    } finally {
      held.dispose()
    }
  })

  it('forgets a Ctrl+C the child owned rather than replaying it', async () => {
    const held = await wire()

    try {
      const release = held.graceful.deferSignalExit({ childOwnsSigint: true })

      expect(held.graceful.classifySignal('SIGINT')).toBe('drop')

      held.fire('SIGINT')
      release()
      await settle(1)

      // The editor got its own copy of the interrupt and handled it. Replaying
      // it here would end the session the user was only stepping out of.
      expect(held.exits).toEqual([])

      held.fire('SIGINT')
      await settle(1)

      expect(held.exits).toEqual([130])
    } finally {
      held.dispose()
    }
  })

  it('still honours a shutdown that arrived during the handoff', async () => {
    const held = await wire()

    try {
      const release = held.graceful.deferSignalExit({ childOwnsSigint: true })

      expect(held.graceful.classifySignal('SIGTERM')).toBe('hold')

      held.fire('SIGTERM')

      expect(held.exits).toEqual([])

      release()
      await settle(1)

      expect(held.exits).toEqual([143])
      expect(held.signals).toEqual(['SIGTERM'])
    } finally {
      held.dispose()
    }
  })

  it('replays a plain deferral, which is not the editor handoff', async () => {
    const held = await wire()

    try {
      const release = held.graceful.deferSignalExit()

      expect(held.graceful.classifySignal('SIGINT')).toBe('hold')

      held.fire('SIGINT')
      release()
      await settle(1)

      expect(held.exits).toEqual([130])
    } finally {
      held.dispose()
    }
  })

  it('the external editor handoff is the kind the child owns', async () => {
    const held = await wire()

    try {
      const editing = held.external.editExternally('draft', ['sh', '-c', 'sleep 0.2'])

      await settle(60)

      expect(held.graceful.classifySignal('SIGINT')).toBe('drop')
      expect(held.graceful.classifySignal('SIGTERM')).toBe('hold')

      await editing

      expect(held.graceful.classifySignal('SIGINT')).toBe('exit')
    } finally {
      held.dispose()
    }
  })
})

describe('key hints', () => {
  it('draws both halves in grey, with the keys the dimmer of the two', () => {
    // A hint is chrome. When `label` became periwinkle these picked up a hue
    // on every prompt and picker in the app, which is what this pins down.
    const theme = new Theme('dark', 3)
    const hint = rawKeyHint(theme, 'esc', 'cancel')

    expect(hint).toBe(`${theme.fg('dim', 'esc')} ${theme.fg('muted', 'cancel')}`)
    expect(hint).not.toContain(theme.fg('label', 'X').split('X')[0]!)
    expect(hint).not.toContain(theme.fg('accent', 'X').split('X')[0]!)
  })
})

describe('setTerminalTitle', () => {
  it('passes the title to the terminal', () => {
    const titles: string[] = []

    setTerminalTitle({ setTitle: (title: string) => titles.push(title) }, 'Antibody design')
    expect(titles).toEqual(['Antibody design'])
  })

  it('swallows a terminal that refuses', () => {
    expect(() =>
      setTerminalTitle(
        {
          setTitle: () => {
            throw new Error('no title here')
          }
        },
        'x'
      )
    ).not.toThrow()
  })
})

describe('osc52Sequence', () => {
  it('base64-encodes the text', () => {
    expect(osc52Sequence('hi', {})).toBe('\x1b]52;c;aGk=\x07')
  })

  it('wraps itself for tmux and screen', () => {
    expect(osc52Sequence('hi', { TMUX: '/tmp/tmux' }).startsWith('\x1bPtmux;')).toBe(true)
    expect(osc52Sequence('hi', { STY: '1.pts-0' }).startsWith('\x1bP\x1b]52')).toBe(true)
  })
})
