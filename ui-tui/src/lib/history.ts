// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See LICENSES/README.md and LICENSES/MIT-hermes-agent.txt.
//
// Persisted prompt history. The on-disk format is the classic CLI's, so both
// front-ends share one file: a `# <timestamp>` line opens an entry and every
// following `+`-prefixed line is one line of it.
//
// Prompts are what the user typed, which is as private as anything this app
// stores, so the file and its directory are created for the owner alone. The
// entry cap applies to the file as well as to the cache: appending is cheap,
// and once the file holds more than the cap it is rewritten — to a sibling
// temp file first, then renamed over, so a reader never sees a half-written
// history. A concurrent writer's entry can be lost in that one rewrite; every
// other append is a plain append and keeps them all.

import {
  appendFileSync,
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  readlinkSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync
} from 'node:fs'
import { homedir, hostname } from 'node:os'
import { dirname, join } from 'node:path'

const MAX_ENTRIES = 1000

/** Owner-only: the file holds everything the user has typed at the prompt. */
const FILE_MODE = 0o600
const DIR_MODE = 0o700

/** How long a writer waits for the lock before going ahead without it. Two
 *  people typing in two terminals contend for microseconds; this is the
 *  ceiling for a pathological case, not an expected wait. */
const LOCK_WAIT_MS = 250
const LOCK_POLL_MS = 2

/** A lock whose owner cannot be identified and which is older than this is
 *  treated as abandoned. A lock that names a living owner is never broken,
 *  however old it is: a rotator paused mid-rename is slow, not dead, and
 *  stealing its lock lost the append that landed while it was away. */
const LOCK_STALE_MS = 10_000

/** Who holds a lock.
 *
 *  The pid alone is not enough. Pids are reused, so the owner's start time is
 *  recorded where the platform exposes it. And a pid is only meaningful on the
 *  machine and PID namespace that issued it: the history file can sit on a
 *  network share, where a pid from another host means nothing at all here.
 *  `host`, `ns` and `boot` say whose numbering this is. */
interface LockOwner {
  boot: string
  host: string
  id: string
  ns: string
  pid: number
  start: string
}

/** Whose PID numbering this process's own pid belongs to.
 *
 *  Exported because a test that forges a lock has to forge one this machine
 *  would accept, and a second copy of these three reads would drift from this
 *  one. Read once: none of them changes while the process lives. */
export function lockProvenance(): { boot: string; host: string; ns: string } {
  return HERE
}

function readFirstLine(path: string): string {
  try {
    return readFileSync(path, 'utf8').trim()
  } catch {
    return ''
  }
}

const HERE = {
  // A reboot restarts pids and their start times together, so without this a
  // lock left by a process that died in the last boot can look alive.
  boot: readFirstLine('/proc/sys/kernel/random/boot_id'),
  host: hostname(),
  // `pid:[4026531836]`, or '' off Linux and inside a sandbox that hides it.
  ns: (() => {
    try {
      return readlinkSync('/proc/self/ns/pid')
    } catch {
      return ''
    }
  })()
}

/** The start time of `pid` as the kernel records it, or '' where it cannot be
 *  read. Field 22 of `/proc/<pid>/stat`, counted after the executable name,
 *  which is parenthesised and may itself contain spaces and brackets. */
function processStart(pid: number): string {
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, 'utf8')
    const tail = stat.slice(stat.lastIndexOf(')') + 2).split(' ')

    return tail[19] ?? ''
  } catch {
    return ''
  }
}

/** Whether this kernel is the one that issued `owner.pid`.
 *
 *  Every field recorded has to match, and a field this machine can read but
 *  the lock does not carry is a mismatch: an absence of evidence is not
 *  evidence that the numbering is ours. */
function sameKernel(owner: LockOwner): boolean {
  return owner.host === HERE.host && owner.ns === HERE.ns && owner.boot === HERE.boot
}

/**
 * What is known about the process named by a lock.
 *
 * `alive` and `dead` are answers from the kernel that issued the pid. Anything
 * else is `unknown`, and the two cases are different in kind:
 *
 * - A lock from another host or PID namespace. Signal 0 here asks *this*
 *   kernel about a number that belongs to another one, and the ESRCH it
 *   answers with is not evidence about the writer. Nor is the file's mtime,
 *   which was set by a clock this machine does not share. Such a lock is never
 *   broken. The cost is that a writer that died on the other host leaves a
 *   lock nothing here will clear, and appends buffer and retry until somebody
 *   removes it; the alternative is overwriting a live writer's history, which
 *   is the failure this records provenance to prevent.
 * - A lock carrying no provenance at all. Nothing this build writes looks like
 *   that, so it is treated the way an unreadable lock has always been treated:
 *   by age.
 */
function ownerState(owner: LockOwner): 'alive' | 'dead' | 'unknown' {
  if (!owner.host) {
    return 'unknown'
  }

  if (!sameKernel(owner)) {
    return 'unknown'
  }

  try {
    process.kill(owner.pid, 0)
  } catch (err) {
    // EPERM means it exists and belongs to another user, which is still alive.
    return (err as NodeJS.ErrnoException).code === 'EPERM' ? 'alive' : 'dead'
  }

  const start = processStart(owner.pid)

  // Unknown start times are believed, because refusing to break is the safe
  // answer.
  return !owner.start || !start || start === owner.start ? 'alive' : 'dead'
}

/** Somewhere to park. Appending is synchronous and on the path that sends a
 *  prompt, so the wait must not go through the event loop. */
const PARK = new Int32Array(new SharedArrayBuffer(4))

/** Where history lives: `$OPENDDE_HARNESS_HOME/.opendde_harness_history`. */
export function historyFilePath(env: NodeJS.ProcessEnv = process.env): string {
  return join(env.OPENDDE_HARNESS_HOME ?? join(homedir(), '.opendde_harness'), '.opendde_harness_history')
}

function stamp(): string {
  return new Date().toISOString().replace('T', ' ').replace('Z', '')
}

function pause(ms: number): void {
  Atomics.wait(PARK, 0, 0, ms)
}

/** One entry as it is stored: the header line, then the `+`-prefixed body. */
function record(entry: string, header = `# ${stamp()}`): string {
  return `\n${header}\n${entry
    .split('\n')
    .map(line => `+${line}`)
    .join('\n')}\n`
}

/**
 * The prompt history file, read once and appended to as prompts are sent.
 *
 * Every failure is swallowed: a history file that cannot be read or written is
 * a lost convenience, never a reason to refuse a prompt.
 */
export class InputHistory {
  private entries: null | string[] = null
  /** How many entries the file holds, as far as this process knows. */
  private onDisk = 0
  /** Entries written but not yet on disk, because the lock was held when they
   *  were. They go out with the next append that gets it. */
  private pending: string[] = []
  /** The stored form of each cached entry, so a rewrite keeps the timestamps
   *  the entries were written with rather than restamping them all with now. */
  private records: string[] = []

  constructor(
    private readonly file: string = historyFilePath(),
    private readonly max: number = MAX_ENTRIES
  ) {}

  /** Oldest first, capped at `max`. Cached after the first read. */
  load(): string[] {
    if (this.entries) {
      return this.entries
    }

    this.entries = this.read()

    return this.entries
  }

  /** Append one prompt, skipping blanks and an immediate repeat. */
  append(line: string): void {
    const trimmed = line.trim()

    if (!trimmed) {
      return
    }

    const entries = this.load()

    if (entries.at(-1) === trimmed) {
      return
    }

    const stored = record(trimmed)

    entries.push(trimmed)
    this.records.push(stored)

    if (entries.length > this.max) {
      entries.splice(0, entries.length - this.max)
      this.records.splice(0, this.records.length - this.max)
    }

    if (!this.write(stored)) {
      return
    }

    this.onDisk += 1

    if (this.onDisk > this.max) {
      this.compact()
    }
  }

  /**
   * Persist whatever an earlier append could not, and say whether the file now
   * holds it.
   *
   * An append that cannot take the lock keeps its entry buffered rather than
   * writing beside another writer, and the buffer goes out with the next
   * append. A session whose last prompt lost that race would otherwise never
   * get another one, so the way out calls this. Nothing buffered is success:
   * there is nothing to lose.
   */
  flush(): boolean {
    if (this.pending.length === 0) {
      return true
    }

    // An empty entry carries the buffer without adding to it: `write` counts
    // everything it flushed except the one it was handed, which is this
    // nothing. The entries themselves were never counted, because the append
    // that made them returned false.
    const written = this.write('')

    if (written && this.onDisk > this.max) {
      this.compact()
    }

    return written
  }

  /** Everything the file holds right now, oldest first, in both forms.
   *  Returns null when there is nothing to read or it cannot be read. */
  private parse(): null | { entries: string[]; records: string[] } {
    try {
      if (!existsSync(this.file)) {
        return { entries: [], records: [] }
      }

      const entries: string[] = []
      const records: string[] = []

      let body: string[] = []
      let header: null | string = null
      let pending: null | string = null

      const close = (): void => {
        if (!body.length) {
          return
        }

        entries.push(body.join('\n'))
        records.push(record(body.join('\n'), header ?? `# ${stamp()}`))
        body = []
        header = null
      }

      for (const line of readFileSync(this.file, 'utf8').split('\n')) {
        if (line.startsWith('+')) {
          if (!body.length) {
            header = pending
          }

          body.push(line.slice(1))
          continue
        }

        close()
        pending = line.startsWith('#') ? line : null
      }

      close()

      return { entries, records }
    } catch {
      return null
    }
  }

  private read(): string[] {
    const stored = this.parse()

    if (!stored) {
      return []
    }

    this.onDisk = stored.entries.length
    this.records = stored.records.slice(-this.max)

    return stored.entries.slice(-this.max)
  }

  /**
   * Append this entry and anything held back, under the lock.
   *
   * Returns whether the file has them. A write that could not take the lock
   * keeps its entries in memory instead of going ahead without it: the holder
   * is most likely a rotation, which read the file before this append and
   * will rename its own copy over the top afterwards — so the entry would be
   * acknowledged, visible, and then gone. Held entries go out with the next
   * append that does get the lock, in the order they were typed, and the
   * editor's own history has them meanwhile.
   */
  private write(stored: string): boolean {
    this.pending.push(stored)

    try {
      const dir = dirname(this.file)

      if (!existsSync(dir)) {
        mkdirSync(dir, { mode: DIR_MODE, recursive: true })
      }

      const flush = (): void => {
        // `mode` applies only where it creates the file, which is the case
        // this is about: an existing file keeps whatever the user gave it.
        appendFileSync(this.file, this.pending.join(''), { mode: FILE_MODE })
      }

      if (!this.locked(flush)) {
        return false
      }

      this.onDisk += this.pending.length - 1
      this.pending = []

      return true
    } catch {
      // A history file we cannot write is not worth interrupting the user for.
      this.pending = []

      return false
    }
  }

  /**
   * Run `write` holding the history file's lock. Returns whether it was held.
   *
   * An advisory lock file, created with `O_EXCL` so exactly one process wins,
   * and broken when it is old enough to have been left by a process that died.
   * Appends and rotations both take it, which is the point: serialising only
   * rotations would still let an append land in the window one had open.
   */
  private locked(write: () => void): boolean {
    const path = `${this.file}.lock`
    const deadline = Date.now() + LOCK_WAIT_MS
    const mine: LockOwner = {
      ...HERE,
      id: `${process.pid}.${Date.now()}.${Math.random().toString(36).slice(2)}`,
      pid: process.pid,
      start: processStart(process.pid)
    }

    for (;;) {
      let held: number

      try {
        held = openSync(path, 'wx', FILE_MODE)
      } catch {
        if (!this.breakStaleLock(path) && Date.now() >= deadline) {
          return false
        }

        pause(LOCK_POLL_MS)
        continue
      }

      try {
        // Who to ask about before ever breaking this lock. Written inside the
        // lock this process just won, so the name in the file is always the
        // name of whoever is holding it.
        writeFileSync(held, JSON.stringify(mine))
      } catch {
        // A lock nobody can read falls back to the age rule, which is what
        // every lock did before it carried a name.
      }

      try {
        write()
      } finally {
        closeSync(held)

        // Only ours to remove. A lock broken while this process held it
        // belongs to whoever made the new one, and unlinking it would hand a
        // third writer the same file.
        if (this.lockOwner(path)?.id === mine.id) {
          try {
            unlinkSync(path)
          } catch {
            // Already gone is gone enough.
          }
        }
      }

      return true
    }
  }

  /** Who holds the lock at `path`, or null when nothing legible does. */
  private lockOwner(path: string): LockOwner | null {
    try {
      const owner = JSON.parse(readFileSync(path, 'utf8')) as Partial<LockOwner>
      const text = (value: unknown): string => (typeof value === 'string' ? value : '')

      // Provenance is kept, not dropped: discarding it was what let a pid from
      // another host be read as one of this kernel's own.
      return typeof owner?.pid === 'number' && typeof owner.id === 'string'
        ? {
            boot: text(owner.boot),
            host: text(owner.host),
            id: owner.id,
            ns: text(owner.ns),
            pid: owner.pid,
            start: text(owner.start)
          }
        : null
    } catch {
      return null
    }
  }

  /**
   * Remove a lock file nothing living is holding.
   *
   * A lock that names a process still running is never broken, however long it
   * has been held: a rotator paused between its read and its rename is slow,
   * not dead, and taking its lock meant the append that landed meanwhile was
   * overwritten by the rename when it resumed. Age is only evidence about a
   * lock whose owner cannot be identified at all — one an older build wrote,
   * or one whose name never reached the disk.
   */
  private breakStaleLock(path: string): boolean {
    const owner = this.lockOwner(path)
    const state = owner ? ownerState(owner) : 'unknown'

    if (state === 'alive') {
      return false
    }

    // A lock whose owner is identified but unreachable — another host, another
    // PID namespace — is kept whatever its age says, because its mtime came
    // from a clock this machine does not share.
    if (state === 'unknown' && owner?.host) {
      return false
    }

    if (state === 'unknown') {
      try {
        if (Date.now() - statSync(path).mtimeMs < LOCK_STALE_MS) {
          return false
        }
      } catch {
        return false
      }
    }

    // Claim the break with a rename, which only one breaker can win, and only
    // after confirming the file still names the owner just judged dead. A lock
    // created in the moment between is then moved aside rather than unlinked,
    // and its holder's release finds an id that is not its own and leaves the
    // next lock alone.
    if (owner && this.lockOwner(path)?.id !== owner.id) {
      return false
    }

    const claimed = `${path}.broken.${process.pid}.${Date.now()}`

    try {
      renameSync(path, claimed)
    } catch {
      return false
    }

    try {
      unlinkSync(claimed)
    } catch {
      // Gone is gone enough; the lock path is free either way.
    }

    return true
  }

  /**
   * Rewrite the file with the newest `max` entries, atomically.
   *
   * Read under the lock, and rewrite what the *file* holds rather than what
   * this instance cached. Two things used to go wrong here and both lost
   * another writer's entries: rewriting from a cache filled when this process
   * started threw away everything appended since, and an append that landed
   * between the read and the rename was in the file the rename replaced. The
   * atomic rename only ever prevented a torn file, not a lost update.
   *
   * A rotation that cannot take the lock does nothing. The file stays over the
   * cap until the next append, which tries again.
   */
  private compact(): void {
    this.locked(() => {
      const stored = this.parse()

      if (!stored) {
        return
      }

      const kept = stored.records.slice(-this.max)
      const temp = `${this.file}.${process.pid}.tmp`

      try {
        writeFileSync(temp, kept.join(''), { mode: FILE_MODE })
        renameSync(temp, this.file)
        this.onDisk = kept.length
      } catch {
        try {
          unlinkSync(temp)
        } catch {
          // Nothing was written, or it is not ours to remove.
        }
      }
    })
  }
}
