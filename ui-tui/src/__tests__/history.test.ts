import { spawn } from 'node:child_process'
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  utimesSync,
  writeFileSync
} from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { historyFilePath, InputHistory, lockProvenance } from '../lib/history.js'

/** The kernel's start time for `pid`, the way the lock records it, so a test
 *  lock is indistinguishable from one this process really took. */
function startOf(pid: number): string {
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, 'utf8')

    return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19] ?? ''
  } catch {
    return ''
  }
}

/** A lock record this machine would accept as its own, so a test lock is
 *  indistinguishable from one this process really took. The provenance comes
 *  from the module under test rather than from a second copy of the same three
 *  reads, which would drift from it. */
function lockRecord(fields: { id: string; pid: number; start?: string }): string {
  return JSON.stringify({ ...lockProvenance(), start: '', ...fields })
}

/** The same, but issued by a machine that is not this one. */
function foreignLockRecord(fields: { id: string; pid: number }): string {
  return JSON.stringify({
    ...lockProvenance(),
    boot: '00000000-0000-4000-8000-000000000000',
    host: 'a-host-that-is-not-this-one',
    ns: 'pid:[4026500000]',
    start: '',
    ...fields
  })
}

/** A pid nothing is running, found rather than assumed. */
function deadPid(): number {
  for (let pid = 4_194_300; pid > 4_100_000; pid -= 7) {
    try {
      process.kill(pid, 0)
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ESRCH') {
        return pid
      }
    }
  }

  throw new Error('every pid in the search range is in use')
}

/** A prompt long enough to tell apart in the file. */
const APPEND_SENTINEL = 'the first thing typed'

describe('historyFilePath', () => {
  it('follows OPENDDE_HARNESS_HOME when it is set', () => {
    expect(historyFilePath({ OPENDDE_HARNESS_HOME: '/srv/state' })).toBe('/srv/state/.opendde_harness_history')
  })
})

describe('InputHistory', () => {
  let dir: string
  let file: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'harness-history-'))
    file = join(dir, 'nested', '.opendde_harness_history')
  })

  afterEach(() => {
    rmSync(dir, { force: true, recursive: true })
  })

  it('starts empty when there is no file', () => {
    expect(new InputHistory(file).load()).toEqual([])
  })

  it('round-trips entries through the file', () => {
    const writer = new InputHistory(file)

    writer.append('first prompt')
    writer.append('second prompt')

    expect(new InputHistory(file).load()).toEqual(['first prompt', 'second prompt'])
  })

  it('keeps the lines of a multi-line prompt together', () => {
    new InputHistory(file).append('line one\nline two')

    expect(new InputHistory(file).load()).toEqual(['line one\nline two'])
  })

  it('skips blanks and an immediate repeat', () => {
    const history = new InputHistory(file)

    history.append('same')
    history.append('   ')
    history.append('same')
    history.append('different')

    expect(new InputHistory(file).load()).toEqual(['same', 'different'])
  })

  it('keeps only the newest entries once the cap is reached', () => {
    const history = new InputHistory(file, 3)

    for (const text of ['a', 'b', 'c', 'd', 'e']) {
      history.append(text)
    }

    expect(history.load()).toEqual(['c', 'd', 'e'])
    expect(new InputHistory(file, 3).load()).toEqual(['c', 'd', 'e'])
  })

  it('reads a file the classic CLI wrote', () => {
    const plain = join(dir, '.opendde_harness_history')

    writeFileSync(plain, '\n# 2026-01-01 00:00:00\n+from the cli\n\n# 2026-01-02 00:00:00\n+two\n+lines\n')

    expect(new InputHistory(plain).load()).toEqual(['from the cli', 'two\nlines'])
  })

  it('survives an unreadable file', () => {
    const broken = join(dir, 'not-a-dir', 'deeper', 'history')

    // The parent does not exist, so both the read and the write fail.
    const history = new InputHistory(broken)

    expect(history.load()).toEqual([])
    expect(() => history.append('still fine')).not.toThrow()
  })

  it('appends rather than rewriting', () => {
    const history = new InputHistory(file)

    history.append('one')
    history.append('two')

    const text = readFileSync(file, 'utf8')

    expect(text.match(/^#/gm)?.length).toBe(2)
    expect(text).toContain('+one')
    expect(text).toContain('+two')
  })

  it('caps the file as well as the cache', () => {
    const history = new InputHistory(file, 3)

    for (const text of ['a', 'b', 'c', 'd', 'e', 'f', 'g']) {
      history.append(text)
    }

    const stored = readFileSync(file, 'utf8')

    // The cap used to apply only to what was kept in memory, so the file grew
    // without limit and every start paid for reading all of it.
    expect(stored.match(/^\+/gm)?.length).toBe(3)
    expect(stored).not.toContain('+a')
    expect(new InputHistory(file, 3).load()).toEqual(['e', 'f', 'g'])
  })

  it('keeps the timestamps the surviving entries were written with', () => {
    const plain = join(dir, '.opendde_harness_history')

    writeFileSync(
      plain,
      '\n# 2020-01-01 00:00:00\n+one\n\n# 2021-01-01 00:00:00\n+two\n\n# 2022-01-01 00:00:00\n+three\n'
    )

    const history = new InputHistory(plain, 3)

    history.append('four')

    const stored = readFileSync(plain, 'utf8')

    expect(stored).toContain('# 2022-01-01 00:00:00')
    expect(stored).not.toContain('# 2020-01-01 00:00:00')
    expect(new InputHistory(plain, 3).load()).toEqual(['two', 'three', 'four'])
  })

  it('leaves no half-written file behind when it rewrites', () => {
    const history = new InputHistory(file, 2)

    for (const text of ['a', 'b', 'c', 'd']) {
      history.append(text)
    }

    expect(readdirSync(dirname(file))).toEqual(['.opendde_harness_history'])
  })

  it('never lets a paused rotator overwrite an entry it acknowledged', async () => {
    const HOLD = join(import.meta.dirname, 'fixtures', 'holdHistoryLock.ts')
    // Straight in the temporary directory: the seeded file and its lock have
    // to exist before the holder starts.
    const plain = join(dir, '.opendde_harness_history')
    const barrier = join(dir, 'barrier')

    writeFileSync(plain, '\n# 2026-01-01 00:00:00\n+old\n')
    writeFileSync(barrier, '')

    const history = new InputHistory(plain, 10)

    history.load()

    // A second process takes the lock and keeps it, standing in for a rotator
    // that has read the file and not yet renamed its copy over the top.
    const holder = spawn('npx', ['tsx', HOLD, plain, barrier], {
      cwd: join(import.meta.dirname, '..', '..'),
      stdio: ['ignore', 'pipe', 'ignore']
    })

    const said = (what: string): Promise<void> =>
      new Promise(resolve => {
        const seen = (chunk: Buffer): void => {
          if (chunk.toString().includes(what)) {
            holder.stdout?.off('data', seen)
            resolve()
          }
        }

        holder.stdout?.on('data', seen)
      })

    await said('HELD')

    history.append('NEW_SENTINEL')

    // Waited out the lock and returned. The entry must not be in the file:
    // the holder's rename would replace it with what it read before.
    expect(readFileSync(plain, 'utf8')).not.toContain('NEW_SENTINEL')

    rmSync(barrier, { force: true })
    await new Promise<void>(resolve => holder.on('close', () => resolve()))

    // The rotator's rename lands, then the held entry goes out after it.
    writeFileSync(plain, '\n# 2026-01-01 00:00:00\n+old\n')
    history.append('LATER')

    expect(new InputHistory(plain, 10).load()).toEqual(['old', 'NEW_SENTINEL', 'LATER'])
  })

  it('keeps another live instance\u2019s entries when it rotates', () => {
    const plain = join(dir, '.opendde_harness_history')

    writeFileSync(plain, '\n# 2026-01-01 00:00:00\n+a\n\n# 2026-01-02 00:00:00\n+b\n')

    // Two TUIs open at once, each holding the file as it was when it started.
    const first = new InputHistory(plain, 2)
    const second = new InputHistory(plain, 2)

    first.load()
    second.load()

    first.append('c')
    second.append('d')

    // Rotating from a stale cache wrote back `b` and dropped `c`, which needs
    // no simultaneous write — one instance being behind is enough.
    expect(new InputHistory(plain, 2).load()).toEqual(['c', 'd'])
  })

  it('holds an entry back while another writer has the lock, then writes it', () => {
    const lock = `${file}.lock`
    const history = new InputHistory(file, 10)

    history.append('before')
    writeFileSync(lock, '')

    try {
      history.append('during')

      // Not written. The lock holder is most likely a rotation that read the
      // file before this append and will rename its own copy over the top:
      // going ahead anyway acknowledged an entry that was about to vanish.
      expect(new InputHistory(file, 10).load()).toEqual(['before'])
      // The editor still has it; only the file is waiting.
      expect(history.load()).toEqual(['before', 'during'])
    } finally {
      rmSync(lock, { force: true })
    }

    history.append('after')

    // Both of them, in the order they were typed.
    expect(new InputHistory(file, 10).load()).toEqual(['before', 'during', 'after'])
  })

  it('does not rotate while another writer holds the lock', () => {
    // Straight in the temporary directory: the seeded file has to exist
    // before the lock does.
    const plain = join(dir, '.opendde_harness_history')
    const lock = `${plain}.lock`

    writeFileSync(plain, '\n# 2026-01-01 00:00:00\n+a\n\n# 2026-01-02 00:00:00\n+b\n\n# 2026-01-03 00:00:00\n+c\n')

    const history = new InputHistory(plain, 2)

    history.load()
    writeFileSync(lock, '')

    try {
      history.append('d')

      // Neither written nor rotated: rewriting the file from a read taken
      // while someone else is writing is exactly the lost update the lock
      // exists for. No temp file either.
      expect(new InputHistory(plain, 10).load()).toEqual(['a', 'b', 'c'])
      expect(readdirSync(dir).filter(name => name.endsWith('.tmp'))).toEqual([])
    } finally {
      rmSync(lock, { force: true })
    }

    // Free again: the held entry and the new one land, and the cap applies.
    history.append('e')

    expect(new InputHistory(plain, 10).load()).toEqual(['d', 'e'])
  })

  it('never takes a lock from a writer that is still running, however old it is', () => {
    const lock = `${file}.lock`
    const history = new InputHistory(file, 10)

    history.append('before')
    // This process is alive by definition, and the lock says so. A rotator
    // paused between its read and its rename looks exactly like this.
    writeFileSync(lock, lockRecord({ id: 'held-by-a-live-writer', pid: process.pid, start: startOf(process.pid) }))

    const old = new Date(Date.now() - 600_000)

    utimesSync(lock, old, old)

    try {
      history.append('during')

      // Ten minutes old and still not broken: age was the only evidence
      // before this, and a slow writer's rename then overwrote the append
      // that was let through.
      expect(existsSync(lock)).toBe(true)
      expect(new InputHistory(file, 10).load()).toEqual(['before'])
    } finally {
      rmSync(lock, { force: true })
    }

    history.append('after')
    expect(new InputHistory(file, 10).load()).toEqual(['before', 'during', 'after'])
  })

  it('takes a lock whose owner is gone without waiting for it to age', () => {
    const lock = `${file}.lock`
    const history = new InputHistory(file, 10)

    history.append('before')
    // A pid nothing is running. The lock is fresh, so the age rule would have
    // made this writer wait out its whole timeout and then buffer.
    writeFileSync(lock, lockRecord({ id: 'held-by-the-departed', pid: deadPid() }))
    history.append('after')

    expect(existsSync(lock)).toBe(false)
    expect(new InputHistory(file, 10).load()).toEqual(['before', 'after'])
  })

  it('never breaks a lock issued by another host, however dead its pid looks here', () => {
    const lock = `${file}.lock`
    const history = new InputHistory(file, 10)

    history.append('before')
    // A pid this kernel is certainly not running — but the lock says another
    // kernel issued it, so the local ESRCH is an answer about the wrong
    // machine. On a shared history path that writer may be mid-rotation.
    writeFileSync(lock, foreignLockRecord({ id: 'held-on-another-host', pid: deadPid() }))

    const old = new Date(Date.now() - 600_000)

    // Nor is age evidence: that mtime was stamped by a clock this machine
    // does not share.
    utimesSync(lock, old, old)

    try {
      history.append('during')

      expect(existsSync(lock)).toBe(true)
      expect(new InputHistory(file, 10).load()).toEqual(['before'])
    } finally {
      rmSync(lock, { force: true })
    }

    // Nothing was lost by waiting: the held entry lands on the next append.
    history.append('after')
    expect(new InputHistory(file, 10).load()).toEqual(['before', 'during', 'after'])
  })

  it('keeps a lock from a PID namespace this kernel does not address', () => {
    const lock = `${file}.lock`
    const history = new InputHistory(file, 10)

    history.append('before')
    // Same hostname, same boot, a namespace whose pid numbering is its own.
    // A container sharing the history file but not the pid table is the
    // ordinary case: its pids neither collide with this one's nor answer to
    // this one's kernel.
    writeFileSync(
      lock,
      JSON.stringify({
        ...lockProvenance(),
        id: 'held-in-a-container',
        ns: 'pid:[4026500001]',
        pid: deadPid(),
        start: ''
      })
    )

    const old = new Date(Date.now() - 600_000)

    utimesSync(lock, old, old)

    try {
      history.append('during')

      // Nothing answers to that pid here, but the writer is not addressed by
      // this pid table at all, so the local ESRCH proves nothing about it.
      expect(existsSync(lock)).toBe(true)
    } finally {
      rmSync(lock, { force: true })
    }
  })

  it('falls back to age for a lock carrying no provenance at all', () => {
    const lock = `${file}.lock`
    const history = new InputHistory(file, 10)

    history.append('before')
    // Nothing this build writes looks like this. Treated the way an
    // unreadable lock always has been: by age, so a fresh one is respected.
    writeFileSync(lock, JSON.stringify({ id: 'no-provenance', pid: deadPid(), start: '' }))
    history.append('during')

    expect(existsSync(lock)).toBe(true)

    const old = new Date(Date.now() - 600_000)

    utimesSync(lock, old, old)
    history.append('after')

    expect(existsSync(lock)).toBe(false)
    expect(new InputHistory(file, 10).load()).toEqual(['before', 'during', 'after'])
  })

  it('leaves a lock alone when it is no longer the one this writer took', () => {
    const plain = join(dir, '.opendde_harness_history')
    const lock = `${plain}.lock`
    const history = new InputHistory(plain, 10)

    // Replace the lock from under the writer, mid-write, the way breaking a
    // stale lock does. Releasing must not remove the new owner's file.
    const other = lockRecord({ id: 'somebody-else', pid: process.pid, start: startOf(process.pid) })

    history.append(APPEND_SENTINEL)
    writeFileSync(lock, other)
    rmSync(lock, { force: true })
    history.append('one more')

    expect(existsSync(lock)).toBe(false)
  })

  it('writes what an earlier append could not, without waiting for another prompt', () => {
    const lock = `${file}.lock`
    const history = new InputHistory(file, 10)

    history.append('before')
    writeFileSync(lock, lockRecord({ id: 'live', pid: process.pid, start: startOf(process.pid) }))

    try {
      history.append('the last thing typed')
      expect(new InputHistory(file, 10).load()).toEqual(['before'])
    } finally {
      rmSync(lock, { force: true })
    }

    // The way out calls this. Before it, a session whose final prompt lost the
    // race needed another prompt to persist it, which it never got.
    expect(history.flush()).toBe(true)
    expect(new InputHistory(file, 10).load()).toEqual(['before', 'the last thing typed'])
    // Nothing left to write, and saying so is not a failure.
    expect(history.flush()).toBe(true)
  })

  it('breaks a lock left behind by a process that died holding it', () => {
    const lock = `${file}.lock`
    const history = new InputHistory(file, 10)

    history.append('before')
    writeFileSync(lock, '')

    const old = new Date(Date.now() - 60_000)

    utimesSync(lock, old, old)
    history.append('after')

    // Removed, which a writer that merely waited out the lock and wrote
    // through would not have done. Nothing is timed: a loaded host makes a
    // duration assertion say more about the host than about the lock.
    expect(existsSync(lock)).toBe(false)
    expect(new InputHistory(file, 10).load()).toEqual(['before', 'after'])
  })

  it('creates the file and its directory for the owner alone', () => {
    new InputHistory(file).append('private')

    // Prompts are what the user typed. Under umask 022 these were 0644/0755.
    expect(statSync(file).mode & 0o777).toBe(0o600)
    expect(statSync(dirname(file)).mode & 0o777).toBe(0o700)
  })
})

describe('InputHistory across processes', () => {
  const FIXTURE = join(import.meta.dirname, 'fixtures', 'appendHistory.ts')

  let dir: string
  let file: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'harness-history-procs-'))
    file = join(dir, '.opendde_harness_history')
  })

  afterEach(() => {
    rmSync(dir, { force: true, recursive: true })
  })

  /** Append `entries` from a second Node process, through the real class. */
  function child(max: number, entries: string[]): ReturnType<typeof spawn> {
    return spawn('npx', ['tsx', FIXTURE, file, String(max), ...entries], {
      cwd: join(import.meta.dirname, '..', '..'),
      stdio: 'ignore'
    })
  }

  const ended = (proc: ReturnType<typeof spawn>): Promise<number> =>
    new Promise(resolve => proc.on('close', code => resolve(code ?? -1)))

  it('keeps what another process wrote while this one was holding a stale view', async () => {
    writeFileSync(file, '\n# 2026-01-01 00:00:00\n+a\n\n# 2026-01-02 00:00:00\n+b\n')

    // Open and load before the other process writes: this instance's cache is
    // `a, b` for the rest of the test.
    const stale = new InputHistory(file, 2)

    stale.load()

    expect(await ended(child(2, ['c', 'd']))).toBe(0)

    // Rotates, because the file is over the cap.
    stale.append('e')

    expect(new InputHistory(file, 2).load()).toEqual(['d', 'e'])
  })

  it('loses nothing when the two processes write at the same time', async () => {
    const max = 6
    const theirs = Array.from({ length: 8 }, (_, i) => `child-${i}`)
    const mine = Array.from({ length: 8 }, (_, i) => `parent-${i}`)
    const history = new InputHistory(file, max)

    history.load()

    const other = child(max, theirs)

    // While the child is starting and writing, so appends and rotations from
    // the two processes interleave.
    for (const entry of mine) {
      history.append(entry)
      await new Promise(resolve => setTimeout(resolve, 12))
    }

    expect(await ended(other)).toBe(0)

    const kept = new InputHistory(file, max).load()

    expect(kept).toHaveLength(max)

    // Whatever survived, each process's surviving entries must be a run from
    // the end of what it wrote. An older entry kept while a newer one is gone
    // is a lost update: someone's rotation wrote back a view of the file that
    // predated the other's append.
    for (const written of [theirs, mine]) {
      const survived = written.filter(entry => kept.includes(entry))

      expect(survived).toEqual(written.slice(written.length - survived.length))
    }
  })
})
