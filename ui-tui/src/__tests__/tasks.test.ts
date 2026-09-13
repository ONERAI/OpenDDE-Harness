// The protein-design task monitor, over real temporary directories and a clock
// the test drives. Nothing here starts a worker, contacts compute, probes a PID
// or writes a snapshot the UI would then read back as its own answer.

import type { FileHandle } from 'node:fs/promises'

import { execFileSync } from 'node:child_process'
import { promises } from 'node:fs'
import { chmod, mkdir, mkdtemp, realpath, rename, rm, stat, symlink, utimes, writeFile } from 'node:fs/promises'
import { syncBuiltinESMExports } from 'node:module'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { ProteinProgressPayload } from '../tasks/monitor.js'

import { isInside, readLogTail, readSnapshot, resolveRoot } from '../tasks/files.js'
import { ProteinDesignTasks } from '../tasks/monitor.js'
import { shortTaskId } from '../tasks/types.js'
import { ManualClock } from './selectorHarness.js'

let root = ''
let clock: ManualClock
let announced: { ok: boolean; text: string }[] = []
const monitors: ProteinDesignTasks[] = []

/** The next timer is armed only after the filesystem refresh has finished. */
const waitForPoll = () => expect.poll(() => clock.ticks.length).toBe(1)

beforeEach(async () => {
  // macOS may expose the temporary directory through /tmp or /var aliases.
  root = await realpath(await mkdtemp(join(tmpdir(), 'opendde-tasks-')))
  clock = new ManualClock(1_700_000_000_000)
  announced = []
})

afterEach(async () => {
  for (const monitor of monitors.splice(0)) {
    monitor.dispose()
  }

  await rm(root, { force: true, recursive: true })
})

function build(options: { refreshMs?: number; root?: string } = {}): ProteinDesignTasks {
  const monitor = new ProteinDesignTasks({
    clock,
    onTerminal: (text, ok) => announced.push({ ok, text }),
    root: options.root ?? root,
    ...(options.refreshMs === undefined ? {} : { refreshMs: options.refreshMs })
  })

  monitors.push(monitor)

  return monitor
}

async function snapshot(taskId: string, fields: Record<string, unknown>): Promise<void> {
  await mkdir(join(root, taskId), { recursive: true })
  await writeFile(join(root, taskId, 'snapshot.json'), JSON.stringify({ task_id: taskId, ...fields }), 'utf8')
}

async function raw(taskId: string, body: string): Promise<void> {
  await mkdir(join(root, taskId), { recursive: true })
  await writeFile(join(root, taskId, 'snapshot.json'), body, 'utf8')
}

let eventCounter = 0

function event(overrides: Partial<ProteinProgressPayload> = {}): ProteinProgressPayload {
  eventCounter += 1

  return {
    actor: 'orchestrator',
    candidate_count: null,
    cycle: null,
    duration_ms: null,
    error: null,
    event_id: `event-${eventCounter}`,
    event_type: 'task',
    has_details: false,
    phase: 'design_cycle',
    skill: null,
    status: 'progress',
    summary: '',
    task_id: 'abcdef0123456789',
    timestamp: '2026-09-11T00:00:00.000Z',
    tool: null,
    total_cycles: null,
    ...overrides
  }
}

describe('when a task was last observed', () => {
  const BASE = 1_600_000_000_000

  /** Four tasks, each snapshot a minute newer than the last. */
  async function four(): Promise<string[]> {
    const ids = ['task-alpha', 'task-bravo', 'task-charlie', 'task-delta']

    for (const [index, id] of ids.entries()) {
      await snapshot(id, { status: 'running', target: id })

      const at = new Date(BASE + index * 60_000)

      await utimes(join(root, id, 'snapshot.json'), at, at)
    }

    return ids
  }

  it('orders tasks discovered in one pass by their snapshot mtime', async () => {
    // All four are found in the same pass and read concurrently. Anything
    // stamped at discovery time orders them by whichever read happened to
    // finish first; the snapshot's own mtime is the only stable answer, and it
    // is the one a user starting four designs expects to see.
    const ids = await four()
    const monitor = build()

    await monitor.refresh()

    expect(monitor.list().map(task => task.taskId)).toEqual([...ids].reverse())
    expect(monitor.list().map(task => task.observedAt)).toEqual([3, 2, 1, 0].map(step => BASE + step * 60_000))
  })

  it('does not let an event with an older timestamp age what a snapshot said', async () => {
    await four()

    const monitor = build()

    await monitor.refresh()
    monitor.applyProgress(
      event({ cycle: 2, status: 'progress', task_id: 'task-delta', timestamp: new Date(BASE).toISOString() })
    )

    // The event is real news about the cycle and stale news about the time.
    const delta = monitor.list().find(task => task.taskId === 'task-delta')!

    expect(delta.cycle).toBe(2)
    expect(delta.observedAt).toBe(BASE + 3 * 60_000)
    expect(monitor.list()[0]!.taskId).toBe('task-delta')
  })

  it('shows no time at all for a task it has never managed to read', async () => {
    await mkdir(join(root, 'task-broken'), { recursive: true })
    await writeFile(join(root, 'task-broken', 'snapshot.json'), 'not json', 'utf8')

    const monitor = build()

    await monitor.refresh()

    const broken = monitor.list().find(task => task.taskId === 'task-broken')!

    expect(broken.observedAt).toBe(0)
    expect(broken.unavailable).toBeTruthy()
  })
})

describe('reading snapshots', () => {
  it('takes the snapshot as the authority on status, target and compute', async () => {
    await snapshot('task-one', {
      best_candidate: { objective: 0.5 },
      compute_url: 'http://10.0.0.4:8080',
      compute_worker_id: 'worker-3',
      cycle: 3,
      phase: 'design_cycle',
      selected_skill: 'relax',
      status: 'running',
      target: 'VEGF binder',
      total_cycles: 10
    })

    const monitor = build()

    await monitor.refresh()

    const [task] = monitor.list()

    expect(task).toMatchObject({
      bestObjective: 0.5,
      computeWorkerId: 'worker-3',
      cycle: 3,
      fromSnapshot: true,
      status: 'running',
      statusSource: 'snapshot',
      target: 'VEGF binder',
      totalCycles: 10
    })
  })

  it('keeps a zero objective and a zero cycle, and leaves missing numbers missing', async () => {
    await snapshot('zeroes', { best_candidate: { objective: 0 }, cycle: 0, status: 'running' })

    const monitor = build()

    await monitor.refresh()

    const [task] = monitor.list()

    expect(task?.bestObjective).toBe(0)
    expect(task?.cycle).toBe(0)
    // No denominator means no denominator; the old store turned it into zero
    // and the bar then drew `0/0`.
    expect(task?.totalCycles).toBeNull()
  })

  it('leaves an unrecognised status unknown rather than guessing running', async () => {
    await snapshot('odd', { status: 'wat' })

    const monitor = build()

    await monitor.refresh()
    expect(monitor.list()[0]?.status).toBe('unknown')
  })

  it('keeps a stopped task, and every other terminal one, in the list', async () => {
    await snapshot('stopped-one', { status: 'stopped', target: 'A' })
    await snapshot('done-one', { status: 'completed', target: 'B' })

    const monitor = build()

    await monitor.refresh()
    expect(
      monitor
        .list()
        .map(task => task.status)
        .sort()
    ).toEqual(['completed', 'stopped'])
  })

  it('refuses a snapshot whose task_id names another task', async () => {
    await mkdir(join(root, 'claimed'), { recursive: true })
    await writeFile(join(root, 'claimed', 'snapshot.json'), JSON.stringify({ status: 'running', task_id: 'other' }))

    const monitor = build()

    await monitor.refresh()
    expect(monitor.list()[0]).toMatchObject({ taskId: 'claimed', unavailable: expect.stringContaining('this task') })
  })

  it('marks a malformed file unavailable without hiding the healthy tasks beside it', async () => {
    await raw('broken', '{ not json')
    await snapshot('healthy', { status: 'running', target: 'ok' })

    const monitor = build()

    await monitor.refresh()

    const byId = Object.fromEntries(monitor.list().map(task => [task.taskId, task]))

    expect(byId.broken?.unavailable).toBeTruthy()
    expect(byId.healthy?.status).toBe('running')
  })

  it('reports an oversize snapshot as unavailable rather than parsing it', async () => {
    await raw('huge', `{"task_id":"huge","status":"running","pad":"${'x'.repeat(1024 * 1024 + 16)}"}`)

    const monitor = build()

    await monitor.refresh()
    expect(monitor.list()[0]?.unavailable).toContain('larger than')
  })

  it('treats a task root that is not there as no tasks at all', async () => {
    const monitor = build({ root: join(root, 'never-created') })

    await monitor.refresh()
    expect(monitor.list()).toEqual([])
    expect(monitor.rootFailure).toBeNull()
  })

  it('drops a snapshot-only row whose directory is gone, and keeps an event-derived one', async () => {
    await snapshot('gone-soon', { status: 'running' })

    const monitor = build()

    await monitor.refresh()
    monitor.applyProgress(event({ task_id: 'event-only', status: 'progress' }))
    expect(monitor.list()).toHaveLength(2)

    await rm(join(root, 'gone-soon'), { force: true, recursive: true })
    await monitor.refresh()

    const ids = monitor.list().map(task => task.taskId)

    expect(ids).toEqual(['event-only'])
    expect(monitor.list()[0]).toMatchObject({ statusSource: 'event', unavailable: expect.stringContaining('host') })
  })
})

describe('staying inside the task root', () => {
  it('recognises what is inside a directory and what only looks like it', async () => {
    expect(isInside('/tasks', '/tasks')).toBe(true)
    expect(isInside('/tasks', '/tasks/a/snapshot.json')).toBe(true)
    // The prefix test has to be on a separator, or a sibling directory whose
    // name merely starts with the root's passes it.
    expect(isInside('/tasks', '/tasks-elsewhere/snapshot.json')).toBe(false)
    expect(isInside('/tasks', '/etc/passwd')).toBe(false)
    expect(isInside('/', '/etc/passwd')).toBe(true)

    expect(await resolveRoot(join(root, 'not-there'))).toBeNull()
    expect(await resolveRoot(root)).toBe(await realpath(root))
  })

  it('refuses a snapshot symlinked out of the root, and says so', async () => {
    const outside = join(dirname(root), 'outside-the-root.json')

    await writeFile(outside, JSON.stringify({ status: 'running', target: 'OUTSIDE', task_id: 'escapee' }), 'utf8')
    await mkdir(join(root, 'escapee'), { recursive: true })
    await symlink(outside, join(root, 'escapee', 'snapshot.json'))

    const monitor = build()

    try {
      await monitor.refresh()

      const [task] = monitor.list()

      // The ID rule stops a path being spelled outside the root, and nothing
      // about the name stops one pointing there. The worker writes regular
      // files, so a link in their place is refused outright.
      expect(task).toMatchObject({ taskId: 'escapee', unavailable: expect.stringContaining('symlink') })
      expect(task?.target).toBeUndefined()
    } finally {
      await rm(outside, { force: true })
    }
  })

  it('refuses a worker log symlinked out of the root', async () => {
    const outside = join(dirname(root), 'outside-the-root.log')

    await writeFile(outside, 'OUTSIDE LOG\n', 'utf8')
    await mkdir(join(root, 'escapee'), { recursive: true })
    await symlink(outside, join(root, 'escapee', 'worker.log'))

    const monitor = build()

    try {
      const tail = await monitor.readLogTail('escapee', 200)

      expect(tail.missing).toBe(true)
      expect(tail.lines).toEqual([])
      expect(tail.reason).toContain('symlink')
    } finally {
      await rm(outside, { force: true })
    }
  })

  it('refuses a snapshot that is a symlink, even to a file inside the root', async () => {
    await mkdir(join(root, 'real'), { recursive: true })
    await mkdir(join(root, 'linked'), { recursive: true })
    await writeFile(
      join(root, 'real', 'snapshot.json'),
      JSON.stringify({ status: 'running', target: 'inside', task_id: 'linked' }),
      'utf8'
    )
    await symlink(join(root, 'real', 'snapshot.json'), join(root, 'linked', 'snapshot.json'))

    const monitor = build()

    await monitor.refresh()

    // The worker writes regular files. Reading a link because it happens to
    // point somewhere acceptable *now* is a promise about a name, and the name
    // is what an attacker gets to change.
    expect(monitor.list().find(task => task.taskId === 'linked')).toMatchObject({
      unavailable: expect.stringContaining('symlink')
    })
  })

  it('refuses a task directory that is a symlink out of the root', async () => {
    const outside = join(dirname(root), 'outside-task-dir')

    await mkdir(outside, { recursive: true })
    await writeFile(
      join(outside, 'snapshot.json'),
      JSON.stringify({ status: 'running', target: 'OUTSIDE', task_id: 'escapee' }),
      'utf8'
    )
    await writeFile(join(outside, 'worker.log'), 'OUTSIDE\n', 'utf8')
    await symlink(outside, join(root, 'escapee'))

    const monitor = build()

    try {
      // Discovery skips it — a symlink is not a directory to `readdir` — but
      // `/task:logs <id>` can name one directly.
      expect(await readSnapshot(await realpath(root), 'escapee')).toMatchObject({
        kind: 'unavailable',
        reason: expect.stringContaining('symlink')
      })

      const tail = await monitor.readLogTail('escapee', 10)

      expect(tail.lines).toEqual([])
      expect(tail.reason).toContain('symlink')
    } finally {
      await rm(outside, { force: true, recursive: true })
    }
  })

  /** A tree whose task root sits under `parent`, created with `mode`, holding
   *  one task that says INSIDE, and one directory outside the root that says
   *  OUTSIDE. */
  async function tree(parent: string, mode: number, taskId = 'aba') {
    const holder = join(root, parent)
    const taskRoot = join(holder, 'tasks')
    const inside = join(taskRoot, taskId)
    const outside = join(root, `outside-${parent}`)

    await mkdir(inside, { recursive: true })
    await mkdir(outside, { recursive: true })
    await writeFile(
      join(inside, 'snapshot.json'),
      JSON.stringify({ status: 'running', target: 'INSIDE', task_id: taskId })
    )
    await writeFile(join(inside, 'worker.log'), 'INSIDE\n', 'utf8')
    await writeFile(
      join(outside, 'snapshot.json'),
      JSON.stringify({ status: 'running', target: 'OUTSIDE', task_id: taskId })
    )
    await writeFile(join(outside, 'worker.log'), 'OUTSIDE\n', 'utf8')
    // Set last: the directories under it have to be created first.
    await chmod(holder, mode)

    return { holder, inside, outside, realRoot: await realpath(taskRoot), taskId }
  }

  /** Run `work` with the fallback branch selected, whatever this host is. */
  async function asDarwin<T>(work: () => Promise<T>): Promise<T> {
    const platform = Object.getOwnPropertyDescriptor(process, 'platform')!

    Object.defineProperty(process, 'platform', { configurable: true, value: 'darwin' })

    try {
      return await work()
    } finally {
      Object.defineProperty(process, 'platform', platform)
    }
  }

  it('reads a task by name where nobody else can write the path to it', async () => {
    // No anchored open, but no way for anyone but this user to rename a
    // component of the path either, so the name still means what it meant.
    const { realRoot, taskId } = await tree('private', 0o700)

    await asDarwin(async () => {
      expect(await readSnapshot(realRoot, taskId)).toMatchObject({ facts: { target: 'INSIDE' }, kind: 'ok' })
      expect((await readLogTail(realRoot, taskId, 10)).lines).toEqual(['INSIDE'])
    })
  })

  it('takes the sticky bit as putting a shared directory back in bounds', async () => {
    // `/tmp` is 1777 on every Unix and is nobody's mistake: sticky means only
    // an entry's owner may rename or remove it, so the swap this check exists
    // to prevent cannot be staged there.
    const { realRoot, taskId } = await tree('sticky', 0o1777)

    await asDarwin(async () => {
      expect(await readSnapshot(realRoot, taskId)).toMatchObject({ facts: { target: 'INSIDE' }, kind: 'ok' })
    })
  })

  it('refuses a root reached through a directory anyone can write, and names it', async () => {
    const { holder, realRoot, taskId } = await tree('shared', 0o777)

    await asDarwin(async () => {
      const snapshot = await readSnapshot(realRoot, taskId)
      const tail = await readLogTail(realRoot, taskId, 10)

      // Actionable: the one directory whose mode has to change is named.
      expect(snapshot).toMatchObject({
        kind: 'unavailable',
        reason: expect.stringContaining(`${holder} is writable by other users`)
      })
      expect(tail.missing).toBe(true)
      expect(tail.reason).toContain(holder)
    })
  })

  it('re-walks the path on every read, because permissions move without the objects doing', async () => {
    // The walk decides about permissions, and a chmod changes no device and no
    // inode. A verdict held against identity therefore goes on saying private
    // about a directory the whole world can write, which is what let a foreign
    // uid rename a task directory out from under a read.
    const { realRoot, taskId } = await tree('rewalked', 0o700)
    const original = promises.lstat
    let calls = 0

    try {
      Object.defineProperty(promises, 'lstat', {
        configurable: true,
        value: async (...args: unknown[]) => {
          calls += 1

          return (original as (...rest: unknown[]) => Promise<unknown>)(...args)
        }
      })
      syncBuiltinESMExports()

      await asDarwin(async () => {
        await readSnapshot(realRoot, taskId)

        const walked = calls

        expect(walked).toBeGreaterThan(2)
        calls = 0
        await readSnapshot(realRoot, taskId)
        expect(calls).toBe(walked)
      })
    } finally {
      Object.defineProperty(promises, 'lstat', { configurable: true, value: original })
      syncBuiltinESMExports()
    }
  })

  it('stops reading a root that was private when it was last read and is not now', async () => {
    // The reviewer's schedule: read once so any verdict is in hand, open the
    // tree up, and read again. Nothing about the root's identity changed.
    const { holder, realRoot, taskId } = await tree('opened-up', 0o700)

    await asDarwin(async () => {
      expect(await readSnapshot(realRoot, taskId)).toMatchObject({ facts: { target: 'INSIDE' }, kind: 'ok' })

      await chmod(holder, 0o777)

      expect(await readSnapshot(realRoot, taskId)).toMatchObject({
        kind: 'unavailable',
        reason: expect.stringContaining(`${holder} is writable by other users`)
      })
      expect((await readLogTail(realRoot, taskId, 10)).missing).toBe(true)
    })
  })

  it('refuses the bytes when the task directory moves during the open', async () => {
    // What is left after the walk: the window between the last check and the
    // open, which only an anchored open could close. A swap that is left in
    // place is caught by reading the name back.
    const { inside, outside, realRoot, taskId } = await tree('window', 0o700)
    const original = promises.open
    let swapped = false

    try {
      Object.defineProperty(promises, 'open', {
        configurable: true,
        value: async (...args: unknown[]) => {
          const openAt = original as (...rest: unknown[]) => Promise<FileHandle>

          if (String(args[0]) === join(inside, 'snapshot.json') && !swapped) {
            swapped = true
            await rm(inside, { force: true, recursive: true })
            await symlink(outside, inside)
          }

          return openAt(...args)
        }
      })
      syncBuiltinESMExports()

      await asDarwin(async () => {
        const snapshot = await readSnapshot(realRoot, taskId)

        expect(swapped).toBe(true)
        expect(JSON.stringify(snapshot)).not.toContain('OUTSIDE')
        expect(snapshot).toMatchObject({
          kind: 'unavailable',
          reason: expect.stringContaining('moved while it was being opened')
        })
      })
    } finally {
      Object.defineProperty(promises, 'open', { configurable: true, value: original })
      syncBuiltinESMExports()
    }
  })

  it('reads the root once its mode is fixed, without a restart', async () => {
    // The refusal names a directory so that someone can go and change it. If
    // the verdict were cached the fix would not take until the next launch.
    const { holder, realRoot, taskId } = await tree('fixable', 0o777)

    await asDarwin(async () => {
      expect(await readSnapshot(realRoot, taskId)).toMatchObject({ kind: 'unavailable' })
      await chmod(holder, 0o700)
      expect(await readSnapshot(realRoot, taskId)).toMatchObject({ facts: { target: 'INSIDE' }, kind: 'ok' })
    })
  })

  it('reads nothing at all when the swap is staged in a tree anyone can write', async () => {
    // The reviewer's schedule, run against the fallback branch by claiming to
    // be macOS: swap the task directory for a link to somewhere else for
    // exactly the length of the file open, then put the directory back. A
    // before/after identity check on the directory sees two identical samples
    // and the descriptor is pointing outside the root. Here the tree is one
    // anyone can write, which is the only tree the swap could be staged in.
    const { inside, outside, realRoot, taskId } = await tree('open', 0o777)
    const parked = join(dirname(inside), `${taskId}-original`)
    const original = promises.open
    const swapped: string[] = []

    try {
      Object.defineProperty(promises, 'open', {
        configurable: true,
        value: async (...args: unknown[]) => {
          const target = String(args[0])
          const openAt = original as (...rest: unknown[]) => Promise<FileHandle>

          if (target !== join(inside, 'snapshot.json') && target !== join(inside, 'worker.log')) {
            return openAt(...args)
          }

          swapped.push(target)
          await rename(inside, parked)
          await symlink(outside, inside)

          const handle = await openAt(...args)

          await rm(inside, { force: true })
          await rename(parked, inside)

          return handle
        }
      })
      syncBuiltinESMExports()

      await asDarwin(async () => {
        const snapshot = await readSnapshot(realRoot, taskId)
        const tail = await readLogTail(realRoot, taskId, 10)

        // Nothing from outside the root, and the reason says why rather than
        // pretending the task is missing.
        expect(JSON.stringify(snapshot)).not.toContain('OUTSIDE')
        expect(tail.lines.join('\n')).not.toContain('OUTSIDE')
        expect(snapshot).toMatchObject({
          kind: 'unavailable',
          reason: expect.stringContaining('writable by other users')
        })
        expect(tail.reason).toContain('writable by other users')
        // The window the fixture opens is never reached: the read is refused
        // before a file is opened by name at all.
        expect(swapped).toEqual([])
      })
    } finally {
      Object.defineProperty(promises, 'open', { configurable: true, value: original })
      syncBuiltinESMExports()
    }
  })

  it.runIf(process.platform === 'linux' || process.platform === 'android')(
    'says the descriptor facility is missing rather than calling the task gone',
    async () => {
      await snapshot('noproc', { status: 'running', target: 'here' })

      const realRoot = await realpath(root)
      const original = promises.realpath

      try {
        Object.defineProperty(promises, 'realpath', {
          configurable: true,
          value: async (...args: unknown[]) => {
            if (String(args[0]).startsWith('/proc/self/fd/')) {
              throw Object.assign(new Error('synthetic proc absent'), { code: 'ENOENT' })
            }

            return (original as (...rest: unknown[]) => Promise<string>)(...args)
          }
        })
        syncBuiltinESMExports()

        expect(await readSnapshot(realRoot, 'noproc')).toMatchObject({
          kind: 'unavailable',
          reason: expect.stringContaining('/proc/self/fd/')
        })
      } finally {
        Object.defineProperty(promises, 'realpath', { configurable: true, value: original })
        syncBuiltinESMExports()
      }
    }
  )

  it.runIf(process.platform === 'linux' || process.platform === 'android')(
    'cannot be redirected by replacing the task directory mid-read',
    async () => {
      // The reviewer's schedule: resolve the task directory, then rename it and
      // drop a symlink to somewhere else in its place before the file is opened.
      // A name-based open follows the new link; a descriptor cannot be moved.
      const inside = join(root, 'racer')
      const outside = join(dirname(root), 'outside-racer')

      await mkdir(inside, { recursive: true })
      await mkdir(outside, { recursive: true })
      await writeFile(
        join(inside, 'snapshot.json'),
        JSON.stringify({ status: 'running', target: 'INSIDE', task_id: 'racer' })
      )
      await writeFile(join(inside, 'worker.log'), 'INSIDE\n', 'utf8')
      await writeFile(
        join(outside, 'snapshot.json'),
        JSON.stringify({ status: 'running', target: 'OUTSIDE', task_id: 'racer' })
      )
      await writeFile(join(outside, 'worker.log'), 'OUTSIDE\n', 'utf8')

      const realRoot = await realpath(root)
      const original = promises.realpath
      let swapped = false

      const swap = async () => {
        if (swapped) {
          return
        }

        swapped = true
        await rename(inside, join(root, 'racer-moved'))
        await symlink(outside, inside)
      }

      try {
        // Every path this reader resolves is resolved through here, so this
        // fires in the window between validating the directory and opening the
        // file inside it — whichever seam the reader happens to use.
        Object.defineProperty(promises, 'realpath', {
          configurable: true,
          value: async (...args: unknown[]) => {
            const result = await (original as (...rest: unknown[]) => Promise<string>)(...args)

            await swap()

            return result
          }
        })
        syncBuiltinESMExports()

        const snapshot = await readSnapshot(realRoot, 'racer')
        const tail = await readLogTail(realRoot, 'racer', 10)

        expect(swapped).toBe(true)
        expect(JSON.stringify(snapshot)).not.toContain('OUTSIDE')
        expect(tail.lines.join('\n')).not.toContain('OUTSIDE')
      } finally {
        Object.defineProperty(promises, 'realpath', { configurable: true, value: original })
        syncBuiltinESMExports()
        await rm(outside, { force: true, recursive: true })
      }
    }
  )

  it('refuses a worker log that is a FIFO instead of waiting for a writer', async () => {
    await mkdir(join(root, 'fifo-log'), { recursive: true })
    execFileSync('mkfifo', [join(root, 'fifo-log', 'worker.log')])

    const monitor = build()

    // A FIFO with no writer parks a blocking open for ever, and the command
    // that awaited it never returns.
    const tail = await monitor.readLogTail('fifo-log', 200)

    expect(tail.missing).toBe(true)
    expect(tail.reason).toContain('not a regular file')
  })

  it('refuses a snapshot that is a FIFO', async () => {
    await mkdir(join(root, 'fifo-snap'), { recursive: true })
    execFileSync('mkfifo', [join(root, 'fifo-snap', 'snapshot.json')])

    const monitor = build()

    await monitor.refresh()
    expect(monitor.list()[0]?.unavailable).toContain('not a regular file')
  })

  it('records the size of the bytes it parsed, from the file it had open', async () => {
    const body = JSON.stringify({ status: 'running', target: 'sized', task_id: 'sized' })

    await mkdir(join(root, 'sized'), { recursive: true })
    await writeFile(join(root, 'sized', 'snapshot.json'), body, 'utf8')

    const monitor = build()

    await monitor.refresh()
    expect(monitor.list()[0]?.snapshotSize).toBe(Buffer.byteLength(body, 'utf8'))

    // A replacement is a different file; its own stamp is what gets stored.
    const longer = JSON.stringify({ status: 'completed', target: 'sized again', task_id: 'sized' })

    await writeFile(join(root, 'sized', 'snapshot.json'), longer, 'utf8')
    await monitor.refresh()

    expect(monitor.list()[0]?.status).toBe('completed')
    expect(monitor.list()[0]?.snapshotSize).toBe(Buffer.byteLength(longer, 'utf8'))
  })

  it('clears the unavailable label once the file can be read again', async () => {
    const path = join(root, 'recovers', 'snapshot.json')
    const body = JSON.stringify({
      cycle: 1,
      status: 'running',
      target: 'back again',
      task_id: 'recovers',
      total_cycles: 4
    })
    // Pinned so the restored file is stamped exactly as the original was, which
    // is what makes the reader answer "unchanged" for it.
    const when = new Date(1_700_000_000_000)

    await mkdir(join(root, 'recovers'), { recursive: true })
    await writeFile(path, body, 'utf8')
    await utimes(path, when, when)

    const monitor = build()

    await monitor.refresh()

    const stamped = await stat(path)

    expect(monitor.list()[0]?.snapshotMtimeMs).toBe(stamped.mtimeMs)
    expect(monitor.list()[0]?.unavailable).toBeUndefined()

    // Unreadable: the name is there but it is not a file any more.
    await rm(path)
    await mkdir(path)
    await monitor.refresh()
    expect(monitor.list()[0]?.unavailable).toContain('not a regular file')

    // Readable again, byte for byte and stamp for stamp. Keeping the stamp of
    // the last good read makes this answer "unchanged", and the row then wears
    // the old error for the rest of the process.
    await rm(path, { recursive: true })
    await writeFile(path, body, 'utf8')
    await utimes(path, when, when)
    expect((await stat(path)).mtimeMs).toBe(stamped.mtimeMs)

    await monitor.refresh()

    expect(monitor.list()[0]?.unavailable).toBeUndefined()
    expect(monitor.list()[0]?.status).toBe('running')
  })
})

describe('merging events with snapshots', () => {
  it('never lets a finished phase, tool or fold finish the whole task', async () => {
    await snapshot('abcdef0123456789', { cycle: 1, status: 'running', target: 'T', total_cycles: 4 })

    const monitor = build()

    await monitor.refresh()

    for (const kind of ['tool', 'phase', 'cycle', 'agent', 'skill', 'fold', 'gate', 'memory'] as const) {
      monitor.applyProgress(event({ event_type: kind, status: 'completed', summary: `${kind} done` }))
      monitor.applyProgress(event({ event_type: kind, status: 'failed', summary: `${kind} failed` }))
    }

    expect(monitor.list()[0]?.status).toBe('running')
    expect(monitor.list()[0]?.provisional).toBeUndefined()
    expect(announced).toEqual([])
  })

  it('labels a task-level terminal event as reported until a snapshot agrees', async () => {
    await snapshot('abcdef0123456789', { cycle: 2, status: 'running', target: 'T', total_cycles: 4 })

    const monitor = build()

    await monitor.refresh()
    monitor.applyProgress(event({ cycle: 4, status: 'completed', total_cycles: 4 }))

    expect(monitor.list()[0]?.provisional).toBe('completed')
    // The snapshot on disk still says running, and an unchanged older file
    // does not get to overrule the event that just arrived.
    expect(monitor.list()[0]?.status).toBe('running')

    await snapshot('abcdef0123456789', { cycle: 4, status: 'completed', target: 'T', total_cycles: 4 })
    await monitor.refresh()

    expect(monitor.list()[0]?.status).toBe('completed')
    expect(monitor.list()[0]?.provisional).toBeUndefined()
  })

  it('gives one transcript line per ending, whichever source saw it first', async () => {
    await snapshot('abcdef0123456789', { cycle: 1, status: 'running', target: 'VEGF', total_cycles: 4 })

    const monitor = build()

    await monitor.refresh()
    monitor.applyProgress(event({ cycle: 4, status: 'completed', total_cycles: 4 }))

    expect(announced).toHaveLength(1)
    expect(announced[0]?.text).toContain('reported completed')
    expect(announced[0]?.text).toContain('VEGF')

    await snapshot('abcdef0123456789', { cycle: 4, status: 'completed', target: 'VEGF', total_cycles: 4 })
    await monitor.refresh()

    expect(announced).toHaveLength(1)
  })

  it('says nothing about tasks that were already finished when it first looked', async () => {
    await snapshot('old-one', { status: 'completed', target: 'history' })
    await snapshot('old-two', { status: 'failed', target: 'history' })

    const monitor = build()

    await monitor.refresh()
    expect(announced).toEqual([])
  })

  it('announces a task that finishes while it is watching', async () => {
    await snapshot('live', { cycle: 1, status: 'running', target: 'live one', total_cycles: 3 })

    const monitor = build()

    await monitor.refresh()
    await snapshot('live', { cycle: 3, status: 'failed', target: 'live one', total_cycles: 3 })
    await monitor.refresh()

    expect(announced).toHaveLength(1)
    expect(announced[0]).toMatchObject({ ok: false })
    expect(announced[0]?.text).toContain('failed')
  })

  it('runs a task on events alone when no snapshot is on this host', () => {
    const monitor = build()

    monitor.applyProgress(event({ cycle: 2, phase: 'fold', status: 'progress', total_cycles: 5 }))

    expect(monitor.list()[0]).toMatchObject({
      cycle: 2,
      fromSnapshot: false,
      phase: 'fold',
      status: 'running',
      statusSource: 'event',
      totalCycles: 5
    })
  })

  it('drops a repeated event id and an out-of-order timestamp', () => {
    const monitor = build()
    const first = event({ cycle: 1, timestamp: '2026-09-11T00:00:10.000Z' })

    monitor.applyProgress(first)
    monitor.applyProgress(first)
    monitor.applyProgress(event({ cycle: 9, timestamp: '2026-09-11T00:00:05.000Z' }))

    expect(monitor.list()[0]?.cycle).toBe(1)

    monitor.applyProgress(event({ cycle: 2, timestamp: '2026-09-11T00:00:20.000Z' }))
    expect(monitor.list()[0]?.cycle).toBe(2)
  })

  it('keeps an event that arrives while a snapshot read is in flight', async () => {
    await snapshot('abcdef0123456789', { cycle: 1, phase: 'setup', status: 'running', target: 'T', total_cycles: 4 })

    const monitor = build()
    const reading = monitor.refresh()

    monitor.applyProgress(event({ cycle: 3, phase: 'scoring', status: 'progress', total_cycles: 4 }))
    await reading

    // The file said cycle 1 in setup; the event is newer than the read that
    // was already on its way, so its patch goes back on top of the facts.
    expect(monitor.list()[0]).toMatchObject({ cycle: 3, phase: 'scoring', target: 'T' })
  })

  it('ignores an event whose task id is not a task id', () => {
    const monitor = build()

    monitor.applyProgress(event({ task_id: '../escape' }))
    monitor.applyProgress(event({ task_id: '' }))

    expect(monitor.list()).toEqual([])
  })
})

describe('polling', () => {
  it('reads once on show, arms exactly one timer, and stops on hide', async () => {
    await snapshot('one', { status: 'running' })

    const monitor = build({ refreshMs: 2000 })

    expect(monitor.visible).toBe(false)
    expect(clock.ticks).toHaveLength(0)

    monitor.setVisible(true)
    await waitForPoll()

    expect(monitor.list()).toHaveLength(1)
    expect(clock.ticks).toHaveLength(1)

    monitor.setVisible(false)
    expect(clock.ticks).toHaveLength(0)
  })

  it('keeps polling while the bar is empty, so a new task can still be found', async () => {
    const monitor = build({ refreshMs: 2000 })

    monitor.setVisible(true)
    await waitForPoll()
    expect(monitor.list()).toEqual([])

    await snapshot('late', { status: 'running' })
    clock.advance(2000)
    await waitForPoll()

    expect(monitor.list().map(task => task.taskId)).toEqual(['late'])
  })

  it('never runs two passes at once', async () => {
    const monitor = build()
    const first = monitor.refresh()

    expect(monitor.refresh()).toBe(first)
    await first
  })

  it('stops the timer for a handoff and starts again with a read', async () => {
    const monitor = build({ refreshMs: 2000 })

    monitor.setVisible(true)
    await waitForPoll()
    expect(clock.ticks).toHaveLength(1)

    monitor.setPaused(true)
    expect(clock.ticks).toHaveLength(0)

    await snapshot('during-handoff', { status: 'running' })
    monitor.setPaused(false)
    await waitForPoll()

    expect(monitor.list().map(task => task.taskId)).toEqual(['during-handoff'])
    expect(clock.ticks).toHaveLength(1)
  })

  it('leaves no timer and no listener behind when it is disposed', async () => {
    const monitor = build({ refreshMs: 2000 })
    let redraws = 0

    monitor.subscribe(() => {
      redraws += 1
    })
    monitor.setVisible(true)
    await waitForPoll()

    const before = redraws

    monitor.dispose()
    clock.advance(10_000)

    expect(clock.ticks).toHaveLength(0)
    expect(redraws).toBe(before)
  })
})

describe('lookup and logs', () => {
  it('resolves an exact id, a unique prefix, and reports an ambiguous one', async () => {
    await snapshot('aaaa1111', { status: 'running' })
    await snapshot('aaaa2222', { status: 'running' })
    await snapshot('bbbb1111', { status: 'running' })

    const monitor = build()

    await monitor.refresh()

    expect(monitor.resolve('aaaa1111')).toMatchObject({ kind: 'ok' })
    expect(monitor.resolve('bbbb')).toMatchObject({ kind: 'ok' })
    const ambiguous = monitor.resolve('aaaa')

    // Ordered by when each was last observed, which for three files written in
    // the same breath is not something to assert on.
    expect(ambiguous.kind).toBe('ambiguous')
    expect([...(ambiguous.kind === 'ambiguous' ? ambiguous.ids : [])].sort()).toEqual(['aaaa1111', 'aaaa2222'])
    expect(monitor.resolve('zzzz')).toEqual({ kind: 'none' })
    expect(monitor.resolve('  ')).toEqual({ kind: 'none' })
  })

  it('shows at least eight characters, and more when eight would be ambiguous', () => {
    const ids = ['abcdefgh1111', 'abcdefgh2222', 'zyxwvutsrq']

    expect(shortTaskId('zyxwvutsrq', ids)).toBe('zyxwvuts')
    expect(shortTaskId('abcdefgh1111', ids)).toBe('abcdefgh1')
    expect(shortTaskId('short', ids)).toBe('short')
  })

  it('tails the worker log and says so when there is none', async () => {
    await mkdir(join(root, 'logged'), { recursive: true })
    await writeFile(join(root, 'logged', 'worker.log'), 'line one\nline two\nline three\n', 'utf8')

    const monitor = build()

    expect(await monitor.readLogTail('logged', 2)).toEqual({
      bytes: 29,
      lines: ['line two', 'line three'],
      missing: false,
      truncated: false
    })
    expect(await monitor.readLogTail('logged', 200)).toMatchObject({ truncated: false })

    await mkdir(join(root, 'quiet'), { recursive: true })
    expect(await monitor.readLogTail('quiet', 200)).toEqual({ bytes: 0, lines: [], missing: true, truncated: false })
  })

  it('keeps multi-byte characters whole in a log tail', async () => {
    await mkdir(join(root, 'utf8'), { recursive: true })
    await writeFile(join(root, 'utf8', 'worker.log'), 'α β γ\n蛋白质设计\n', 'utf8')

    const monitor = build()

    expect((await monitor.readLogTail('utf8', 10)).lines).toEqual(['α β γ', '蛋白质设计'])
  })

  it('says a log window held no complete line rather than calling it empty', async () => {
    await mkdir(join(root, 'oneline'), { recursive: true })
    // One unterminated line past the read window: the incomplete prefix is
    // dropped, so there is nothing left to show, but the file is far from
    // empty and the worker may still be writing to it.
    await writeFile(join(root, 'oneline', 'worker.log'), 'x'.repeat(1024 * 1024 + 1), 'utf8')

    const monitor = build()
    const tail = await monitor.readLogTail('oneline', 200)

    expect(tail.lines).toEqual([])
    expect(tail.truncated).toBe(true)
    expect(tail.bytes).toBe(1024 * 1024 + 1)
    expect(tail.missing).toBe(false)
  })

  it('refuses a log path that is not a task id', async () => {
    const monitor = build()

    await expect(monitor.readLogTail('../etc', 10)).rejects.toThrow('invalid protein-design task id')
  })
})
