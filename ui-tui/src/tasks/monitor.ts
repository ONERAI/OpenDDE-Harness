// The protein-design task monitor: one owner for what the UI knows about the
// detached workers on this host.
//
// Two sources feed it and neither is a sequence. `protein_design.progress`
// events are task-global background news with an ID and an ISO time but no
// revision; `snapshot.json` is the persisted authority with an mtime the worker
// never promised to be monotonic against those events. So the merge below is
// explicit about what each one may say: a snapshot owns the task's status, an
// event owns the current phase and cycle, and only a task-level terminal event
// may claim the whole design finished — and then only provisionally, until a
// snapshot agrees.
//
// It polls only while the bar is shown, one read at a time, re-arming the
// countdown after each refresh settles rather than on a fixed interval.

import type { CountdownClock } from '../components/countdownTimer.js'
import type { ProteinDesignProgressEvent } from '../rpc/index.js'
import type { LogTail } from './files.js'
import type { SnapshotFacts, TaskRecord, TaskStatus } from './types.js'

import { CountdownTimer } from '../components/countdownTimer.js'
import { listTaskIds, mapLimit, readLogTail, readSnapshot, READ_CONCURRENCY, resolveRoot, taskRoot } from './files.js'
import { formatProgress, isActiveStatus, isTaskId, isTerminalStatus, sanitizeLine, shortTaskId } from './types.js'

export type ProteinProgressPayload = ProteinDesignProgressEvent['payload']

/** The old UI's cadence, kept. */
export const REFRESH_MS = 2000

/** Event IDs remembered per task, to drop repeats. */
const SEEN_EVENTS = 32

export type TaskResolution = { ids: string[]; kind: 'ambiguous' } | { kind: 'none' } | { kind: 'ok'; task: TaskRecord }

export interface ProteinDesignTasksOptions {
  /** The one timing primitive. Scheduling only: no display, no render. */
  clock: CountdownClock
  env?: NodeJS.ProcessEnv
  /** One line for a task that reached a terminal state, for the transcript. */
  onTerminal?: (text: string, ok: boolean) => void
  refreshMs?: number
  /** Overrides the environment; tests point it at a temporary directory. */
  root?: string
}

/** The event-derived fields, kept so a snapshot read that started before an
 *  event arrived cannot quietly undo that event's patch. */
interface EventPatch {
  activity?: string
  cycle: null | number
  error?: string
  phase?: string
  skill?: string
  totalCycles: null | number
}

interface Entry {
  /** Set once a terminal line has been printed, so an event and the poll that
   *  confirms it cannot produce two. */
  announced: boolean
  eventAt: null | number
  /** Bumped by every applied event. */
  eventRevision: number
  /** The entry has been through one update; the next terminal transition is a
   *  transition rather than the discovery of an already-finished task. */
  known: boolean
  patch: EventPatch | null
  record: TaskRecord
  sawEvent: boolean
  seen: string[]
}

/** A task we have found but not yet learned anything about.
 *
 *  `observedAt` is zero rather than the clock: discovering a directory is not
 *  an observation of the task in it, and stamping it with `now` would make
 *  every task found in one pass look freshly observed and sort by the accident
 *  of which read finished first. Zero loses every comparison, so the first
 *  snapshot mtime or event time wins outright. */
function blankRecord(taskId: string): TaskRecord {
  return {
    cycle: null,
    fromSnapshot: false,
    observedAt: 0,
    status: 'unknown',
    statusSource: 'none',
    taskId,
    totalCycles: null
  }
}

/** Drop the keys an update left undefined rather than store them as holes, so
 *  `'error' in record` and a JSON dump of it agree with what is known. */
function compact(record: TaskRecord): TaskRecord {
  const out: Partial<Record<string, unknown>> = { ...record }

  for (const key of Object.keys(out)) {
    if (out[key] === undefined) {
      delete out[key]
    }
  }

  return out as unknown as TaskRecord
}

export class ProteinDesignTasks {
  private readonly clock: CountdownClock
  private readonly entries = new Map<string, Entry>()
  private readonly listeners = new Set<() => void>()
  private readonly onTerminal: ((text: string, ok: boolean) => void) | undefined
  private readonly refreshMs: number
  private readonly root: string

  private disposed = false
  private focused: null | string = null
  /** Bumped whenever recurring work must stop mattering: hide, pause, dispose. */
  private generation = 0
  private paused = false
  private pending: null | Promise<void> = null
  private revisionCount = 0
  /** Why the root itself could not be listed, if it could not. */
  private rootError: null | string = null
  private timer: CountdownTimer | undefined
  private visibleFlag = false

  constructor(options: ProteinDesignTasksOptions) {
    this.clock = options.clock
    this.onTerminal = options.onTerminal
    this.refreshMs = options.refreshMs ?? REFRESH_MS
    this.root = options.root ?? taskRoot(options.env)
  }

  /** Bumped by every change worth redrawing; components cache on it. */
  get revision(): number {
    return this.revisionCount
  }

  /** The `/tasks show` setting. True even with no rows: otherwise a task
   *  started after the bar was shown could never be discovered. */
  get visible(): boolean {
    return this.visibleFlag
  }

  get focusedTaskId(): null | string {
    return this.focused
  }

  /** Why the task root could not be read, or null. Reported, never hidden. */
  get rootFailure(): null | string {
    return this.rootError
  }

  get rootPath(): string {
    return this.root
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener)

    return () => void this.listeners.delete(listener)
  }

  /** Most recently observed first, ID as the tie-break. */
  list(): TaskRecord[] {
    return [...this.entries.values()]
      .map(entry => entry.record)
      .sort((a, b) => b.observedAt - a.observedAt || a.taskId.localeCompare(b.taskId))
  }

  /** Exact ID first, then a unique case-sensitive prefix. */
  resolve(query: string): TaskResolution {
    const needle = query.trim()

    if (!needle) {
      return { kind: 'none' }
    }

    const exact = this.entries.get(needle)

    if (exact) {
      return { kind: 'ok', task: exact.record }
    }

    const matches = this.list().filter(task => task.taskId.startsWith(needle))

    if (matches.length === 1) {
      return { kind: 'ok', task: matches[0]! }
    }

    return matches.length === 0 ? { kind: 'none' } : { ids: matches.map(task => task.taskId), kind: 'ambiguous' }
  }

  /** The task the bar puts first. */
  focus(taskId: null | string): void {
    this.focused = taskId === null || this.entries.has(taskId) ? taskId : this.focused
    this.bump()
  }

  setVisible(visible: boolean): void {
    if (visible === this.visibleFlag || this.disposed) {
      return
    }

    this.visibleFlag = visible
    this.generation += 1

    if (visible) {
      void this.pollLoop(this.generation)
    } else {
      this.stopTimer()
    }

    this.bump()
  }

  /** A child process owns the terminal. Recurring polling stops; the next
   *  resume starts with an immediate read. */
  setPaused(paused: boolean): void {
    if (paused === this.paused || this.disposed) {
      return
    }

    this.paused = paused
    this.generation += 1

    if (paused) {
      this.stopTimer()
    } else if (this.visibleFlag) {
      void this.pollLoop(this.generation)
    }
  }

  /**
   * One pass over the task root. Callable while hidden — a read-only command
   * does exactly that — and never overlapping: a caller that arrives while a
   * pass is running awaits that pass instead of starting a second one.
   */
  refresh(): Promise<void> {
    if (this.pending) {
      return this.pending
    }

    this.pending = this.runRefresh().finally(() => {
      this.pending = null
    })

    return this.pending
  }

  async readLogTail(taskId: string, maxLines: number): Promise<LogTail> {
    const realRoot = await resolveRoot(this.root)

    if (realRoot === null) {
      return { bytes: 0, lines: [], missing: true, truncated: false }
    }

    return readLogTail(realRoot, taskId, maxLines)
  }

  dispose(): void {
    this.disposed = true
    this.generation += 1
    this.stopTimer()
    this.listeners.clear()
  }

  // ── events ─────────────────────────────────────────────────────────

  /**
   * Apply one `protein_design.progress` event.
   *
   * Only `event_type === 'task'` speaks for the whole design. A phase, tool or
   * fold that reports `completed` has completed a phase, a tool or a fold — the
   * orchestrator emits those for every step — so it updates the activity line
   * and nothing else.
   */
  applyProgress(payload: ProteinProgressPayload): void {
    if (this.disposed || !isTaskId(payload.task_id)) {
      return
    }

    const entry = this.entry(payload.task_id)

    if (entry.seen.includes(payload.event_id)) {
      return
    }

    const at = Date.parse(payload.timestamp)

    // A strictly older event is late news about a state we have moved past.
    // Equal or unparseable times keep receipt order, which is all the wire
    // gives us.
    if (Number.isFinite(at) && entry.eventAt !== null && at < entry.eventAt) {
      return
    }

    entry.seen.push(payload.event_id)

    if (entry.seen.length > SEEN_EVENTS) {
      entry.seen.shift()
    }

    if (Number.isFinite(at)) {
      entry.eventAt = at
    }

    entry.eventRevision += 1
    entry.sawEvent = true
    // Every string on this event is foreign text on its way to a terminal, so
    // none of it reaches a record un-sanitised: an event phase is exactly as
    // able to carry an escape sequence as a summary is.
    entry.patch = {
      cycle: payload.cycle,
      totalCycles: payload.total_cycles,
      ...(payload.phase ? { phase: sanitizeLine(payload.phase, 60) } : {}),
      ...(payload.skill ? { skill: sanitizeLine(payload.skill, 80) } : {}),
      ...(payload.summary ? { activity: sanitizeLine(payload.summary) } : {}),
      ...(payload.error ? { error: sanitizeLine(payload.error) } : {})
    }

    let record = this.withPatch(entry.record, entry.patch)

    // The newest thing we know of, in both directions: an event carrying an
    // older timestamp than the snapshot we already read has not made our
    // knowledge older.
    record.observedAt = Math.max(record.observedAt, Number.isFinite(at) ? at : this.clock.nowMs())

    if (payload.event_type === 'task') {
      record = this.applyTaskEvent(entry, record, payload.status)
    }

    this.commit(entry, record)

    // A design the agent started is news. The bar used to wait to be asked
    // for — `/tasks show` — which meant a run nobody had typed a command about
    // progressed invisibly. Watching starts here instead, and the bar draws
    // nothing on its own once every task it knows about has finished.
    if (isActiveStatus(record.status)) {
      this.setVisible(true)
    }

    this.bump()
  }

  private applyTaskEvent(entry: Entry, record: TaskRecord, status: ProteinProgressPayload['status']): TaskRecord {
    if (status !== 'completed' && status !== 'failed') {
      // The design is alive again: a terminal claim nothing confirmed is over.
      return { ...record, provisional: undefined }
    }

    const next = { ...record, provisional: status }

    this.announce(entry, next, status, 'reported')

    return next
  }

  // ── the filesystem ─────────────────────────────────────────────────

  private async runRefresh(): Promise<void> {
    const generation = this.generation
    // Taken before the first await: everything an event patches from here on
    // is newer than whatever this pass is about to read off the disk, and goes
    // back on top of it.
    const revisions = new Map([...this.entries].map(([id, entry]) => [id, entry.eventRevision]))
    // Resolved once for the whole pass: every file read below is required to
    // land inside this directory once its own symlinks are followed.
    const realRoot = await resolveRoot(this.root)
    let ids: string[]

    try {
      ids = realRoot === null ? [] : await listTaskIds(realRoot)
      this.rootError = null
    } catch (err) {
      this.rootError = err instanceof Error ? err.message : String(err)
      this.bump()

      return
    }

    if (generation !== this.generation || this.disposed) {
      return
    }

    let changed = false

    await mapLimit(ids, READ_CONCURRENCY, async id => {
      if (realRoot === null) {
        return
      }

      const entry = this.entries.get(id)
      const revisionBefore = revisions.get(id) ?? 0
      const read = await readSnapshot(realRoot, id, {
        ...(entry?.record.snapshotMtimeMs === undefined ? {} : { mtimeMs: entry.record.snapshotMtimeMs }),
        ...(entry?.record.snapshotSize === undefined ? {} : { size: entry.record.snapshotSize })
      })

      // Hidden, paused, disposed or re-rooted while the read was in flight.
      if (generation !== this.generation || this.disposed) {
        return
      }

      if (read.kind === 'unchanged') {
        // The stamp matched, so this file is readable right now. A label left
        // by an earlier failed attempt is describing a state that has passed.
        changed = this.clearUnavailable(id) || changed

        return
      }

      if (read.kind === 'ok') {
        changed = this.applySnapshot(id, read.facts, read.mtimeMs, read.size, revisionBefore) || changed

        return
      }

      const reason = read.kind === 'gone' ? 'no snapshot.json in the task directory' : read.reason

      changed = this.markUnavailable(id, reason) || changed
    })

    const present = new Set(ids)

    for (const [id, entry] of this.entries) {
      if (present.has(id)) {
        continue
      }

      if (entry.sawEvent) {
        // Keep what the events said and label it: the files are not here.
        changed = this.markUnavailable(id, 'no task directory on this host') || changed
      } else {
        this.entries.delete(id)
        changed = true
      }
    }

    if (changed) {
      this.bump()
    }
  }

  private applySnapshot(
    taskId: string,
    facts: SnapshotFacts,
    mtimeMs: number,
    size: number,
    revisionBefore: number
  ): boolean {
    const entry = this.entry(taskId)
    let record: TaskRecord = {
      ...entry.record,
      bestObjective: facts.bestObjective,
      computeUrl: facts.computeUrl,
      computeWorkerId: facts.computeWorkerId,
      cycle: facts.cycle,
      error: facts.error,
      fromSnapshot: true,
      observedAt: Math.max(entry.record.observedAt, mtimeMs),
      phase: facts.phase,
      selectedSkill: facts.selectedSkill,
      snapshotMtimeMs: mtimeMs,
      snapshotSize: size,
      status: facts.status,
      statusSource: 'snapshot',
      target: facts.target,
      totalCycles: facts.totalCycles,
      unavailable: undefined
    }

    // An event landed while this read was in flight. Its phase/cycle patch is
    // newer than the file we just parsed, so it goes back on top.
    if (entry.eventRevision !== revisionBefore && entry.patch) {
      record = this.withPatch(record, entry.patch)
    }

    if (isTerminalStatus(facts.status)) {
      // The snapshot answers whatever an event had only reported.
      record.provisional = undefined
      this.announce(entry, record, facts.status, 'confirmed')
    }

    this.commit(entry, record)

    return true
  }

  /**
   * A task directory we can see but cannot read. The row appears anyway: a
   * task whose state is unreadable is not a task that does not exist.
   *
   * The stamp of the last good read goes with it. It describes a version of
   * the file this attempt could not confirm still exists, and keeping it is
   * how a snapshot that becomes readable again — a permission put back, a
   * directory remounted — answers "unchanged" forever and wears the old error
   * for the rest of the process.
   */
  private markUnavailable(taskId: string, reason: string): boolean {
    const entry = this.entry(taskId)
    const before = entry.record

    if (before.unavailable === reason && before.snapshotMtimeMs === undefined && before.snapshotSize === undefined) {
      return false
    }

    this.commit(entry, {
      ...before,
      snapshotMtimeMs: undefined,
      snapshotSize: undefined,
      unavailable: reason
    })

    return before.unavailable !== reason
  }

  /** The file read cleanly this time; whatever went wrong before is over. */
  private clearUnavailable(taskId: string): boolean {
    const entry = this.entries.get(taskId)

    if (!entry || entry.record.unavailable === undefined) {
      return false
    }

    this.commit(entry, { ...entry.record, unavailable: undefined })

    return true
  }

  // ── shared ─────────────────────────────────────────────────────────

  private entry(taskId: string): Entry {
    const existing = this.entries.get(taskId)

    if (existing) {
      return existing
    }

    const created: Entry = {
      announced: false,
      eventAt: null,
      eventRevision: 0,
      known: false,
      patch: null,
      record: blankRecord(taskId),
      sawEvent: false,
      seen: []
    }

    this.entries.set(taskId, created)

    return created
  }

  private withPatch(record: TaskRecord, patch: EventPatch): TaskRecord {
    // Snapshot-only fields (target, best objective, compute) are untouched: an
    // event carries none of them and absence is not a new value.
    return {
      ...record,
      cycle: patch.cycle ?? record.cycle,
      totalCycles: patch.totalCycles ?? record.totalCycles,
      ...(patch.phase ? { phase: patch.phase } : {}),
      ...(patch.skill ? { selectedSkill: patch.skill } : {}),
      ...(patch.activity ? { activity: patch.activity } : {}),
      ...(patch.error ? { error: patch.error } : {})
    }
  }

  /** Store the record with its status derived from what is actually known. */
  private commit(entry: Entry, record: TaskRecord): void {
    const resolved: TaskRecord = { ...record }

    if (!resolved.fromSnapshot) {
      // No file on this host. Events are evidence of work, not of a lifecycle:
      // they can say "running" and they can report a terminal claim, and both
      // are labelled as event-derived.
      resolved.status = resolved.provisional ?? (entry.sawEvent ? 'running' : 'unknown')
      resolved.statusSource = entry.sawEvent ? 'event' : 'none'
    }

    entry.record = compact(resolved)
    entry.known = true
  }

  /**
   * One transcript line per task, at most.
   *
   * A snapshot found already finished is history, not news: the file may be
   * days old, so discovering it is silent. A task-level event is live by
   * definition and is always worth the line. Either way the first signal wins,
   * which is what stops an ending producing one line for the event and another
   * for the poll that confirms it.
   */
  private announce(entry: Entry, record: TaskRecord, status: TaskStatus, kind: 'confirmed' | 'reported'): void {
    if (entry.announced) {
      return
    }

    entry.announced = true

    if (!this.onTerminal || (kind === 'confirmed' && !entry.known)) {
      return
    }

    const mark = status === 'completed' ? '✓' : status === 'failed' ? '✗' : '⏹'
    const word = kind === 'reported' ? `reported ${status}` : status
    const name = record.target ? `${sanitizeLine(record.target, 40)} ` : ''
    const id = shortTaskId(record.taskId, [...this.entries.keys()])
    const progress = record.cycle === null ? '' : ` · cycle ${formatProgress(record.cycle, record.totalCycles)}`

    this.onTerminal(`${mark} protein design ${name}(${id}) ${word}${progress}`, status === 'completed')
  }

  private stopTimer(): void {
    this.timer?.dispose()
    this.timer = undefined
  }

  /** Read, then arm the next read. Never the other way round: a slow disk must
   *  not be able to accumulate polls behind itself. */
  private async pollLoop(generation: number): Promise<void> {
    await this.refresh()

    if (generation !== this.generation || this.disposed || this.paused || !this.visibleFlag) {
      return
    }

    this.stopTimer()
    this.timer = new CountdownTimer(
      this.clock.nowMs() + this.refreshMs,
      this.clock,
      // Scheduling only: no countdown display and no render of its own.
      undefined,
      () => {},
      () => {
        this.timer = undefined
        void this.pollLoop(generation)
      }
    )
  }

  private bump(): void {
    this.revisionCount += 1

    for (const listener of this.listeners) {
      listener()
    }
  }
}
