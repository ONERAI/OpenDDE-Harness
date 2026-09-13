// What the TUI knows about one detached protein-design task.
//
// A task has two sources and they disagree by design. The gateway broadcasts
// `protein_design.progress` events, which are task-global background news with
// no session attribution and no ordering revision; the worker writes
// `<root>/<task_id>/snapshot.json`, which is the persisted authority but is
// only ever as fresh as the last write. This module holds the record both feed
// into, the rules for reading a snapshot honestly, and nothing that touches a
// filesystem or a clock.

/** The plugin's own task-ID rule (`plugin/protein_design/core/detached.py`).
 *  Applied to events, directory entries and command lookups alike, so nothing
 *  derived from a prefix can ever name a path outside the task root. */
const TASK_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/

export function isTaskId(value: unknown): value is string {
  return typeof value === 'string' && TASK_ID.test(value)
}

export type TaskStatus = 'completed' | 'failed' | 'queued' | 'running' | 'stopped' | 'unknown'

const STATUSES = new Set<string>(['completed', 'failed', 'queued', 'running', 'stopped'])

/** The snapshot's own word for the state, or `unknown`. Anything unrecognised
 *  stays unknown: the old store guessed `running`, which turned a corrupt file
 *  into a live worker. */
export function parseStatus(value: unknown): TaskStatus {
  const text = typeof value === 'string' ? value.trim().toLowerCase() : ''

  return STATUSES.has(text) ? (text as TaskStatus) : 'unknown'
}

export function isTerminalStatus(status: TaskStatus): boolean {
  return status === 'completed' || status === 'failed' || status === 'stopped'
}

export function isActiveStatus(status: TaskStatus): boolean {
  return status === 'queued' || status === 'running'
}

/** Where a record's status came from, so the UI can say so rather than imply a
 *  liveness it never observed. */
export type StatusSource = 'event' | 'none' | 'snapshot'

export interface TaskRecord {
  /** The last event's one-line summary. Not progress the UI measured. */
  activity?: string
  bestObjective?: number
  computeUrl?: string
  computeWorkerId?: string
  cycle: null | number
  error?: string
  /** A snapshot has been read for this task at least once. */
  fromSnapshot: boolean
  /** Epoch ms of the newest thing we learned: the snapshot's mtime, or the
   *  clock when an event arrived. Observation metadata, not an ordering token. */
  observedAt: number
  phase?: string
  /** A task-level event reported this terminal state and no snapshot confirms
   *  it yet. Shown as a disagreement, never as the lifecycle. */
  provisional?: 'completed' | 'failed'
  selectedSkill?: string
  snapshotMtimeMs?: number
  snapshotSize?: number
  status: TaskStatus
  statusSource: StatusSource
  target?: string
  taskId: string
  totalCycles: null | number
  /** Why the snapshot could not be read on the last attempt. The row stays;
   *  one bad file never hides the healthy tasks beside it. */
  unavailable?: string
}

/** The fields a `snapshot.json` contributes. Missing numbers stay missing:
 *  turning them into zero is how "no best objective yet" became "0.0000". */
export interface SnapshotFacts {
  bestObjective?: number
  computeUrl?: string
  computeWorkerId?: string
  cycle: null | number
  error?: string
  phase?: string
  selectedSkill?: string
  status: TaskStatus
  target?: string
  totalCycles: null | number
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** A string field from a snapshot, cleaned on the way in. These are foreign
 *  text: a `phase` of `\u001b[2J` is a valid JSON string and a screen-clearing
 *  escape sequence, and display-width truncation is not a sanitiser. */
function text(value: unknown, maxChars = 200): string | undefined {
  if (typeof value !== 'string' || !value.trim()) {
    return undefined
  }

  return sanitizeLine(value, maxChars) || undefined
}

/** A snapshot's error text. Keeps its line breaks — the detail view wraps it —
 *  but loses every control sequence. */
function sanitizedError(value: unknown): string | undefined {
  if (typeof value !== 'string' || !value.trim()) {
    return undefined
  }

  return sanitizeText(value, 1000) || undefined
}

/** Finite numbers only. Zero is a value, not a missing one. */
function finite(value: unknown): null | number {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

/**
 * Read a parsed `snapshot.json`. Returns null when it is not an object or when
 * its `task_id` disagrees with the directory it was found in — a file naming a
 * different task is not this task's state, however well-formed it is.
 */
export function parseSnapshot(value: unknown, taskId: string): null | SnapshotFacts {
  if (!isRecord(value) || value.task_id !== taskId) {
    return null
  }

  const best = isRecord(value.best_candidate) ? finite(value.best_candidate.objective) : null
  const computeUrl = text(value.compute_url)
  const computeWorkerId = text(value.compute_worker_id)
  // The error keeps its line breaks; the detail view wraps it.
  const error = sanitizedError(value.error)
  const phase = text(value.phase, 60)
  const selectedSkill = text(value.selected_skill, 80)
  const target = text(value.target)

  return {
    cycle: finite(value.cycle),
    status: parseStatus(value.status),
    totalCycles: finite(value.total_cycles),
    ...(best === null ? {} : { bestObjective: best }),
    ...(computeUrl ? { computeUrl } : {}),
    ...(computeWorkerId ? { computeWorkerId } : {}),
    ...(error ? { error } : {}),
    ...(phase ? { phase } : {}),
    ...(selectedSkill ? { selectedSkill } : {}),
    ...(target ? { target } : {})
  }
}

// C0 controls and DEL, minus tab and newline. Worker logs and snapshot error
// strings are foreign text on their way into a terminal: an escape sequence in
// one of them would move the cursor or repaint the screen.
// eslint-disable-next-line no-control-regex
const CONTROL = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g

export function sanitizeText(value: string, maxChars = 4000): string {
  const clean = value.replace(CONTROL, '')

  return clean.length > maxChars ? `${clean.slice(0, maxChars)}…` : clean
}

/** One line of foreign text: no controls, no newlines, bounded. */
export function sanitizeLine(value: string, maxChars = 200): string {
  return sanitizeText(value.replace(/\s+/g, ' ').trim(), maxChars)
}

/**
 * The shortest prefix of `taskId` that is at least `min` characters and names
 * no other task in `all`. Lists show this; details always show the full ID.
 */
export function shortTaskId(taskId: string, all: readonly string[], min = 8): string {
  for (let length = Math.min(min, taskId.length); length < taskId.length; length += 1) {
    const prefix = taskId.slice(0, length)

    if (!all.some(other => other !== taskId && other.startsWith(prefix))) {
      return prefix
    }
  }

  return taskId
}

/** `3/10`, `3/?`, `?/10`, `?`. No percentage: a missing denominator has none. */
export function formatProgress(cycle: null | number, total: null | number): string {
  if (cycle === null && total === null) {
    return '?'
  }

  return `${cycle ?? '?'}/${total ?? '?'}`
}

/** `Setup`, `Design Cycle`.
 *
 *  Sanitised here as well as at the point the value entered the record. This
 *  is the function every surface calls — the bar, the list and the detail
 *  block — so it is the one place that can promise the phase text reaching a
 *  terminal carries no escape sequences. */
export function formatPhase(phase: string | undefined): string {
  const raw = sanitizeLine(phase ?? '', 60)
    .replace(/[_-]+/g, ' ')
    .trim()

  return raw ? raw.replace(/\b\w/g, char => char.toUpperCase()) : 'Unknown'
}

/** The status word plus, when they disagree, what an event reported. */
export function formatState(record: TaskRecord): string {
  const source =
    record.statusSource === 'snapshot' ? 'snapshot' : record.statusSource === 'event' ? 'progress events' : 'no source'
  const provisional = record.provisional ? `; reported ${record.provisional}, awaiting snapshot` : ''

  return `${record.status} (${source}${provisional})`
}
