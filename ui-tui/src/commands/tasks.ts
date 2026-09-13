// `/tasks` and `/task`: reading the local protein-design task root.
//
// These are local-file commands. They do not go through `slash.exec`, there is
// no `tasks.list` RPC to call, and `/task logs` reads the worker log here
// rather than shelling out to `tail`. Nothing they do asks a worker to stop, or
// starts one.

import type { TaskRecord } from '../tasks/types.js'
import type { Command, CommandContext } from './types.js'

import { MAX_LOG_BYTES } from '../tasks/files.js'
import {
  formatPhase,
  formatProgress,
  formatState,
  isActiveStatus,
  sanitizeLine,
  sanitizeText,
  shortTaskId
} from '../tasks/types.js'

/** Rows `/tasks` prints. */
const LIST_LIMIT = 20

/** Lines `/task logs` prints. */
const LOG_LINES = 200

/** Every task command needs the same thing: which task. */
function usage(command: string): string {
  return `usage: /${command} <id>`
}

/** `2026-09-11 12:30:02` in local time, or nothing we can vouch for.
 *
 *  Zero is the record's "nothing observed yet", which a task whose snapshot has
 *  never been readable keeps. Printing 1970 for it would be a worse lie than
 *  admitting we have no time to show. */
function formatTime(ms: number): string {
  const at = new Date(ms)

  if (!Number.isFinite(ms) || ms <= 0 || Number.isNaN(at.getTime())) {
    return 'unknown'
  }

  const pad = (value: number) => String(value).padStart(2, '0')

  return (
    `${at.getFullYear()}-${pad(at.getMonth() + 1)}-${pad(at.getDate())} ` +
    `${pad(at.getHours())}:${pad(at.getMinutes())}:${pad(at.getSeconds())}`
  )
}

function formatObjective(value: number | undefined): string {
  return value === undefined ? '—' : value.toFixed(4)
}

/** One row of `/tasks`. */
function listRow(task: TaskRecord, ids: readonly string[]): string {
  const target = sanitizeLine(task.target ?? 'protein design', 24).padEnd(24)
  const progress = formatProgress(task.cycle, task.totalCycles).padEnd(7)
  const phase = formatPhase(task.phase).padEnd(16)

  return `${task.status.padEnd(9)} ${target} ${progress} ${phase} best ${formatObjective(task.bestObjective).padEnd(9)} ${shortTaskId(task.taskId, ids)}`
}

function detail(task: TaskRecord): string {
  const compute = [task.computeWorkerId, task.computeUrl].filter(Boolean).join(' · ')
  const lines = [
    `target: ${sanitizeLine(task.target ?? 'protein design', 80)}`,
    `id: ${task.taskId}`,
    `state: ${formatState(task)}`,
    `progress: ${formatProgress(task.cycle, task.totalCycles)}`,
    `phase: ${formatPhase(task.phase)}`,
    `skill: ${task.selectedSkill ? sanitizeLine(task.selectedSkill, 60) : '—'}`,
    `best objective: ${formatObjective(task.bestObjective)}`,
    `compute: ${compute ? sanitizeLine(compute, 120) : '—'}`,
    `last observed: ${formatTime(task.observedAt)}`
  ]

  if (task.activity) {
    lines.push(`last event: ${task.activity}`)
  }

  if (task.unavailable) {
    lines.push(`snapshot: ${sanitizeLine(task.unavailable, 160)}`)
  }

  if (task.error) {
    lines.push(`error: ${sanitizeText(task.error, 1000)}`)
  }

  return lines.join('\n')
}

/** Resolve a query, saying why it failed. Null means it was reported. */
function resolveOrReport(ctx: CommandContext, query: string): null | TaskRecord {
  const found = ctx.tasks.resolve(query)

  if (found.kind === 'ok') {
    return found.task
  }

  if (found.kind === 'ambiguous') {
    const ids = found.ids.slice(0, 10).join('\n')

    ctx.transcript.printBlock(ids, `${found.ids.length} tasks start with ${query} — use a longer prefix`)

    return null
  }

  ctx.transcript.print(`no protein-design task matches ${query}`, 'warn')

  return null
}

/** `/tasks` with no argument: one refresh, then the most recent tasks. */
async function listTasks(ctx: CommandContext): Promise<void> {
  const epoch = ctx.tasks.sessionEpoch()

  await ctx.tasks.refresh()

  if (epoch !== ctx.tasks.sessionEpoch()) {
    return
  }

  const failure = ctx.tasks.rootFailure()

  if (failure) {
    ctx.transcript.print(`the task root could not be read: ${sanitizeLine(failure, 160)}`, 'warn')

    return
  }

  const tasks = ctx.tasks.list()

  if (tasks.length === 0) {
    ctx.transcript.print(`no protein-design tasks under ${ctx.tasks.rootPath()}`)

    return
  }

  // Asking what is running is what starts the watch: the bar used to wait for
  // `/tasks show`, which is a switch to remember about a progress bar. It
  // draws nothing while nothing is running, so there is no state to turn off.
  if (tasks.some(task => isActiveStatus(task.status))) {
    ctx.tasks.setVisible(true)
  }

  const ids = tasks.map(task => task.taskId)
  const shown = tasks.slice(0, LIST_LIMIT)
  const rows = shown.map(task => listRow(task, ids))

  if (tasks.length > shown.length) {
    rows.push(`… ${tasks.length - shown.length} more`)
  }

  ctx.transcript.printBlock(rows.join('\n'), 'Protein design tasks')
}

async function showTask(ctx: CommandContext, query: string): Promise<void> {
  const epoch = ctx.tasks.sessionEpoch()

  await ctx.tasks.refresh()

  if (epoch !== ctx.tasks.sessionEpoch()) {
    return
  }

  const task = resolveOrReport(ctx, query)

  if (task) {
    ctx.transcript.printBlock(detail(task), `${sanitizeLine(task.target ?? 'protein design', 40)} · ${task.taskId}`)
  }
}

/** `/task follow`: task-bar priority, not a filesystem tail and not a turn. */
async function followTask(ctx: CommandContext, query: string): Promise<void> {
  const epoch = ctx.tasks.sessionEpoch()

  await ctx.tasks.refresh()

  if (epoch !== ctx.tasks.sessionEpoch()) {
    return
  }

  const task = resolveOrReport(ctx, query)

  if (!task) {
    return
  }

  ctx.tasks.focus(task.taskId)
  ctx.tasks.setVisible(true)

  const id = shortTaskId(
    task.taskId,
    ctx.tasks.list().map(row => row.taskId)
  )

  ctx.transcript.print(`following ${sanitizeLine(task.target ?? 'protein design', 40)} · ${id}`)

  // A task that is already over has no bar row to give, so say what it ended
  // as rather than leave the acknowledgement looking like nothing happened.
  if (task.status !== 'queued' && task.status !== 'running') {
    ctx.transcript.printBlock(detail(task), `${sanitizeLine(task.target ?? 'protein design', 40)} · ${task.taskId}`)
  }
}

async function taskLogs(ctx: CommandContext, query: string): Promise<void> {
  const epoch = ctx.tasks.sessionEpoch()

  await ctx.tasks.refresh()

  if (epoch !== ctx.tasks.sessionEpoch()) {
    return
  }

  const task = resolveOrReport(ctx, query)

  if (!task) {
    return
  }

  const tail = await ctx.tasks.readLogTail(task.taskId, LOG_LINES)

  if (epoch !== ctx.tasks.sessionEpoch()) {
    return
  }

  const title = `${sanitizeLine(task.target ?? 'protein design', 40)} · ${task.taskId} · worker log`

  if (tail.missing) {
    const why = tail.reason ? `: ${sanitizeLine(tail.reason, 160)}` : ' on this host'

    ctx.transcript.print(`no worker log for ${task.taskId}${why}`, 'warn')

    return
  }

  if (tail.bytes === 0) {
    ctx.transcript.print(`the worker log for ${task.taskId} is empty`)

    return
  }

  // A log of one enormous line — a stack trace, a JSON blob — leaves the read
  // window with no complete line in it. That is a read limit, not an empty
  // file, and saying "empty" about a log that is actively being written is the
  // worse of the two lies.
  if (tail.lines.length === 0) {
    ctx.transcript.print(
      `the last ${MAX_LOG_BYTES} bytes of the worker log for ${task.taskId} hold no complete line`,
      'warn'
    )

    return
  }

  const body = tail.lines.map(line => sanitizeLine(line, 500)).join('\n')
  const notice = tail.truncated ? `\n… earlier lines are past the ${MAX_LOG_BYTES}-byte read limit` : ''

  ctx.transcript.printBlock(body + notice, title)
}

export function taskCommands(): Command[] {
  return [
    {
      argumentHint: '[id | hide | show]',
      description: 'protein-design tasks: all of them, one in detail, or hide/show the bar',
      name: 'tasks',
      run: async (ctx, arg) => {
        const query = arg.trim()

        // The bar appears while tasks run and goes when they are done. `hide`
        // is for the reader who does not want it above the composer while a
        // long run is on; `show` brings it back.
        if (query === 'hide') {
          ctx.tasks.setVisible(false)
          ctx.transcript.print('design task bar hidden; /tasks show brings it back')

          return
        }

        if (query === 'show') {
          ctx.tasks.setVisible(true)

          return listTasks(ctx)
        }

        return query ? showTask(ctx, query) : listTasks(ctx)
      }
    },

    {
      argumentHint: '<id>',
      description: "read one task's worker log",
      name: 'task:logs',
      run: async (ctx, arg) => {
        const query = arg.trim()

        return query ? taskLogs(ctx, query) : ctx.transcript.print(usage('task:logs'), 'warn')
      }
    },

    {
      argumentHint: '<id>',
      description: 'follow one task until it finishes',
      name: 'task:follow',
      run: async (ctx, arg) => {
        const query = arg.trim()

        return query ? followTask(ctx, query) : ctx.transcript.print(usage('task:follow'), 'warn')
      }
    }
  ]
}
