// The design-task bar: a few quiet lines between the transcript and the queue
// saying what the detached protein-design workers are doing.
//
// It shows only work that is still going — queued or running — so it empties
// itself as tasks finish; the finished ones stay in `/tasks` and `/task`. There
// is no percentage, because a task with no total cycle count has no denominator
// to make one out of.

import type { Component } from '@earendil-works/pi-tui'

import { truncateToWidth } from '@earendil-works/pi-tui'

import type { ProteinDesignTasks } from '../tasks/monitor.js'
import type { TaskRecord } from '../tasks/types.js'
import type { Theme } from '../theme.js'

import { formatPhase, formatProgress, isActiveStatus, sanitizeLine } from '../tasks/types.js'

/** Rows drawn before the bar collapses into a "+N more". */
export const MAX_ROWS = 3

export class ProteinDesignTaskBar implements Component {
  private cachedKey = ''
  private cachedLines: string[] = []

  constructor(
    private readonly theme: Theme,
    private readonly tasks: ProteinDesignTasks
  ) {}

  invalidate(): void {
    this.cachedKey = ''
    this.cachedLines = []
  }

  render(width: number): string[] {
    // The palette joins the revision and the width in the key: see
    // `UserMessage.render`.
    const key = `${width}:${this.tasks.revision}:${this.theme.scheme}`

    if (key === this.cachedKey) {
      return this.cachedLines
    }

    this.cachedKey = key
    this.cachedLines = this.build(Math.max(1, width))

    return this.cachedLines
  }

  /** Focused task first, then most recently observed. */
  private active(): TaskRecord[] {
    const focused = this.tasks.focusedTaskId
    const rows = this.tasks.list().filter(task => isActiveStatus(task.status))

    return focused ? rows.sort((a, b) => Number(b.taskId === focused) - Number(a.taskId === focused)) : rows
  }

  private build(width: number): string[] {
    if (!this.tasks.visible) {
      return []
    }

    const rows = this.active()

    if (rows.length === 0) {
      return []
    }

    const t = this.theme
    const lines = [t.fg('muted', `Design tasks (${rows.length}) · /tasks for detail`)]

    for (const task of rows.slice(0, MAX_ROWS)) {
      const tone = task.status === 'queued' ? 'muted' : 'text'
      const suffix = `${formatProgress(task.cycle, task.totalCycles)} · ${formatPhase(task.phase)}`
      const target = sanitizeLine(task.target ?? 'protein design', 48)

      lines.push(t.fg(tone, `${target} · ${suffix}`))
    }

    if (rows.length > MAX_ROWS) {
      lines.push(t.fg('muted', `+${rows.length - MAX_ROWS} more · /tasks`))
    }

    // One blank line under the bar, the way the session panel spaces itself
    // off the transcript.
    return [...lines.map(line => truncateToWidth(line, width)), '']
  }
}
