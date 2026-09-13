// The design-task bar and the two commands that read the same monitor.

import { stripTerminalSequences } from '@earendil-works/pi-tui'
import { execFileSync } from 'node:child_process'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { ProteinProgressPayload } from '../tasks/monitor.js'

import { dispatch } from '../commands/dispatch.js'
import { CommandRegistry } from '../commands/registry.js'
import { ProteinDesignTaskBar } from '../components/proteinDesignTaskBar.js'
import { ProteinDesignTasks } from '../tasks/monitor.js'
import { Theme } from '../theme.js'
import { createTestContext } from './commandContext.js'
import { ManualClock } from './selectorHarness.js'

const registry = new CommandRegistry()
const theme = new Theme('dark', 0)

let root = ''
const monitors: ProteinDesignTasks[] = []

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'opendde-taskbar-'))
})

afterEach(async () => {
  for (const monitor of monitors.splice(0)) {
    monitor.dispose()
  }

  await rm(root, { force: true, recursive: true })
})

async function snapshot(taskId: string, fields: Record<string, unknown>): Promise<void> {
  await mkdir(join(root, taskId), { recursive: true })
  await writeFile(join(root, taskId, 'snapshot.json'), JSON.stringify({ task_id: taskId, ...fields }), 'utf8')
}

function monitor(): ProteinDesignTasks {
  const made = new ProteinDesignTasks({ clock: new ManualClock(), root })

  monitors.push(made)

  return made
}

function draw(bar: ProteinDesignTaskBar, width: number): string[] {
  return bar.render(width).map(line => stripTerminalSequences(line))
}

let counter = 0

function event(overrides: Partial<ProteinProgressPayload> = {}): ProteinProgressPayload {
  counter += 1

  return {
    actor: 'orchestrator',
    candidate_count: null,
    cycle: 2,
    duration_ms: null,
    error: null,
    event_id: `bar-${counter}`,
    event_type: 'task',
    has_details: false,
    phase: 'design_cycle',
    skill: null,
    status: 'progress',
    summary: 'folding',
    task_id: 'abcdef0123456789',
    timestamp: '2026-09-11T00:00:00.000Z',
    tool: null,
    total_cycles: 6,
    ...overrides
  }
}

describe('the design task bar', () => {
  it('draws nothing until it is shown', async () => {
    await snapshot('one', { cycle: 1, status: 'running', target: 'VEGF binder', total_cycles: 4 })

    const tasks = monitor()
    const bar = new ProteinDesignTaskBar(theme, tasks)

    await tasks.refresh()
    expect(draw(bar, 80)).toEqual([])
  })

  it('shows the running work, three rows at most, and counts the rest', async () => {
    for (const index of [1, 2, 3, 4, 5]) {
      await snapshot(`task-${index}`, {
        cycle: index,
        phase: 'design_cycle',
        status: 'running',
        target: `target ${index}`,
        total_cycles: 6
      })
    }

    const tasks = monitor()
    const bar = new ProteinDesignTaskBar(theme, tasks)

    tasks.setVisible(true)
    await tasks.refresh()

    const lines = draw(bar, 80)

    expect(lines[0]).toContain('Design tasks (5) · /tasks for detail')
    expect(lines.filter(line => line.startsWith('target '))).toHaveLength(3)
    expect(lines.some(line => line.includes('+2 more · /tasks'))).toBe(true)
  })

  it('puts the followed task first', async () => {
    await snapshot('aaaa1111', { cycle: 1, status: 'running', target: 'first seen', total_cycles: 4 })
    await snapshot('bbbb2222', { cycle: 1, status: 'running', target: 'followed', total_cycles: 4 })

    const tasks = monitor()
    const bar = new ProteinDesignTaskBar(theme, tasks)

    tasks.setVisible(true)
    await tasks.refresh()
    tasks.focus('bbbb2222')

    expect(draw(bar, 80)[1]).toContain('followed')
  })

  it('empties as the work finishes, and keeps the finished tasks in the list', async () => {
    await snapshot('done', { cycle: 4, status: 'running', target: 'ending', total_cycles: 4 })

    const tasks = monitor()
    const bar = new ProteinDesignTaskBar(theme, tasks)

    tasks.setVisible(true)
    await tasks.refresh()
    expect(draw(bar, 80)).not.toEqual([])

    await snapshot('done', { cycle: 4, status: 'completed', target: 'ending', total_cycles: 4 })
    await tasks.refresh()

    expect(draw(bar, 80)).toEqual([])
    expect(tasks.list()).toHaveLength(1)
  })

  it('writes no line wider than the terminal, at any width', async () => {
    await snapshot('wide', {
      cycle: 3,
      phase: 'a_very_long_phase_name_indeed',
      status: 'running',
      target: 'a target with a name far longer than any narrow terminal could show',
      total_cycles: 12
    })

    const tasks = monitor()
    const bar = new ProteinDesignTaskBar(theme, tasks)

    tasks.setVisible(true)
    await tasks.refresh()

    for (const width of [1, 20, 80, 120]) {
      bar.invalidate()

      for (const line of draw(bar, width)) {
        expect(line.length).toBeLessThanOrEqual(width)
      }
    }
  })

  it('paints no escape sequence a snapshot asked it to', async () => {
    await snapshot('controls', {
      cycle: 1,
      phase: '\u001b[2Jcleared',
      status: 'running',
      target: '\u001b[31mred target',
      total_cycles: 4
    })

    const tasks = monitor()
    const bar = new ProteinDesignTaskBar(theme, tasks)

    tasks.setVisible(true)
    await tasks.refresh()

    // Display-width truncation preserves escape sequences; it is not a
    // sanitiser. The phase has to be cleaned before it is formatted.
    const raw = bar.render(80).join('\n')

    // The escape character is gone; what is left is inert printable text.
    expect(raw).not.toContain('\u001b[2J')
    expect(raw).not.toContain('\u001b[31m')
    expect(draw(bar, 80).join('\n')).toContain('2Jcleared')
    expect(draw(bar, 80).join('\n')).toContain('31mred target')
  })

  it('paints no escape sequence a progress event asked it to', () => {
    const tasks = monitor()
    const bar = new ProteinDesignTaskBar(theme, tasks)

    tasks.setVisible(true)
    tasks.applyProgress(event({ phase: '\u001b[2Jwiped', summary: '\u001b[2Jwiped' }))

    expect(bar.render(80).join('\n')).not.toContain('\u001b[2J')
  })

  it('shows `?` rather than inventing a denominator', () => {
    const tasks = monitor()
    const bar = new ProteinDesignTaskBar(theme, tasks)

    tasks.setVisible(true)
    tasks.applyProgress(event({ cycle: 2, total_cycles: null }))

    expect(draw(bar, 80)[1]).toContain('2/?')
  })
})

describe('the task commands', () => {
  it('lists every state, and says where it looked when there is nothing', async () => {
    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/tasks')
    expect(harness.text()).toContain(root)

    await snapshot('aaaa1111', { best_candidate: { objective: 0 }, cycle: 2, status: 'running', target: 'live' })
    await snapshot('bbbb2222', { status: 'completed', target: 'done' })

    await dispatch(registry, harness.context, '/tasks')

    const printed = harness.text()

    expect(printed).toContain('running')
    expect(printed).toContain('completed')
    expect(printed).toContain('best 0.0000')
  })

  it('starts watching when it lists, without touching the gateway', async () => {
    const harness = createTestContext({}, { taskRoot: root })

    // Nothing running, nothing to watch: the bar was a command to switch on
    // and off, and this is what replaced it.
    await dispatch(registry, harness.context, '/tasks')
    expect(harness.tasks.visible).toBe(false)

    await snapshot('cccc3333', { cycle: 1, status: 'running', target: 'live one' })
    await dispatch(registry, harness.context, '/tasks')

    expect(harness.tasks.visible).toBe(true)
    expect(harness.transport.calls).toEqual([])
  })

  it('asks for the id the namespaced actions need, without reading anything', async () => {
    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/task:logs')
    await dispatch(registry, harness.context, '/task:follow')

    expect(harness.printed).toHaveLength(2)
    expect(harness.text()).toContain('usage: /task:logs <id>')
    expect(harness.text()).toContain('usage: /task:follow <id>')
  })

  it('prints the full id and the compute binding in the detail view', async () => {
    await snapshot('abcdef0123456789', {
      compute_url: 'http://10.0.0.4:8080',
      compute_worker_id: 'worker-3',
      cycle: 2,
      error: 'gate rejected candidate 4',
      phase: 'design_cycle',
      selected_skill: 'relax',
      status: 'failed',
      target: 'VEGF binder',
      total_cycles: 6
    })

    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/tasks abcdef')

    const printed = harness.text()

    expect(printed).toContain('id: abcdef0123456789')
    expect(printed).toContain('state: failed (snapshot)')
    expect(printed).toContain('progress: 2/6')
    expect(printed).toContain('worker-3 · http://10.0.0.4:8080')
    expect(printed).toContain('gate rejected candidate 4')
  })

  it('says the last observation is unknown rather than printing 1970', async () => {
    // A directory whose snapshot has never parsed has been found but never
    // read, so there is no time to show. The record carries zero for that, and
    // zero is a real date.
    await mkdir(join(root, 'abcdef0123456789'), { recursive: true })
    await writeFile(join(root, 'abcdef0123456789', 'snapshot.json'), 'not json', 'utf8')

    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/tasks abcdef')

    const printed = harness.text()

    expect(printed).toContain('last observed: unknown')
    expect(printed).not.toContain('1970')
  })

  it('asks for a longer prefix when one would match two tasks', async () => {
    await snapshot('aaaa1111', { status: 'running' })
    await snapshot('aaaa2222', { status: 'running' })

    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/tasks aaaa')
    expect(harness.printed[0]?.title).toContain('use a longer prefix')
  })

  it('follow focuses the task and shows the bar', async () => {
    await snapshot('abcdef0123456789', { cycle: 1, status: 'running', target: 'VEGF', total_cycles: 4 })

    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/task:follow abcdef')

    expect(harness.tasks.visible).toBe(true)
    expect(harness.tasks.focusedTaskId).toBe('abcdef0123456789')
    expect(harness.text()).toContain('following VEGF')
  })

  it('logs tail the worker log and report an empty or missing one', async () => {
    await snapshot('abcdef0123456789', { status: 'running', target: 'VEGF' })

    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/task:logs abcdef')
    expect(harness.text()).toContain('no worker log')

    await writeFile(join(root, 'abcdef0123456789', 'worker.log'), 'starting\nfolding\n', 'utf8')
    await dispatch(registry, harness.context, '/task:logs abcdef')

    expect(harness.text()).toContain('folding')
    expect(harness.printed.at(-1)?.title).toContain('worker log')
  })

  it('strips terminal control sequences out of the phase in the list and the details', async () => {
    await snapshot('controls0', {
      cycle: 1,
      phase: '\u001b[2Jcleared',
      status: 'running',
      target: '\u001b[31mred',
      total_cycles: 4
    })

    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/tasks')
    await dispatch(registry, harness.context, '/tasks controls0')

    const printed = harness.printed.map(line => `${line.title ?? ''}\n${line.text}`).join('\n')

    expect(printed).not.toContain('\u001b[2J')
    expect(printed).not.toContain('\u001b[31m')
    expect(printed).toContain('2Jcleared')
  })

  it('calls a log window with no complete line a read limit, not an empty file', async () => {
    await snapshot('bigline', { status: 'running', target: 'noisy' })
    await writeFile(join(root, 'bigline', 'worker.log'), 'y'.repeat(1024 * 1024 + 1), 'utf8')

    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/task:logs bigline')

    expect(harness.text()).toContain('no complete line')
    expect(harness.text()).not.toContain('is empty')
  })

  it('still calls a zero-byte log empty', async () => {
    await snapshot('quietlog', { status: 'running', target: 'quiet' })
    await writeFile(join(root, 'quietlog', 'worker.log'), '', 'utf8')

    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/task:logs quietlog')

    expect(harness.text()).toContain('is empty')
  })

  it('says why a log could not be read when it was refused', async () => {
    await snapshot('fifolog', { status: 'running', target: 'fifo' })
    execFileSync('mkfifo', [join(root, 'fifolog', 'worker.log')])

    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/task:logs fifolog')

    expect(harness.text()).toContain('not a regular file')
  })

  it('strips terminal control sequences out of worker output', async () => {
    await snapshot('abcdef0123456789', { status: 'running', target: 'VEGF' })
    await writeFile(join(root, 'abcdef0123456789', 'worker.log'), 'plain [2Jwiped\n', 'utf8')

    const harness = createTestContext({}, { taskRoot: root })

    await dispatch(registry, harness.context, '/task:logs abcdef')

    expect(harness.text()).not.toContain('\u001b')
    expect(harness.text()).toContain('plain [2Jwiped')
  })
})
