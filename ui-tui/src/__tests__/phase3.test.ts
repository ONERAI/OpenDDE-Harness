// Phase 3 as the app wires it: the layout the components are mounted in, the
// fullscreen toggle over that same tree, and the session facts the header and
// footer grew.

import { stripTerminalSequences } from '@earendil-works/pi-tui'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { HarnessApp } from '../app.js'
import { ProteinDesignTaskBar } from '../components/proteinDesignTaskBar.js'
import { Gateway } from '../gateway.js'
import { RendererOwner } from '../renderer.js'
import { Theme } from '../theme.js'
import { FakeTerminal, FakeTransport } from './fakes.js'
import { ManualClock } from './selectorHarness.js'

const INFO = {
  cwd: '/work/proj',
  model: 'gpt-5',
  provider: 'openai',
  reasoning_effort: 'high',
  skills: { design: ['relax', 'dock'] },
  tools: { builtin: ['read', 'write'], mcp: ['fold'] },
  update_available: 'v0.5.0',
  update_command: 'ddeharness self-update'
}

const SCRIPT = {
  'commands.catalog': { categories: [], pairs: [['help', 'help']], skill_count: 2 },
  'config.get': { config: {} },
  'session.create': { info: INFO, session_id: 'tui:abc' },
  'setup.status': { provider_configured: true },
  'system.hello': { server_capabilities: [], server_version: '9.9.9', session: {} },
  'turn.send': { accepted: true, turn_id: 't1' }
}

let root = ''

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'opendde-phase3-'))
})

afterEach(async () => {
  await rm(root, { force: true, recursive: true })
})

function build(env: NodeJS.ProcessEnv = {}, mode: 'fullscreen' | 'regular' = 'regular', clock?: ManualClock) {
  const transport = new FakeTransport(SCRIPT)
  const terminal = new FakeTerminal(100, 30)
  const theme = new Theme('dark', 3)
  const renderer = new RendererOwner({ env, mode, terminal, theme })
  const exits: number[] = []
  const app = new HarnessApp({
    cwd: '/nonexistent',
    env: { OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT: root, ...env },
    gateway: new Gateway(transport),
    listenResize: () => () => {},
    onExit: code => exits.push(code),
    renderer,
    theme,
    tui: renderer.reference,
    ...(clock ? { clock } : {})
  })

  renderer.reference.start()

  const screen = () => stripTerminalSequences(terminal.output())

  return { app, exits, renderer, screen, terminal, transport }
}

async function snapshot(taskId: string, fields: Record<string, unknown>): Promise<void> {
  await mkdir(join(root, taskId), { recursive: true })
  await writeFile(join(root, taskId, 'snapshot.json'), JSON.stringify({ task_id: taskId, ...fields }), 'utf8')
}

const ALT_UP = '\x1b[1;3A'
const CTRL_C = '\x03'
const CTRL_D = '\x04'
const CTRL_U = '\x15'
const ENTER_ALT = '\x1b[?1049h'
const LEAVE_ALT = '\x1b[?1049l'

const screenOf = (terminal: FakeTerminal) => stripTerminalSequences(terminal.output())
const terminalBytes = (renderer: RendererOwner) => (renderer.renderer.terminal as FakeTerminal).output()
const USAGE_ZERO = { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }

/** Let a `setTimeout(0)` the app scheduled run. Queued after it, so it runs
 *  after it: same threshold, and the timer list is in order. */
const settleTimers = () => new Promise(resolve => setTimeout(resolve, 0))

/**
 * Wait for something observable rather than for a duration.
 *
 * `/tasks show` starts a poll it does not await, so a fixed sleep after it is
 * a guess about how long a directory read takes — and on a loaded machine it
 * is the wrong guess.
 */
async function waitFor(condition: () => boolean, what: string, timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs

  while (!condition()) {
    if (Date.now() > deadline) {
      throw new Error(`timed out waiting for ${what}`)
    }

    await new Promise(resolve => setTimeout(resolve, 2))
  }
}

describe('the layout', () => {
  it('mounts the task bar between the transcript and the queue', () => {
    const { renderer } = build()
    const children = renderer.reference.children

    // Session panel and transcript scroll together as one document; the bar,
    // the queue, the status row, the editor slot and the footer are the dock
    // below it.
    expect(children).toHaveLength(6)
    expect(children[1]).toBeInstanceOf(ProteinDesignTaskBar)

    renderer.dispose()
  })

  it('shows the design bar once /tasks show has found something', async () => {
    await snapshot('abcdef0123456789', { cycle: 2, status: 'running', target: 'VEGF binder', total_cycles: 6 })

    const { app, renderer, screen } = build()

    await app.boot()
    await app.submit('/tasks')

    await waitFor(() => {
      renderer.reference.renderNow()

      return screen().includes('Design tasks (1)')
    }, 'the design task bar')

    expect(screen()).toContain('VEGF binder')

    await app.submit('/tasks')
    renderer.reference.renderNow()

    renderer.dispose()
  })

  it('never sends the queued message that is open in the editor', async () => {
    const { app, renderer, screen, terminal, transport } = build()

    await app.boot()

    const sent = () =>
      transport.calls
        .filter(call => call.method === 'turn.send')
        .map(call => (call.params as { content: string }).content)
    const type = (data: string) => terminal.onInput?.(data)

    await app.submit('first')
    await waitFor(() => sent().length === 1, 'the first turn to start')

    // Queued behind the running turn, then pulled back with alt+up and
    // rewritten, through the real key path.
    await app.submit('queued original')
    type(ALT_UP)
    type(CTRL_U)
    type('queued edited')

    renderer.reference.renderNow()
    expect(screen()).toContain('Editing: queued original')

    // The turn ends. The drain must not take that message out from under the
    // edit: it would send the text being replaced and strand the rewrite with
    // nothing left to apply it to.
    transport.emit({ payload: { turn_id: 't1', usage: USAGE_ZERO }, type: 'message.complete' })
    await settleTimers()

    expect(sent()).toEqual(['first'])

    // Enter applies the rewrite, which releases the reservation, and the queue
    // drains straight away rather than waiting for the next turn to end.
    type('\r')
    await waitFor(() => sent().length === 2, 'the edited message to go out')

    expect(sent()).toEqual(['first', 'queued edited'])

    renderer.dispose()
  })
})

describe('/fullscreen', () => {
  it('reports, toggles and is idempotent', async () => {
    const { app, renderer, screen } = build()

    await app.boot()

    await app.submit('/fullscreen status')
    renderer.reference.renderNow()
    expect(screen()).toContain('fullscreen: off')

    await app.submit('/fullscreen')
    expect(renderer.mode).toBe('fullscreen')

    await app.submit('/fullscreen on')
    renderer.reference.renderNow()
    expect(screen()).toContain('fullscreen is already on')

    await app.submit('/fullscreen')
    expect(renderer.mode).toBe('regular')

    await app.submit('/fullscreen nope')
    renderer.reference.renderNow()
    expect(screen()).toContain('usage: /fullscreen')

    renderer.dispose()
  })

  it('keeps the same components, and the transcript, across a toggle', async () => {
    const { app, renderer } = build()

    await app.boot()

    const before = [...renderer.reference.children]

    await app.submit('/fullscreen on')
    expect(renderer.reference.children).toEqual(before)

    await app.submit('/fullscreen off')
    expect(renderer.reference.children).toEqual(before)

    renderer.dispose()
  })

  it('is the mode the process starts in', async () => {
    const { app, renderer, screen } = build({}, 'fullscreen')

    await app.boot()
    renderer.reference.renderNow()

    expect(renderer.mode).toBe('fullscreen')
    expect(terminalBytes(renderer)).toContain(ENTER_ALT)
    expect(screen()).toContain('OpenDDE')

    await app.submit('/fullscreen off')
    expect(renderer.mode).toBe('regular')

    await app.submit('/fullscreen on')
    expect(renderer.mode).toBe('fullscreen')

    renderer.dispose()
  })

  it.each(['/quit', CTRL_D, CTRL_C])('leaves the alternate screen and the conversation behind on %j', async key => {
    const { app, exits, renderer, terminal, transport } = build({}, 'fullscreen')

    await app.boot()
    await app.submit('remember this exchange')
    await waitFor(() => screenOf(terminal).includes('remember this exchange'), 'the prompt echo')

    // Idle before the key: Ctrl+C's first rung during a turn is a cancel, not
    // an exit, and this is about what leaving looks like.
    transport.emit({ payload: { turn_id: 't1', usage: USAGE_ZERO }, type: 'message.complete' })
    await settleTimers()

    const before = terminal.writes.length

    if (key.startsWith('/')) {
      await app.submit(key)
    } else {
      terminal.onInput?.(key)

      // Ctrl+C offers to exit before it exits, so it takes a second press.
      // Ctrl+D is the explicit exit key and goes in one, like pi's.
      if (key === CTRL_C) {
        expect(exits).toEqual([])
        terminal.onInput?.(key)
      }
    }

    await waitFor(() => exits.length === 1, `the exit after ${key}`)

    const written = terminal.writes.slice(before).join('')

    expect(exits).toEqual([0])
    expect(renderer.mode).toBe('regular')
    // The alternate buffer is left, and what was in it is written into the
    // shell's own so the conversation is still there afterwards.
    expect(written).toContain(LEAVE_ALT)
    expect(stripTerminalSequences(written)).toContain('remember this exchange')
  })
})

describe('shutting down', () => {
  it('stops the busy row and its clock when the app is quit mid-turn', async () => {
    const clock = new ManualClock()
    const { app, exits, renderer, transport } = build({}, 'regular', clock)

    await app.boot()
    expect(clock.ticks).toHaveLength(0)

    await app.submit('something that takes a while')
    await waitFor(() => transport.methods.includes('turn.send'), 'the turn to start')

    // The spinner's animation and the row's once-a-second redraw have separate
    // owners; this is the row's.
    expect(clock.ticks).toHaveLength(1)

    await app.submit('/quit')

    expect(exits).toEqual([0])
    // Nothing reports the turn idle on the way out — `turn.detach()` does not —
    // so quitting has to let go of the row itself.
    expect(clock.ticks).toHaveLength(0)

    renderer.dispose()
  })

  it('leaves the row alone when the turn ends normally', async () => {
    const clock = new ManualClock()
    const { app, renderer, transport } = build({}, 'regular', clock)

    await app.boot()
    await app.submit('something short')
    await waitFor(() => transport.methods.includes('turn.send'), 'the turn to start')
    expect(clock.ticks).toHaveLength(1)

    transport.emit({ payload: { turn_id: 't1', usage: USAGE_ZERO }, type: 'message.complete' })
    await settleTimers()

    expect(clock.ticks).toHaveLength(0)

    renderer.dispose()
  })
})

describe('what the session serves', () => {
  it('shows the real tool and skill counts and the update notice', async () => {
    const { app, renderer, screen } = build()

    await app.boot()
    renderer.reference.renderNow()

    const text = screen()

    expect(text).toContain('3 tools')
    expect(text).toContain('2 skills')
    expect(text).toContain('v0.5.0')
    expect(text).toContain('ddeharness self-update')
    // And once in the transcript, in a sentence, not only as the status bar's glyph.
    expect(text).toContain('OpenDDE Harness v0.5.0 is on PyPI')
    expect(text).toContain('Update with: ddeharness self-update')

    renderer.dispose()
  })
})
