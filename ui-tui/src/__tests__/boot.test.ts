import type { AddressInfo } from 'node:net'

import { stripTerminalSequences, TuiMainScreen } from '@earendil-works/pi-tui'
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { TuiSession } from '../boot.js'

import { HarnessApp } from '../app.js'
import { bootTui } from '../boot.js'
import { Gateway } from '../gateway.js'
import { Theme } from '../theme.js'
import { FakeTerminal, FakeTransport } from './fakes.js'

const ESC = '\x1b'
const BEL = '\x07'
/** The lockup's violet, one palette each: `primary` at truecolor. */
const DARK_PRIMARY = '38;2;167;139;250'
const LIGHT_PRIMARY = '38;2;124;58;237'

const HELLO = {
  server_version: '9.9.9',
  server_capabilities: [],
  session: { default_channel: 'tui', default_session_key: 'tui:default' }
}

const CATALOG = {
  pairs: [
    ['help', 'help'],
    ['model', 'model']
  ],
  categories: [],
  skill_count: 7
}

const INFO = {
  cwd: '/work/proj',
  model: 'gpt-5',
  provider: 'openai',
  reasoning_effort: 'high',
  version: '0.4.2'
}

function scripted(overrides: Record<string, unknown> = {}) {
  return new FakeTransport({
    'system.hello': HELLO,
    'commands.catalog': CATALOG,
    'config.get': { config: {} },
    'setup.status': { provider_configured: true },
    'session.create': { session_id: 'tui:abc123', info: INFO },
    'session.resume': { session_id: 'tui:older', info: INFO, messages: [] },
    'session.most_recent': { session_id: 'tui:older' },
    'turn.send': { turn_id: 't1', accepted: true },
    'turn.cancel': { cancelled: true },
    ...overrides
  })
}

function build(transport: FakeTransport, env: NodeJS.ProcessEnv = {}) {
  const terminal = new FakeTerminal(100, 30)
  const tui = new TuiMainScreen(terminal)
  const theme = new Theme('dark', 3)
  const app = new HarnessApp({ env, gateway: new Gateway(transport), theme, tui })

  return { app, terminal, theme, tui }
}

describe('boot sequence', () => {
  let transport: FakeTransport

  beforeEach(() => {
    transport = scripted()
  })

  it('handshakes, reads the catalog and config, gates on setup, then opens a session', async () => {
    const { app } = build(transport)

    await app.boot()

    expect(transport.methods).toEqual([
      'system.hello',
      'commands.catalog',
      'config.get',
      'setup.status',
      'session.create',
      'turn.subscribe'
    ])
  })

  it('opens a session and subscribes to its key', async () => {
    const { app } = build(transport)

    await app.boot()

    const create = transport.calls.find(call => call.method === 'session.create')
    const subscribe = transport.calls.find(call => call.method === 'turn.subscribe')

    // No width: `session.create` does not read one. Terminal size reaches the
    // gateway through `terminal.resize`.
    expect(create?.params).toEqual({})
    expect(subscribe?.params).toEqual({ session_key: 'tui:abc123' })
  })

  it('resumes the session named by OPENDDE_HARNESS_TUI_RESUME', async () => {
    const { app } = build(transport, { OPENDDE_HARNESS_TUI_RESUME: 'tui:pinned' })

    await app.boot()

    expect(transport.methods).not.toContain('session.create')
    expect(transport.calls.find(call => call.method === 'session.resume')?.params).toEqual({
      session_id: 'tui:pinned'
    })
  })

  it('reopens the most recent session when the config asks for it', async () => {
    transport = scripted({ 'config.get': { config: { display: { tui_auto_resume_recent: true } } } })

    const { app } = build(transport)

    await app.boot()

    expect(transport.methods).toEqual([
      'system.hello',
      'commands.catalog',
      'config.get',
      'setup.status',
      'session.most_recent',
      'session.resume',
      'turn.subscribe'
    ])
  })

  it('creates a session when auto-resume finds nothing to reopen', async () => {
    transport = scripted({
      'config.get': { config: { 'display.tui_auto_resume_recent': true } },
      'session.most_recent': { session_id: null }
    })

    const { app } = build(transport)

    await app.boot()

    expect(transport.methods).toContain('session.most_recent')
    expect(transport.methods).toContain('session.create')
  })

  it('stops before the session when no provider is configured', async () => {
    transport = scripted({ 'setup.status': { provider_configured: false } })

    const { app, terminal, tui } = build(transport)

    await app.boot()

    expect(transport.methods).toEqual(['system.hello', 'commands.catalog', 'config.get', 'setup.status'])

    tui.renderNow()

    expect(terminal.output()).toContain('Setup required')
  })

  it('sends OPENDDE_HARNESS_TUI_QUERY once the session is live', async () => {
    const { app } = build(transport, { OPENDDE_HARNESS_TUI_QUERY: 'Say hello in five words' })

    await app.boot()

    const send = transport.calls.find(call => call.method === 'turn.send')

    expect(transport.methods.indexOf('turn.send')).toBeGreaterThan(transport.methods.indexOf('turn.subscribe'))
    expect(send?.params).toEqual({ content: 'Say hello in five words', session_key: 'tui:abc123' })
  })
})

describe('turn handling', () => {
  it('streams deltas into the transcript and ends the turn on message.complete', async () => {
    const transport = scripted()
    const { app, terminal, tui } = build(transport)

    await app.boot()
    await app.submit('hi')

    transport.emit({ type: 'message.start', payload: { turn_id: 't1' } })
    transport.emit({ type: 'token.delta', payload: { text: 'Hello ' } })
    transport.emit({ type: 'token.delta', payload: { text: 'there' } })
    transport.emit({ type: 'message.complete', payload: { turn_id: 't1', usage: {} } })

    tui.renderNow()

    const output = terminal.output()

    expect(output).toContain('Hello there')
    expect(output).toContain('hi')
  })

  it('names the tool on both ends of a tool call', async () => {
    const transport = scripted()
    const { app, terminal, tui } = build(transport)

    await app.boot()

    transport.emit({
      type: 'tool.start',
      payload: { arguments: { path: '/etc/hosts' }, name: 'read', tool_call_id: 'c1' }
    })
    transport.emit({
      type: 'tool.complete',
      payload: { result_preview: '127.0.0.1 localhost', tool_call_id: 'c1', truncated: true }
    })

    tui.renderNow()

    const output = terminal.output()

    expect(output).toContain('read')
    expect(output).toContain('/etc/hosts')
    expect(output).toContain('truncated')
  })

  it('holds a submit made while a turn is running, then sends it', async () => {
    const transport = scripted()
    const { app, terminal, tui } = build(transport)

    await app.boot()
    await app.submit('first')
    transport.emit({ type: 'message.start', payload: { turn_id: 't1' } })
    await app.submit('second')

    tui.renderNow()
    expect(terminal.output()).toContain('Queued: second')

    expect(transport.calls.filter(call => call.method === 'turn.send')).toHaveLength(1)

    transport.emit({ type: 'message.complete', payload: { turn_id: 't1', usage: {} } })
    await new Promise(resolve => setTimeout(resolve, 5))

    const sent = transport.calls.filter(call => call.method === 'turn.send')

    expect(sent).toHaveLength(2)
    expect(sent[1]!.params).toMatchObject({ content: 'second' })
  })

  it('reports a cancelled turn without an error block', async () => {
    const transport = scripted()
    const { app, terminal, tui } = build(transport)

    await app.boot()
    await app.submit('long one')

    transport.emit({
      type: 'error',
      payload: { code: -32800, message: 'cancelled', reason: 'cancelled_by_client' }
    })

    tui.renderNow()

    const output = terminal.output()

    expect(output).toContain('turn cancelled')
    expect(output).not.toContain('error:')
  })

  it('shows an approval prompt in the editor slot', async () => {
    const transport = scripted()
    const { app, terminal, tui } = build(transport)

    await app.boot()
    app.handleNotification('approval.request', {
      action_digest: 'digest',
      approval_id: 'approval-1',
      command: 'echo test',
      conversation_id: 'tui:abc123',
      created_at: Math.floor(Date.now() / 1000),
      description: 'test',
      // Unix seconds, as the broker sends them.
      expires_at: Math.floor(Date.now() / 1000) + 30,
      tool_call_id: 'tool-1',
      turn_id: 'turn-1'
    })

    tui.renderNow()

    const screen = stripTerminalSequences(terminal.output())

    expect(screen).toContain('Approval required')
    expect(screen).toContain('echo test')
    expect(screen).toContain('Allow once')
  })
})

describe('guarded boot', () => {
  it('exits 3 when the socket target is malformed', async () => {
    const codes: number[] = []
    const errors: string[] = []
    const session: TuiSession = {}

    // `net.createConnection` rejects this port synchronously, from inside the
    // client's constructor — the case that used to escape as Node's exit 1.
    const started = await bootTui(session, {
      exit: code => codes.push(code),
      socketPath: '127.0.0.1:99999',
      stderr: text => errors.push(text)
    })

    expect(started).toBeUndefined()
    expect(codes).toEqual([3])
    expect(errors.join('')).toMatch(/^opendde-tui: .*65536/)
    // Whatever cleanup was wired before the boot has nothing to close.
    expect(session.client).toBeUndefined()
  })
})

/** Wait for something observable rather than for a duration. */
async function waitFor(condition: () => boolean, what: string, timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs

  while (!condition()) {
    if (Date.now() > deadline) {
      throw new Error(`timed out waiting for ${what}`)
    }

    await new Promise(resolve => setTimeout(resolve, 2))
  }
}

describe('what a launched UI persists', () => {
  let dir: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'harness-boot-'))
  })

  afterEach(() => {
    rmSync(dir, { force: true, recursive: true })
  })

  /** A loopback gateway that answers the boot calls and records the sends.
   *  `reject` makes it refuse every call, which is a handshake that fails
   *  after the renderer has already taken the terminal. */
  async function gateway(options: { reject?: boolean; stall?: string } = {}): Promise<{
    close: () => Promise<void>
    port: number
  }> {
    const results: Record<string, unknown> = {
      'commands.catalog': CATALOG,
      'config.get': { config: {} },
      'session.create': { session_id: 'tui:abc123', info: INFO },
      'setup.status': { provider_configured: true },
      'system.hello': HELLO,
      'turn.send': { accepted: true, turn_id: 't1' },
      'turn.subscribe': { subscription_id: 'sub-1' }
    }

    const server = createServer(connection => {
      let buffer = ''

      connection.on('data', chunk => {
        buffer += chunk.toString('utf-8')

        let nl = buffer.indexOf('\n')

        while (nl !== -1) {
          const line = buffer.slice(0, nl)

          buffer = buffer.slice(nl + 1)

          if (line.trim().startsWith('{')) {
            const frame = JSON.parse(line) as { id?: number; method: string }

            if (frame.id !== undefined && frame.method !== options.stall) {
              const answer = options.reject
                ? { error: { code: -32603, message: 'synthetic boot failure' }, id: frame.id, jsonrpc: '2.0' }
                : { id: frame.id, jsonrpc: '2.0', result: results[frame.method] ?? {} }

              connection.write(`${JSON.stringify(answer)}\n`)
            }
          }

          nl = buffer.indexOf('\n')
        }
      })
    })

    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

    return {
      close: () => new Promise<void>(resolve => server.close(() => resolve())),
      port: (server.address() as AddressInfo).port
    }
  }

  it('starts on the alternate screen, and on the main screen when the environment says so', async () => {
    for (const [value, mode, entersAlt] of [
      [undefined, 'fullscreen', true],
      ['0', 'regular', false]
    ] as const) {
      const peer = await gateway()
      const session: TuiSession = {}
      const terminal = new FakeTerminal(100, 30)
      const started = await bootTui(session, {
        env: { OPENDDE_HARNESS_HOME: dir, ...(value === undefined ? {} : { OPENDDE_HARNESS_TUI_FULLSCREEN: value }) },
        exit: () => {},
        socketPath: `127.0.0.1:${peer.port}`,
        terminal
      })

      try {
        expect(started?.renderer.mode).toBe(mode)
        expect(terminal.output().includes('\x1b[?1049h')).toBe(entersAlt)
      } finally {
        started?.renderer.dispose()
        session.client?.close()
        await peer.close()
      }
    }
  })

  it('gives the screen back when the handshake fails after the renderer took it', async () => {
    const peer = await gateway({ reject: true })
    const session: TuiSession = {}
    const terminal = new FakeTerminal(100, 30)
    const codes: number[] = []
    const errors: string[] = []

    const started = await bootTui(session, {
      env: { OPENDDE_HARNESS_HOME: dir },
      exit: code => codes.push(code),
      socketPath: `127.0.0.1:${peer.port}`,
      stderr: text => errors.push(text),
      terminal
    })

    try {
      expect(started).toBeUndefined()
      expect(codes).toEqual([3])
      expect(errors.join('')).toContain('synthetic boot failure')

      const written = terminal.output()

      // The renderer starts before the handshake completes, so a boot that
      // fails after it has entered the alternate screen and turned autowrap
      // off is exactly the case that must still hand the terminal back.
      expect(written).toContain('\x1b[?1049h')
      expect(written).toContain('\x1b[?1049l')
      expect(session.stopRenderer).toBeDefined()

      // And the cleanup the entrypoint calls is safe to call again.
      session.stopRenderer?.()
    } finally {
      session.client?.close()
      await peer.close()
    }
  })

  it('hands the terminal back for a signal that lands while the boot is still in flight', async () => {
    // The handshake never comes back, so the renderer has taken the screen and
    // `app.boot()` is still awaiting when the cleanup runs.
    const peer = await gateway({ stall: 'system.hello' })
    const session: TuiSession = {}
    const terminal = new FakeTerminal(100, 30)

    const booting = bootTui(session, {
      env: { OPENDDE_HARNESS_HOME: dir },
      exit: () => {},
      socketPath: `127.0.0.1:${peer.port}`,
      stderr: () => {},
      terminal
    })

    try {
      await waitFor(() => terminal.output().includes('\x1b[?1049h'), 'the alternate screen')
      expect(session.stopRenderer).toBeDefined()

      // What the signal cleanup does.
      session.stopRenderer?.()
      expect(terminal.output()).toContain('\x1b[?1049l')
    } finally {
      session.client?.close()
      await peer.close()
      await booting
    }
  })

  it('paints the welcome in the palette the terminal answers with, and never in the guess', async () => {
    // A terminal that answers the background query the way a light one does,
    // as promptly as a local pty does.
    class LightTerminal extends FakeTerminal {
      write(data: string): void {
        super.write(data)

        if (data.includes(`${ESC}]11;?`)) {
          queueMicrotask(() => this.onInput?.(`${ESC}]11;rgb:ffff/ffff/ffff${BEL}`))
        }
      }
    }

    const peer = await gateway()
    const session: TuiSession = {}
    const terminal = new LightTerminal(100, 30)
    const started = await bootTui(session, {
      env: { OPENDDE_HARNESS_HOME: dir, OPENDDE_HARNESS_TUI_COLOR: 'truecolor' },
      exit: () => {},
      socketPath: `127.0.0.1:${peer.port}`,
      terminal
    })

    try {
      started?.tui.renderNow()

      expect(started?.theme.scheme).toBe('light')

      const written = terminal.output()

      // The lockup, in the light palette's violet.
      expect(written).toContain(LIGHT_PRIMARY)
      // And not once in the dark palette's. The app is built after the scheme
      // settles precisely so the guessed palette never reaches the terminal:
      // when it was built before, the lockup was composed dark, cached, and
      // painted dark on a light terminal.
      expect(written).not.toContain(DARK_PRIMARY)
    } finally {
      started?.renderer.dispose()
      session.client?.close()
      await peer.close()
    }
  })

  it('keeps the guessed palette when the terminal does not answer', async () => {
    const peer = await gateway()
    const session: TuiSession = {}
    const terminal = new FakeTerminal(100, 30)
    const started = await bootTui(session, {
      env: { OPENDDE_HARNESS_HOME: dir, OPENDDE_HARNESS_TUI_COLOR: 'truecolor' },
      exit: () => {},
      socketPath: `127.0.0.1:${peer.port}`,
      terminal
    })

    try {
      started?.tui.renderNow()

      expect(started?.theme.scheme).toBe('dark')
      expect(terminal.output()).toContain(DARK_PRIMARY)
    } finally {
      started?.renderer.dispose()
      session.client?.close()
      await peer.close()
    }
  })

  it('loads the prompt history from disk and writes new prompts back to it', async () => {
    const home = join(dir, 'home')
    const file = join(home, '.opendde_harness_history')

    mkdirSync(home)
    writeFileSync(file, '\n# 2026-01-01 00:00:00\n+a prompt from last time\n')

    const peer = await gateway()
    const session: TuiSession = {}
    const codes: number[] = []

    const started = await bootTui(session, {
      env: { OPENDDE_HARNESS_HOME: home },
      exit: code => codes.push(code),
      socketPath: `127.0.0.1:${peer.port}`,
      terminal: new FakeTerminal(100, 30)
    })

    try {
      expect(codes).toEqual([])
      expect(started).toBeDefined()

      await started!.app.submit('a prompt from this time')

      // Boot used to construct the app without any history at all, so a
      // launched UI neither read the previous process's prompts nor kept its
      // own.
      const stored = readFileSync(file, 'utf8')

      expect(stored).toContain('+a prompt from last time')
      expect(stored).toContain('+a prompt from this time')
    } finally {
      session.client?.close()
      await peer.close()
    }
  })
})
