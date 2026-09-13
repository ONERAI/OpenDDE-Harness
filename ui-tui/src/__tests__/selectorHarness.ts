// Shared scaffolding for the phase-2 tests: a real TUI over a fake terminal, a
// gateway backed by the scripted transport, and a clock the test drives by hand.

import { Container, stripTerminalSequences, Text, TuiMainScreen } from '@earendil-works/pi-tui'

import type { CountdownClock } from '../components/countdownTimer.js'
import type { SelectorLease } from '../components/selector.js'
import type { LoginStepPush } from '../gateway.js'
import type { SelectorDeps, SelectorSession } from '../selectors/deps.js'

import { Gateway } from '../gateway.js'
import { createKeybindings } from '../lib/keybindings.js'
import { Theme } from '../theme.js'
import { FakeTerminal, FakeTransport } from './fakes.js'

export interface PrintedLine {
  text: string
  title?: string
  tone?: string
}

/** A clock whose time only moves when the test says so. */
export class ManualClock implements CountdownClock {
  readonly ticks: (() => void)[] = []

  private current: number

  constructor(startMs = 1_000_000) {
    this.current = startMs
  }

  nowMs(): number {
    return this.current
  }

  every(_ms: number, callback: () => void): () => void {
    this.ticks.push(callback)

    return () => {
      const index = this.ticks.indexOf(callback)

      if (index >= 0) {
        this.ticks.splice(index, 1)
      }
    }
  }

  /** Move time on and fire every live interval once. */
  advance(ms: number): void {
    this.current += ms

    for (const tick of [...this.ticks]) {
      tick()
    }
  }
}

export interface SelectorHarness {
  clock: ManualClock
  deps: SelectorDeps
  editor: Text
  editorContainer: Container
  gateway: Gateway
  /** Every URL a stage asked to open in the browser. */
  opened: string[]
  printed: PrintedLine[]
  session: SelectorSession
  terminal: FakeTerminal
  transport: FakeTransport
  tui: TuiMainScreen
  /** What a component draws, escape codes stripped. */
  draw(component: { render(width: number): string[] }, width?: number): string
  /** Push one sign-in step, the way the gateway's `login.step` notification
   *  reaches whatever stage is showing a login. */
  pushLoginStep(step: LoginStepPush): void
}

export function createHarness(
  results: Record<string, unknown> = {},
  session: Partial<SelectorSession> = {}
): SelectorHarness {
  const terminal = new FakeTerminal(100, 40)
  const tui = new TuiMainScreen(terminal)
  const theme = new Theme('dark', 0)
  const transport = new FakeTransport(results)
  const gateway = new Gateway(transport)
  const printed: PrintedLine[] = []
  const clock = new ManualClock()
  const editor = new Text('EDITOR', 0, 0)
  const editorContainer = new Container()
  const live: SelectorSession = { epoch: 1, id: 'tui:abc', ...session }
  const loginListeners = new Set<(step: LoginStepPush) => void>()
  const opened: string[] = []

  editorContainer.addChild(editor)

  const deps: SelectorDeps = {
    gateway,
    keybindings: createKeybindings(),
    onLoginStep: handler => {
      loginListeners.add(handler)

      return () => loginListeners.delete(handler)
    },
    openBrowser: url => opened.push(url),
    session: () => live,
    theme,
    transcript: {
      print: (text, tone) => printed.push({ text, ...(tone ? { tone } : {}) }),
      printBlock: (text, title) => printed.push({ text, ...(title ? { title } : {}) })
    },
    tui
  }

  return {
    clock,
    deps,
    draw: (component, width = 100) => stripTerminalSequences(component.render(width).join('\n')),
    editor,
    editorContainer,
    gateway,
    opened,
    printed,
    pushLoginStep: step => {
      for (const listener of [...loginListeners]) {
        listener(step)
      }
    },
    session: live,
    terminal,
    transport,
    tui
  }
}

/** Let queued microtasks run; the selectors do their loading in promises. */
export async function settle(times = 4): Promise<void> {
  for (let i = 0; i < times; i += 1) {
    await Promise.resolve()
  }
}

export const KEYS = {
  ctrlA: '\x01',
  ctrlC: '\x03',
  ctrlD: '\x04',
  ctrlE: '\x05',
  ctrlS: '\x13',
  down: '\x1b[B',
  enter: '\r',
  escape: '\x1b',
  pageDown: '\x1b[6~',
  pageUp: '\x1b[5~',
  shiftTab: '\x1b[Z',
  tab: '\t',
  up: '\x1b[A'
}

/** One provider row, with the fields the picker reads.
 *
 *  One of pi's own: it carries its own address and wire, so `base_url` and
 *  `api` are null and the credential is a key. Every model id is qualified
 *  (`<provider>/<model>`), which is what `config.set` and `model.add_model`
 *  take, and `model_labels` is what shortens one for a list already headed by
 *  its provider. */
export function providerRow(overrides: Record<string, unknown> = {}) {
  return {
    api: null,
    // pi's own list of ways in, as the gateway reads it off pi's provider
    // objects: this one takes a key and nothing else.
    auth_methods: ['key'],
    auth_type: 'key',
    authenticated: true,
    base_url: null,
    key_label: 'OpenAI API key',
    login_label: null,
    is_current: false,
    key_env: 'OPENAI_API_KEY',
    model_labels: { 'openai/gpt-5': { label: 'gpt-5' } },
    models: ['openai/gpt-5'],
    models_loaded: true,
    name: 'OpenAI',
    needs_api_key: false,
    needs_base_url: false,
    slug: 'openai',
    total_models: 1,
    warning: '',
    ...overrides
  }
}

/** One provider the config *declares*: reached by an address, and by the wire
 *  that address serves. Both are asked for together or not at all, and its key
 *  is optional because a self-hosted server usually wants none. */
export function declaredRow(overrides: Record<string, unknown> = {}) {
  return providerRow({
    api: null,
    auth_methods: ['key'],
    auth_type: 'endpoint',
    authenticated: false,
    base_url: null,
    key_env: null,
    key_label: null,
    model_labels: {},
    models: [],
    name: 'Local llama.cpp',
    needs_api_key: false,
    needs_base_url: true,
    slug: 'llama-cpp',
    total_models: 0,
    ...overrides
  })
}

export interface FakeLease {
  closes: string[]
  /** Pretend a newer selector took the slot. */
  invalidate(): void
  lease: SelectorLease
}

/** A lease a test can revoke, without a whole SelectorHost. */
export function fakeLease(sessionId: null | string = 'tui:abc', epoch = 1): FakeLease {
  const controller = new AbortController()
  const closes: string[] = []
  let current = true

  return {
    closes,
    invalidate() {
      current = false
      controller.abort('replaced')
    },
    lease: {
      done(reason = 'done') {
        closes.push(reason)
        current = false
      },
      isCurrent: () => current,
      sessionEpoch: epoch,
      sessionId,
      signal: controller.signal,
      token: Symbol('lease')
    }
  }
}
