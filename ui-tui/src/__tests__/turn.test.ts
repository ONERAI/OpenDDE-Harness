// The turn model, driven by scripted event sequences against a fake terminal.

import type { Component } from '@earendil-works/pi-tui'

import { Container, stripTerminalSequences, TuiMainScreen } from '@earendil-works/pi-tui'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TurnEvent } from '../rpc/index.js'

import { AssistantMessage } from '../components/assistantMessage.js'
import { ToolExecution } from '../components/toolExecution.js'
import { UserMessage } from '../components/userMessage.js'
import { Gateway } from '../gateway.js'
import { Theme } from '../theme.js'
import { DEFAULT_WATCHDOG_MS, TurnController } from '../turn.js'
import { FakeTerminal, FakeTransport } from './fakes.js'

const WIDTH = 80

function text(component: Component, width = WIDTH): string {
  return component
    .render(width)
    .map(line => stripTerminalSequences(line).trimEnd())
    .join('\n')
}

/** Non-blank rendered lines, for asserting on a transcript's shape. */
function visibleLines(component: Component, width = WIDTH): string[] {
  return text(component, width)
    .split('\n')
    .filter(line => line.trim())
}

/** A transport that can answer a method only when the test says so, which is
 *  how a gateway that writes its response and the turn's terminal events into
 *  one chunk looks from here: the read loop delivers the events first, and the
 *  awaiting `send()` resumes afterwards. */
class HoldingTransport extends FakeTransport {
  private holding = new Set<string>()
  private waiting: (() => void)[] = []

  hold(method: string): void {
    this.holding.add(method)
  }

  release(method: string): void {
    this.holding.delete(method)

    const waiting = this.waiting

    this.waiting = []

    for (const resume of waiting) {
      resume()
    }
  }

  override async rpc<R = unknown, P = unknown>(method: string, params: P): Promise<R> {
    if (this.holding.has(method)) {
      await new Promise<void>(resume => this.waiting.push(resume))
    }

    return super.rpc<R, P>(method, params)
  }
}

function build(overrides: Record<string, unknown> = {}, transport = new FakeTransport()) {
  Object.assign(transport.results, {
    'turn.cancel': { cancelled: true },
    'turn.send': { accepted: true, turn_id: 't1' },
    ...overrides
  })

  const chat = new Container()
  const tui = new TuiMainScreen(new FakeTerminal(WIDTH, 30))
  const controller = new TurnController({
    chat,
    gateway: new Gateway(transport),
    now: () => clock,
    theme: new Theme('dark', 3),
    tui
  })

  return { chat, controller, transport, tui }
}

let clock = 0

async function attached(overrides: Record<string, unknown> = {}, transport?: FakeTransport) {
  const harness = build(overrides, transport)

  await harness.controller.attach('tui:test')

  const emit = (...events: TurnEvent[]): void => {
    for (const event of events) {
      harness.transport.emit(event)
    }
  }

  return { ...harness, emit }
}

const tokens = (...parts: string[]): TurnEvent[] =>
  parts.map(part => ({ payload: { text: part }, type: 'token.delta' }) as TurnEvent)

const START: TurnEvent = { payload: { turn_id: 't1' }, type: 'message.start' }
const USAGE_ZERO = { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }
const COMPLETE: TurnEvent = { payload: { turn_id: 't1', usage: USAGE_ZERO }, type: 'message.complete' }

/** A `message.complete` usage as the gateway sends one: this turn's figures,
 *  and the session's running totals beside them. The footer shows the session's
 *  and `/status` reports the same sum, so a completion that carried only the
 *  turn's is not a shape this UI has to read. */
function completion(
  turn: Record<string, null | number>,
  session: { calls?: number; cost?: null | number; input?: number; listCost?: null | number; output?: number } = {}
): TurnEvent {
  return {
    payload: {
      turn_id: 't1',
      usage: {
        completion_tokens: 0,
        prompt_tokens: 0,
        total_tokens: 0,
        ...turn,
        session_calls: session.calls ?? 1,
        session_cost_usd: session.cost ?? null,
        session_input_tokens: session.input ?? 0,
        session_list_cost_usd: session.listCost ?? null,
        session_output_tokens: session.output ?? 0
      }
    },
    type: 'message.complete'
  } as TurnEvent
}

beforeEach(() => {
  clock = 1_000
})

describe('sending', () => {
  it('echoes the message, sends it, and reports the turn active', async () => {
    const { chat, controller, transport } = await attached()

    await controller.send('hello there')

    expect(transport.methods).toEqual(['turn.subscribe', 'turn.send'])
    expect(transport.calls[1]!.params).toEqual({ content: 'hello there', session_key: 'tui:test' })
    expect(text(chat)).toContain('hello there')
    expect(controller.active).toBe(true)
  })

  it('refuses a second turn while one is in progress', async () => {
    const { controller } = await attached()

    await controller.send('first')

    await expect(controller.send('second')).rejects.toThrow('turn already in progress')
  })

  it('refuses to send with no session', async () => {
    const { controller } = build()

    await expect(controller.send('orphan')).rejects.toThrow('no session')
  })

  it('recovers and reports when the server does not accept the turn', async () => {
    const { chat, controller } = await attached({ 'turn.send': { accepted: false, turn_id: '' } })

    await controller.send('hello')

    expect(text(chat)).toContain('did not accept')
    expect(controller.active).toBe(false)
  })

  it('surfaces a failed turn.send and idles', async () => {
    const { chat, controller } = await attached({ 'turn.send': new Error('socket closed') })

    await expect(controller.send('hello')).rejects.toThrow('socket closed')
    expect(text(chat)).toContain('socket closed')
    expect(controller.active).toBe(false)
  })
})

describe('streaming', () => {
  it('renders one assistant message per turn, mutated per delta', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, ...tokens('Hello', ', ', 'world.'))

    expect(text(chat)).toContain('Hello, world.')

    const before = chat.children.length

    emit(...tokens(' And more.'))

    expect(chat.children.length).toBe(before)
    expect(text(chat)).toContain('Hello, world. And more.')

    emit(COMPLETE)

    expect(controller.active).toBe(false)
  })

  it('collapses thinking until it is toggled', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, { payload: { text: 'weighing the options' }, type: 'thinking.delta' })

    expect(text(chat)).toContain('Thinking…')
    expect(text(chat)).not.toContain('weighing the options')

    expect(controller.toggleThinking()).toBe(true)
    expect(text(chat)).toContain('weighing the options')

    expect(controller.toggleThinking()).toBe(false)
    expect(text(chat)).not.toContain('weighing the options')
  })

  it('keeps thinking and text in the order they arrived', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, { payload: { text: 'first thought' }, type: 'thinking.delta' }, ...tokens('the answer'), {
      payload: { text: 'second thought' },
      type: 'thinking.delta'
    })
    controller.toggleThinking()

    const lines = visibleLines(chat)

    expect(lines.findIndex(line => line.includes('first thought'))).toBeLessThan(
      lines.findIndex(line => line.includes('the answer'))
    )
    expect(lines.findIndex(line => line.includes('the answer'))).toBeLessThan(
      lines.findIndex(line => line.includes('second thought'))
    )
  })

  it('starts a new assistant message at an episode boundary', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, ...tokens('first call'))

    const afterFirst = chat.children.length

    emit({ payload: { index: 1 }, type: 'episode.start' }, ...tokens('second call'))

    expect(chat.children.length).toBe(afterFirst + 1)
    expect(text(chat)).toContain('first call')
    expect(text(chat)).toContain('second call')
  })

  it('brackets a reply with OSC 133 markers only when no tool follows', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, ...tokens('plain answer'))

    expect(chat.render(WIDTH).join('')).toContain('\x1b]133;A\x07')

    emit({
      payload: { arguments: {}, name: 'search', tool_call_id: 'c1' },
      type: 'tool.start'
    })

    const assistant = chat.children[1]!

    expect(assistant.render(WIDTH).join('')).not.toContain('\x1b]133;A\x07')
  })
})

describe('tools', () => {
  const start = (id = 'c1', name = 'read'): TurnEvent => ({
    payload: { arguments: { path: '/tmp/x' }, name, tool_call_id: id },
    type: 'tool.start'
  })

  it('renders the lifecycle: name, progress, result and duration', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, start())

    expect(text(chat)).toContain('read')
    expect(text(chat)).toContain('/tmp/x')

    emit({ payload: { preview: 'reading 40 lines', tool_call_id: 'c1' }, type: 'tool.progress' })
    expect(text(chat)).toContain('reading 40 lines')

    clock += 1_500
    emit({ payload: { result_preview: 'ok: 40 lines', tool_call_id: 'c1', truncated: false }, type: 'tool.complete' })

    // Collapsed, the row is the invocation and the timing; what the tool
    // returned is one key away and the progress line is gone.
    expect(text(chat)).toContain('1.5s')
    expect(text(chat)).toContain('ctrl+o to expand')
    expect(text(chat)).not.toContain('ok: 40 lines')
    expect(text(chat)).not.toContain('reading 40 lines')

    expect(controller.toggleTools()).toBe(true)
    expect(text(chat)).toContain('ok: 40 lines')
  })

  it('prefers the tool-authored display over an argument preview', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, {
      payload: { arguments: { path: '/tmp/x' }, display: 'read /tmp/x', name: 'read', tool_call_id: 'c1' },
      type: 'tool.start'
    })

    expect(text(chat)).toContain('read /tmp/x')
    expect(text(chat)).not.toContain('path=/tmp/x')
  })

  it('marks the truncated flag from the wire', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, start(), {
      payload: { result_preview: 'a long result', tool_call_id: 'c1', truncated: true },
      type: 'tool.complete'
    })

    expect(text(chat)).toContain('truncated')
  })

  it('hides a long result and expands it on demand', async () => {
    const { chat, controller, emit } = await attached()
    const result = Array.from({ length: 18 }, (_, i) => `line ${i}`).join('\n')

    await controller.send('hi')
    emit(START, start(), {
      payload: { result_preview: result, tool_call_id: 'c1', truncated: false },
      type: 'tool.complete'
    })

    expect(text(chat)).not.toContain('line 0')
    expect(text(chat)).toContain('ctrl+o to expand')

    expect(controller.toggleTools()).toBe(true)
    expect(text(chat)).toContain('line 17')

    expect(controller.toggleTools()).toBe(false)
    expect(text(chat)).not.toContain('line 17')
  })

  it('shows the first lines again for a session that turned quiet off', async () => {
    const { chat, controller, emit } = await attached()
    const result = Array.from({ length: 18 }, (_, i) => `line ${i}`).join('\n')

    await controller.send('hi')
    emit(START, start(), {
      payload: { result_preview: result, tool_call_id: 'c1', truncated: false },
      type: 'tool.complete'
    })

    // `/quiet-tools off` after the row is already on screen: the setting is the
    // transcript's, not only the next call's.
    controller.setToolsQuiet(false)

    expect(controller.quietTools).toBe(false)
    expect(text(chat)).toContain('line 9')
    expect(text(chat)).not.toContain('line 10')
    expect(text(chat)).toContain('8 more lines, ctrl+o to expand')

    controller.setToolsQuiet(true)

    expect(text(chat)).not.toContain('line 9')
  })

  it('says nothing more to the gateway when a row is folded or unfolded', async () => {
    const { controller, emit, transport } = await attached()

    await controller.send('hi')
    emit(START, start(), {
      payload: { result_preview: 'ok: 40 lines', tool_call_id: 'c1', truncated: false },
      type: 'tool.complete'
    })

    const sent = JSON.stringify(transport.calls)

    controller.setToolsQuiet(false)
    controller.setToolsQuiet(true)
    controller.toggleTools()

    // Drawing is the whole of it. What the model was given was settled by the
    // gateway before this panel saw a word of it, and no key or command here
    // sends anything about it.
    expect(JSON.stringify(transport.calls)).toBe(sent)
    expect(transport.methods).toEqual(['turn.subscribe', 'turn.send'])
  })

  it('shows what the tool returned, whatever word it opens with', async () => {
    const { chat, controller, emit } = await attached()

    // Loud, so the output itself is on screen: the point is that a result
    // opening with the word error is still drawn as the success it is.
    controller.setToolsQuiet(false)
    await controller.send('hi')
    emit(START, start(), {
      payload: { result_preview: 'Error: no such file', tool_call_id: 'c1', truncated: false },
      type: 'tool.complete'
    })

    expect(text(chat)).toContain('Error: no such file')
  })

  it('closes a tool that never reported a result when the turn ends', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, start(), {
      payload: { code: -32603, message: 'turn_failed', reason: 'internal' },
      type: 'error'
    })

    expect(text(chat)).toContain('no result')
    expect(controller.active).toBe(false)
  })
})

describe('notice', () => {
  it('shows a notice as a system line above the reply, and keeps the reply', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, {
      payload: { kind: 'model_fallback', text: 'Could not restore model openai/gpt-5-mini' },
      type: 'turn.notice'
    })

    expect(text(chat)).toContain('Could not restore model openai/gpt-5-mini')

    emit(...tokens('the answer'))

    // The line stands beside the reply, not inside it.
    expect(text(chat)).toContain('Could not restore model openai/gpt-5-mini')
    expect(text(chat)).toContain('the answer')
  })
})

describe('retry', () => {
  const retry = (discard: boolean): TurnEvent => ({
    payload: { attempt: 2, discard, reason: 'network', total: 4 },
    type: 'turn.retry'
  })

  it('drops the current call when the retry discards it', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, ...tokens('half an answ'), {
      payload: { arguments: {}, name: 'search', tool_call_id: 'c1' },
      type: 'tool.start'
    })

    expect(text(chat)).toContain('half an answ')
    expect(text(chat)).toContain('search')

    emit(retry(true))

    expect(text(chat)).not.toContain('half an answ')
    expect(text(chat)).not.toContain('search')
    expect(text(chat)).toContain('retrying 2/4: network')

    emit(...tokens('the whole answer'))
    expect(text(chat)).toContain('the whole answer')
  })

  it('keeps earlier calls when a later one is discarded', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(
      START,
      ...tokens('first call output'),
      { payload: { index: 1 }, type: 'episode.start' },
      ...tokens('second call')
    )
    emit(retry(true))

    expect(text(chat)).toContain('first call output')
    expect(text(chat)).not.toContain('second call')
  })

  it('keeps the streamed text when the retry does not discard', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, ...tokens('kept text'), retry(false))

    expect(text(chat)).toContain('kept text')
    expect(text(chat)).toContain('retrying 2/4')
  })
})

describe('the ack watchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('restores the input when no event arrives', async () => {
    const { chat, controller } = await attached()

    await controller.send('hi')
    expect(controller.active).toBe(true)

    vi.advanceTimersByTime(DEFAULT_WATCHDOG_MS)

    expect(text(chat)).toContain('turn produced no response')
    expect(controller.active).toBe(false)
  })

  it('is disarmed by the first event, whatever it is', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START)

    vi.advanceTimersByTime(DEFAULT_WATCHDOG_MS * 3)

    expect(text(chat)).not.toContain('turn produced no response')
    expect(controller.active).toBe(true)
  })
})

describe('ending a turn', () => {
  it('reports a cancellation as a plain note', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START, ...tokens('partial'), {
      payload: { code: -32000, message: 'cancelled', reason: 'cancelled_by_client' },
      type: 'error'
    })

    expect(text(chat)).toContain('turn cancelled')
    expect(text(chat)).toContain('partial')
    expect(controller.active).toBe(false)
  })

  it('attaches a failure to the reply it interrupted', async () => {
    const statuses: string[] = []
    const { chat, controller, emit } = await attached()

    controller.onStatus = status => statuses.push(status)

    await controller.send('hi')
    emit(START, ...tokens('as far as I got'), {
      payload: { code: -32603, detail: 'boom', message: 'turn_failed', reason: 'internal' },
      type: 'error'
    })

    expect(text(chat)).toContain('as far as I got')
    expect(text(chat)).toContain('error: turn_failed')
    expect(statuses.at(-1)).toBe('error')
  })

  it('asks the server to cancel and waits for its event', async () => {
    const { controller, transport } = await attached()

    await controller.send('hi')
    await controller.cancel()

    expect(transport.methods).toContain('turn.cancel')
    expect(controller.active).toBe(true)
  })

  it('calls back when the turn goes idle', async () => {
    const { controller, emit } = await attached()

    let idle = 0

    controller.onIdle = () => idle++

    await controller.send('hi')
    emit(START, COMPLETE)

    expect(idle).toBe(1)
  })

  it('does not go idle twice when a dropped turn reports back late', async () => {
    const { controller, emit } = await attached()

    let idle = 0

    controller.onIdle = () => idle++

    await controller.send('hi')
    controller.forceReset('dropped')

    expect(idle).toBe(1)

    emit({
      payload: {
        turn_id: 't1',
        usage: { completion_tokens: 7, prompt_tokens: 3, session_output_tokens: 7, total_tokens: 10 }
      },
      type: 'message.complete'
    })

    expect(idle).toBe(1)
    expect(controller.usage.output).toBe(7)
  })
})

describe('stale sends and stale events', () => {
  it('a send answered after its turn already finished reactivates nothing', async () => {
    const transport = new HoldingTransport()
    const { controller, emit } = await attached({}, transport)

    let idle = 0

    controller.onIdle = () => idle++

    // The valid wire order: the gateway writes the send response and both
    // terminal events into one chunk, and the read loop drains the events
    // before this continuation resumes.
    transport.hold('turn.send')

    const sending = controller.send('hi')

    emit(START, COMPLETE)

    expect(controller.active).toBe(false)
    expect(idle).toBe(1)

    transport.release('turn.send')
    await sending

    expect(controller.active).toBe(false)
    expect(idle).toBe(1)

    // And the next turn is not refused as one already in progress.
    await expect(controller.send('again')).resolves.toBeUndefined()
  })

  it('a send answered after the watchdog dropped the turn leaves it dropped', async () => {
    vi.useFakeTimers()

    try {
      const transport = new HoldingTransport()
      const { chat, controller } = await attached({}, transport)

      transport.hold('turn.send')

      const sending = controller.send('hi')

      vi.advanceTimersByTime(DEFAULT_WATCHDOG_MS)

      expect(text(chat)).toContain('turn produced no response')
      expect(controller.active).toBe(false)

      transport.release('turn.send')
      await sending

      expect(controller.active).toBe(false)
    } finally {
      vi.useRealTimers()
    }
  })

  it('a send answered after a session switch does not reach the new session', async () => {
    const transport = new HoldingTransport()
    const { controller } = await attached({}, transport)

    transport.hold('turn.send')

    const sending = controller.send('for the old session')

    await controller.attach('tui:new')
    transport.release('turn.send')
    await sending

    expect(controller.active).toBe(false)
  })

  it('a completion for a superseded turn does not end the one running', async () => {
    const { controller, emit } = await attached()

    let idle = 0

    controller.onIdle = () => idle++

    await controller.send('one')
    emit(START)
    controller.forceReset('dropped')

    expect(idle).toBe(1)

    await controller.send('two')
    emit({ payload: { turn_id: 't2' }, type: 'message.start' })

    emit({
      payload: {
        turn_id: 't1',
        usage: { completion_tokens: 5, prompt_tokens: 1, session_output_tokens: 5, total_tokens: 6 }
      },
      type: 'message.complete'
    })

    expect(controller.active).toBe(true)
    expect(idle).toBe(1)
    expect(controller.usage.output).toBe(0)

    emit({
      payload: {
        turn_id: 't2',
        usage: { completion_tokens: 9, prompt_tokens: 1, session_output_tokens: 9, total_tokens: 10 }
      },
      type: 'message.complete'
    })

    expect(controller.active).toBe(false)
    expect(idle).toBe(2)
    expect(controller.usage.output).toBe(9)
  })

  it('does not let a late start resurrect a turn that was dropped', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('hi')
    emit(START)
    controller.forceReset('dropped')

    expect(controller.active).toBe(false)

    // The gateway writes one more `message.start` for the turn that was just
    // dropped. Taking it reactivated the turn with no watchdog armed, and
    // every later send was refused as already in progress.
    emit(START)

    expect(controller.active).toBe(false)
    await expect(controller.send('the next one')).resolves.toBeUndefined()
    expect(text(chat)).toContain('the next one')
  })

  it('counts a completion once, however many times it arrives', async () => {
    const { controller, emit } = await attached()

    const complete = completion(
      { completion_tokens: 10, prompt_tokens: 100, total_tokens: 110 },
      { input: 100, output: 10 }
    )

    await controller.send('hi')
    emit(START, complete)

    expect(controller.usage).toMatchObject({ input: 100, output: 10 })

    // The same report again, while idle. It used to settle a second time. The
    // totals are the gateway's now, so a duplicate could not double them --
    // but it must still not end a turn twice, which is what it is dropped for.
    emit(complete)

    expect(controller.usage).toMatchObject({ input: 100, output: 10 })
  })

  it('drops an unlabelled completion when no turn is running', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(
      START,
      completion({ completion_tokens: 10, prompt_tokens: 100, total_tokens: 110 }, { input: 100, output: 10 })
    )

    emit({
      payload: {
        turn_id: null,
        usage: { completion_tokens: 5, prompt_tokens: 50, session_input_tokens: 150, total_tokens: 55 }
      },
      type: 'message.complete'
    })

    // Nothing names it and nothing is running, so there is nothing to
    // attribute it to.
    expect(controller.usage).toMatchObject({ input: 100, output: 10 })
  })

  it('ignores events from a subscription it has left', async () => {
    const { chat, controller, transport } = await attached()

    const stale = transport.emit

    await controller.attach('tui:new')

    stale({ payload: { text: 'from the old session' }, type: 'token.delta' })

    expect(text(chat)).not.toContain('from the old session')

    await controller.detach()

    transport.emit({ payload: { text: 'after the detach' }, type: 'token.delta' })

    expect(text(chat)).not.toContain('after the detach')
  })
})

describe('what the gateway stored', () => {
  const echoes = (chat: Container): UserMessage[] =>
    chat.children.filter((child): child is UserMessage => child instanceof UserMessage)

  const failed: TurnEvent = {
    payload: { code: -32008, message: 'model_not_available', reason: 'internal' },
    type: 'error'
  }

  it('marks a turn that failed before producing anything as never saved', async () => {
    const unsaved: string[] = []
    const { chat, controller, emit } = await attached()

    controller.onUnsaved = content => unsaved.push(content)

    await controller.send('nothing came of this')
    // The gateway's no-scheduler path: accepted, `message.start`, an error,
    // and no message stored.
    emit(START, failed)

    expect(echoes(chat).map(echo => echo.saved)).toEqual([false])
    expect(unsaved).toEqual(['nothing came of this'])
  })

  it('leaves a turn that produced something alone', async () => {
    const unsaved: string[] = []
    const { chat, controller, emit } = await attached()

    controller.onUnsaved = content => unsaved.push(content)

    await controller.send('this got an answer')
    emit(START, ...tokens('half an answer'), failed)

    // Whether the loop stored a partial turn is its business; the UI only
    // knows for certain about the turn that produced nothing at all.
    expect(echoes(chat).map(echo => echo.saved)).toEqual([true])
    expect(unsaved).toEqual([])
  })

  it('marks a cancelled turn as never saved, however much it streamed', async () => {
    const unsaved: string[] = []
    const { chat, controller, emit } = await attached()

    controller.onUnsaved = content => unsaved.push(content)

    await controller.send('cancelled halfway')
    emit(START, ...tokens('half an answer'), {
      payload: { code: -32800, message: 'cancelled', reason: 'cancelled_by_client' },
      type: 'error'
    })

    // The loop saves in `_after_turn`, which runs only after the agent loop
    // returns; cancellation propagates instead, so nothing was written no
    // matter how much of the reply reached the screen.
    expect(echoes(chat).map(echo => echo.saved)).toEqual([false])
    expect(unsaved).toEqual(['cancelled halfway'])
  })

  it('marks a send the gateway never accepted as never saved', async () => {
    const { chat, controller } = await attached({ 'turn.send': { accepted: false, turn_id: '' } })

    await controller.send('refused')

    expect(echoes(chat).map(echo => echo.saved)).toEqual([false])
  })

  it('marks a send that failed on the wire as never saved', async () => {
    const { chat, controller } = await attached({ 'turn.send': new Error('socket closed') })

    await expect(controller.send('never left')).rejects.toThrow('socket closed')
    expect(echoes(chat).map(echo => echo.saved)).toEqual([false])
  })
})

describe('details across the transcript', () => {
  it('collapses a panel that was expanded on its own', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('read it')
    emit(
      START,
      { payload: { arguments: { path: '/tmp/x' }, name: 'read', tool_call_id: 'c1' }, type: 'tool.start' },
      {
        payload: {
          result_preview: Array.from({ length: 14 }, (_, i) => `row ${i}`).join('\n'),
          tool_call_id: 'c1',
          truncated: false
        },
        type: 'tool.complete'
      }
    )

    const panel = chat.children.find((child): child is ToolExecution => child instanceof ToolExecution)

    // What a click on the panel does: this one alone, leaving the controller's
    // default at collapsed.
    panel!.setExpanded(true)
    expect(text(chat)).toContain('row 13')

    // What `/details tools collapsed` does. Applying it only when the default
    // moved left this panel expanded while reporting success.
    controller.setToolsExpanded(false)
    expect(text(chat)).not.toContain('row 13')
  })

  it('collapses reasoning that was expanded on its own', async () => {
    const { chat, controller, emit } = await attached()

    await controller.send('think about it')
    emit(START, { payload: { text: 'weighing two options' }, type: 'thinking.delta' })

    const reply = chat.children.find((child): child is AssistantMessage => child instanceof AssistantMessage)

    reply!.toggleThinking()
    expect(text(chat)).toContain('weighing two options')

    controller.setThinkingCollapsed(true)
    expect(text(chat)).not.toContain('weighing two options')
  })
})

describe('usage', () => {
  it("takes the session's totals from each completion, and the last call's context", async () => {
    const { controller, emit } = await attached()

    await controller.send('one')
    emit(
      START,
      completion(
        {
          cache_hit_percent: 50,
          completion_tokens: 20,
          context_max: 200_000,
          context_used: 900,
          cost_usd: 0.25,
          prompt_tokens: 100
        },
        { cost: 0.25, input: 100, output: 20 }
      )
    )

    await controller.send('two')
    emit(
      { payload: { turn_id: 't2' }, type: 'message.start' },
      {
        payload: {
          turn_id: 't2',
          usage: {
            cache_hit_percent: 90,
            completion_tokens: 5,
            context_max: 200_000,
            context_used: 1_100,
            cost_usd: 0.1,
            prompt_tokens: 200,
            // The session, summed by the gateway over both turns. Adding the
            // turn figures here instead is what let this drift from `/status`.
            session_calls: 2,
            session_cost_usd: 0.35,
            session_input_tokens: 300,
            session_output_tokens: 25,
            total_tokens: 205
          }
        },
        type: 'message.complete'
      }
    )

    expect(controller.usage).toEqual({
      cacheHitPercent: 90,
      contextMax: 200_000,
      contextUsed: 1_100,
      cost: 0.35,
      input: 300,
      listCost: null,
      output: 25,
      outputEstimated: false
    })
  })

  it('reads reasoning as a share of the output, not as more of it', async () => {
    const { controller, emit } = await attached()
    const seen: number[] = []

    controller.onUsage = usage => seen.push(usage.output)

    await controller.send('hi')
    emit(START, { payload: { calls: 1, completion_tokens: 12, reasoning_tokens: 8 }, type: 'turn.usage' })

    expect(controller.usage.output).toBe(12)

    emit(completion({ completion_tokens: 12, prompt_tokens: 10, total_tokens: 22 }, { input: 10, output: 12 }))

    expect(controller.usage.output).toBe(12)
    // Three publications, not two: sending the prompt is itself a change to
    // what the window holds, and the footer is told at once rather than at the
    // first token. The reply is zero at that point, which is what it is.
    expect(seen).toEqual([0, 12, 12])
  })

  it('settles the whole turn output, not just the last call of it', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(
      START,
      // Three calls: 12, then 12+10, then 12+10+8. `message.complete`'s own
      // `completion_tokens` reports only the last call, because the loop
      // overwrites its snapshot per iteration; the session field carries all of
      // them, which is why the figure survives the completion.
      { payload: { calls: 1, completion_tokens: 12, reasoning_tokens: 4 }, type: 'turn.usage' },
      { payload: { calls: 2, completion_tokens: 22, reasoning_tokens: 4 }, type: 'turn.usage' },
      { payload: { calls: 3, completion_tokens: 30, reasoning_tokens: 4 }, type: 'turn.usage' }
    )

    expect(controller.usage.output).toBe(30)

    emit(completion({ completion_tokens: 8, prompt_tokens: 10, total_tokens: 18 }, { calls: 3, input: 10, output: 30 }))

    expect(controller.usage.output).toBe(30)
  })

  it('takes the completion figure when the turn reported no running total', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(START, completion({ completion_tokens: 30, prompt_tokens: 10, total_tokens: 40 }, { input: 10, output: 30 }))

    expect(controller.usage.output).toBe(30)
  })

  it('starts again for another session, and keeps the figures for the same one', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(
      START,
      completion(
        { completion_tokens: 20, cost_usd: 0.25, prompt_tokens: 100, total_tokens: 120 },
        { cost: 0.25, input: 100, output: 20 }
      )
    )

    expect(controller.usage).toMatchObject({ cost: 0.25, input: 100, output: 20 })

    // A reconnect to the same session is not a new session.
    await controller.attach('tui:test')
    expect(controller.usage).toMatchObject({ cost: 0.25, input: 100, output: 20 })

    await controller.attach('tui:other')
    expect(controller.usage).toEqual({
      cacheHitPercent: null,
      contextMax: 0,
      contextUsed: 0,
      cost: null,
      input: 0,
      listCost: null,
      output: 0,
      outputEstimated: false
    })
  })

  it('estimates the output that is streaming, and gives way to the real figure', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(START, ...tokens('x'.repeat(400)), { payload: { text: 'y'.repeat(400) }, type: 'thinking.delta' })

    // 800 characters of reply and reasoning, at the four-characters-a-token
    // rule of thumb every vendor quotes. Before this the counter sat at zero
    // for the whole of a long answer.
    expect(controller.usage.output).toBe(200)
    expect(controller.usage.outputEstimated).toBe(true)

    emit({ payload: { calls: 1, completion_tokens: 180, reasoning_tokens: 60 }, type: 'turn.usage' })

    expect(controller.usage.output).toBe(180)
    expect(controller.usage.outputEstimated).toBe(false)

    // And the estimate resumes for what streams after that report.
    emit(...tokens('z'.repeat(40)))

    expect(controller.usage.output).toBe(190)
    expect(controller.usage.outputEstimated).toBe(true)

    emit(completion({ completion_tokens: 200, prompt_tokens: 10, total_tokens: 210 }, { input: 10, output: 200 }))

    expect(controller.usage.output).toBe(200)
    expect(controller.usage.outputEstimated).toBe(false)
  })

  it('publishes the estimate being dropped when a turn is cancelled', async () => {
    const seen: { estimated: boolean; output: number }[] = []
    const { controller, emit } = await attached()

    controller.onUsage = usage => seen.push({ estimated: usage.outputEstimated, output: usage.output })

    await controller.send('hi')
    emit(START, ...tokens('x'.repeat(40)))

    expect(seen.at(-1)).toEqual({ estimated: true, output: 10 })

    emit({
      payload: { code: -32800, message: 'cancelled', reason: 'cancelled_by_client' },
      type: 'error'
    })

    // Clearing the counters is not enough: whoever holds the last snapshot
    // keeps showing the estimate for a turn that is over.
    expect(controller.usage).toMatchObject({ output: 0, outputEstimated: false })
    expect(seen.at(-1)).toEqual({ estimated: false, output: 0 })
  })

  it('publishes the estimate being dropped on a force reset and a detach', async () => {
    const seen: boolean[] = []
    const { controller, emit } = await attached()

    controller.onUsage = usage => seen.push(usage.outputEstimated)

    await controller.send('hi')
    emit(START, ...tokens('x'.repeat(40)))
    controller.forceReset('dropped')

    expect(seen.at(-1)).toBe(false)

    await controller.send('again')
    emit({ payload: { turn_id: 't2' }, type: 'message.start' }, ...tokens('y'.repeat(40)))

    expect(seen.at(-1)).toBe(true)

    await controller.detach()

    expect(seen.at(-1)).toBe(false)
  })

  it("takes the session's list price, and keeps the vendor figure apart from it", async () => {
    const { controller, emit } = await attached()

    await controller.send('one')
    emit(
      START,
      completion(
        { completion_tokens: 10, cost_usd: null, list_cost_usd: 0.031, prompt_tokens: 100, total_tokens: 110 },
        { input: 100, listCost: 0.031, output: 10 }
      )
    )

    await controller.send('two')
    emit(
      { payload: { turn_id: 't2' }, type: 'message.start' },
      {
        payload: {
          turn_id: 't2',
          usage: {
            completion_tokens: 5,
            cost_usd: null,
            list_cost_usd: 0.025,
            prompt_tokens: 50,
            session_cost_usd: null,
            session_list_cost_usd: 0.056,
            total_tokens: 55
          }
        },
        type: 'message.complete'
      }
    )

    // On a plan the vendor reports no cost at all. The list price is what the
    // session's tokens are worth, summed call by call by the gateway -- the
    // same sum `/status` prints, which is the point of taking it rather than
    // adding the turn figures up here.
    expect(controller.usage.listCost).toBeCloseTo(0.056, 6)
    expect(controller.usage.cost).toBeNull()
  })

  it('keeps the reported cache share rather than inventing a token count', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(
      START,
      completion(
        { cache_hit_percent: 50, completion_tokens: 10, prompt_tokens: 100, total_tokens: 110 },
        { input: 100, output: 10 }
      )
    )

    // `input` is the total prompt, cached tokens included. The absolute cached
    // count that used to sit beside it was rebuilt from a rounded percentage
    // and read exact.
    expect(controller.usage).toMatchObject({ cacheHitPercent: 50, input: 100 })
    expect('cacheRead' in controller.usage).toBe(false)
  })

  it('leaves cost null when the provider states no price', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(START, COMPLETE)

    expect(controller.usage.cost).toBeNull()
  })
})

describe('what the context bar measures', () => {
  const completeWith = (usage: Record<string, boolean | number>): TurnEvent => ({
    payload: { turn_id: 't1', usage: { ...USAGE_ZERO, ...usage } },
    type: 'message.complete'
  })

  it('opens a resumed session at what its stored history holds', async () => {
    const { controller } = await attached()

    // The server's estimate of the messages it just handed over. Before this
    // the bar read 0% under a full transcript until the next turn reported.
    controller.setContextBaseline(40_000)

    expect(controller.usage.contextUsed).toBe(40_000)
  })

  it('counts what has been said since, on top of the opening estimate', async () => {
    const { controller } = await attached()

    controller.setContextBaseline(40_000)
    await controller.send('q'.repeat(400))

    expect(controller.usage.contextUsed).toBe(40_100)
  })

  it('drops the opening estimate once a call reports its own prompt', async () => {
    const { controller, emit } = await attached()

    controller.setContextBaseline(40_000)
    await controller.send('hi')
    emit(START, completeWith({ context_max: 200_000, context_used: 41_000 }))

    expect(controller.usage.contextUsed).toBe(41_000)
  })

  it("keeps the selected model's window when a turn on the old one finishes after the switch", async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    // The turn is running on a 1,000-token model when the session switches.
    // `session.info` for the model now selected says 2,000, and the footer
    // moves to it.
    controller.setContextWindow(2_000)

    expect(controller.usage.contextMax).toBe(2_000)

    // Now the old turn lands. Its capacity describes the model it was sent to,
    // which this session has left; its occupancy still describes this
    // conversation, which has not changed.
    emit(START, completeWith({ context_max: 1_000, context_used: 500 }))

    expect(controller.usage.contextMax).toBe(2_000)
    expect(controller.usage.contextUsed).toBe(500)
  })

  it('takes the window back from calls once a turn runs on the selected model', async () => {
    const { controller, emit } = await attached()

    await controller.send('one')
    controller.setContextWindow(2_000)
    emit(START, completeWith({ context_max: 1_000, context_used: 500 }))

    // A turn started after the switch belongs to the model now selected, so
    // its own report is authoritative again -- the gateway's figure is a
    // stand-in until a real call states one, not a permanent override.
    await controller.send('two')
    emit(
      { payload: { turn_id: 't2' }, type: 'message.start' },
      {
        payload: { turn_id: 't2', usage: { ...USAGE_ZERO, context_max: 3_000, context_used: 900 } },
        type: 'message.complete'
      }
    )

    expect(controller.usage.contextMax).toBe(3_000)
  })

  it('notices a switch between two models that happen to share a window size', async () => {
    const { controller, emit } = await attached()

    await controller.send('one')
    emit(START, completeWith({ context_max: 1_000, context_used: 100 }))

    await controller.send('two')
    // Same number, different model: the gateway's figure for the model now
    // selected happens to match the one already on screen. Nothing about the
    // size says whether the selection moved, so it cannot be what decides.
    controller.setContextWindow(1_000)

    // The old turn lands reporting a capacity that disagrees with both -- the
    // gateway answered about the new model, this call is the old one's own
    // account of itself. It belongs to a model this session has left.
    emit(
      { payload: { turn_id: 't2' }, type: 'message.start' },
      {
        payload: { turn_id: 't2', usage: { ...USAGE_ZERO, context_max: 5_000, context_used: 600 } },
        type: 'message.complete'
      }
    )

    expect(controller.usage.contextMax).toBe(1_000)
  })

  it('keeps counting after a turn that failed, because the failure was sent', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(START, completeWith({ context_max: 200_000, context_used: 1_000 }))

    await controller.send('q'.repeat(800))
    emit({ payload: { turn_id: 't2' }, type: 'message.start' }, ...tokens('y'.repeat(400)), {
      payload: { code: -32_000, message: 'upstream refused' },
      type: 'error'
    })

    // 800 characters of prompt and 400 of reply, at four characters a token,
    // on top of the last figure any call stated. The bar used to sit at the
    // pre-failure number until a turn succeeded.
    expect(controller.usage.contextUsed).toBe(1_300)
  })

  it('does not count a prompt the gateway never stored', async () => {
    const { controller, emit } = await attached({ 'turn.send': { accepted: false, turn_id: null } })

    await controller.send('hi')
    emit(START, completeWith({ context_max: 200_000, context_used: 1_000 }))

    // Refused: nothing was sent, so nothing joined the window. Counting it
    // walked the bar up on every failed attempt.
    await expect(controller.send('q'.repeat(800))).resolves.toBeUndefined()
    expect(controller.usage.contextUsed).toBe(1_000)
  })

  it('takes a discarded call back out of what the window holds', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(START, completeWith({ context_max: 200_000, context_used: 1_000 }))

    await controller.send('go')
    emit({ payload: { turn_id: 't2' }, type: 'message.start' }, ...tokens('y'.repeat(800)), {
      payload: { attempt: 1, discard: true, reason: 'stream broke', total: 3 },
      type: 'turn.retry'
    })

    // The retry sends the same conversation again, without a word of what the
    // broken call streamed. Only the two-character prompt joined the window.
    expect(controller.usage.contextUsed).toBe(1_001)

    // And the call that follows the retry counts from there.
    emit(...tokens('z'.repeat(400)))
    expect(controller.usage.contextUsed).toBe(1_101)
  })

  it('lets a new baseline supersede what a call reported', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(START, completeWith({ context_max: 200_000, context_used: 136_000 }))

    expect(controller.usage.contextUsed).toBe(136_000)

    // `/undo` took the exchange that call saw back out of the session.
    controller.setContextBaseline(27_200)

    expect(controller.usage.contextUsed).toBe(27_200)

    // And the estimate counts up from there again.
    await controller.send('q'.repeat(400))
    expect(controller.usage.contextUsed).toBe(27_300)
  })

  it('takes a baseline of zero literally, because an emptied session holds nothing', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(START, completeWith({ context_max: 200_000, context_used: 136_000 }))
    controller.setContextBaseline(0)

    expect(controller.usage.contextUsed).toBe(0)
  })

  it('counts a first turn that failed, with nothing measured before it', async () => {
    const { controller, emit } = await attached()

    await controller.send('q'.repeat(800))
    emit(START, ...tokens('y'.repeat(400)), {
      payload: { code: -32_000, message: 'upstream refused' },
      type: 'error'
    })

    // 800 characters of prompt and 400 of reply, at four characters a token.
    // Nothing had measured this session yet, and the bar read 0% under a
    // conversation the next call will send in full.
    expect(controller.usage.contextUsed).toBe(300)
  })

  it('says what the window holds when the prompt is sent, not when the reply starts', async () => {
    const { controller } = await attached()
    const seen: (null | number)[] = []

    controller.setContextBaseline(1_000)
    controller.onUsage = usage => seen.push(usage.contextUsed)

    await controller.send('q'.repeat(800))

    // The prompt is in front of the model from here. Before this the footer
    // held its last number for the whole of a long first token.
    expect(seen).toEqual([1_200])
  })

  it('takes back the whole of a cancelled turn, reply and all', async () => {
    const { controller, emit } = await attached()

    controller.setContextBaseline(1_000)
    await controller.send('q'.repeat(800))
    emit(START, ...tokens('y'.repeat(400)), {
      payload: { code: -32_001, message: 'cancelled', reason: 'cancelled_by_client' },
      type: 'error'
    })

    // The gateway stored nothing for it, which is why `/undo` skips it. The
    // estimate used to drop the prompt and keep the reply.
    expect(controller.usage.contextUsed).toBe(1_000)
  })

  it('keeps a failed turn that did produce something, because the gateway stored it', async () => {
    const { controller, emit } = await attached()

    controller.setContextBaseline(1_000)
    await controller.send('q'.repeat(800))
    emit(START, ...tokens('y'.repeat(400)), {
      payload: { code: -32_000, message: 'the stream broke' },
      type: 'error'
    })

    // An interrupted reply is one the loop returns and stores, so it is in the
    // conversation the next call sends.
    expect(controller.usage.contextUsed).toBe(1_300)
  })

  it('says nothing at all once a compaction has replaced the conversation', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(START, completeWith({ context_compacted: true, context_max: 200_000, context_used: 190_000 }))

    // 190k described the prompt the compaction just replaced. Nothing measures
    // what stands in its place, and pi shows `?` rather than the old figure.
    expect(controller.usage.contextUsed).toBeNull()

    await controller.send('q'.repeat(400))
    expect(controller.usage.contextUsed).toBeNull()
  })

  it('measures the window again as soon as a call runs after the compaction', async () => {
    const { controller, emit } = await attached()

    await controller.send('hi')
    emit(START, completeWith({ context_compacted: true, context_max: 200_000, context_used: 190_000 }))

    await controller.send('again')
    emit(
      { payload: { turn_id: 't2' }, type: 'message.start' },
      {
        payload: { turn_id: 't2', usage: { ...USAGE_ZERO, context_max: 200_000, context_used: 12_000 } },
        type: 'message.complete'
      }
    )

    expect(controller.usage.contextUsed).toBe(12_000)
  })

  it('forgets what one session held when another is attached', async () => {
    const { controller } = await attached()

    controller.setContextBaseline(40_000)
    await controller.attach('tui:other')

    expect(controller.usage.contextUsed).toBe(0)
  })
})

describe('background notices', () => {
  it('prints a delivered reminder', async () => {
    const { chat, emit } = await attached()

    emit({
      payload: { fired_at: '2026-09-11T09:00:00', job_id: 'j1', name: 'standup', text: 'write the update' },
      type: 'cron.delivered'
    })

    expect(text(chat)).toContain('standup')
    expect(text(chat)).toContain('write the update')
  })

  it('summarises missed reminders', async () => {
    const { chat, emit } = await attached()

    emit({
      payload: {
        count: 2,
        items: [
          { message: 'stand up', name: 'standup', scheduled_at: '2026-09-10T09:00:00' },
          { message: 'review', name: 'review', scheduled_at: '2026-09-10T17:30:00' }
        ]
      },
      type: 'cron.missed'
    })

    expect(text(chat)).toContain('missed 2 reminders')
    expect(text(chat)).toContain('09-10 09:00')
    expect(text(chat)).toContain('09-10 17:30')
  })

  type ProgressPayload = Extract<TurnEvent, { type: 'protein_design.progress' }>['payload']

  const progress = (
    status: ProgressPayload['status'],
    cycle?: number,
    eventType: ProgressPayload['event_type'] = 'task'
  ): TurnEvent => ({
    payload: {
      actor: '',
      candidate_count: null,
      cycle: cycle ?? null,
      duration_ms: null,
      error: null,
      event_id: 'event-1',
      event_type: eventType,
      has_details: false,
      phase: '',
      skill: null,
      status,
      summary: '',
      task_id: 'abcdef0123456789',
      timestamp: '2026-09-11T00:00:00Z',
      tool: null,
      total_cycles: 4
    },
    type: 'protein_design.progress'
  })

  it('hands every protein-design event to the monitor and writes none of them itself', async () => {
    const { chat, controller, emit } = await attached()
    const seen: ProgressPayload[] = []

    controller.onProteinProgress = payload => seen.push(payload)

    emit(progress('started'), progress('progress', 1), progress('completed', 4), progress('completed', 1, 'tool'))

    // The transcript line for a finished design belongs to the task monitor,
    // which is the only place that can tell an event apart from the snapshot
    // that confirms it.
    expect(visibleLines(chat)).toEqual([])
    expect(seen.map(payload => `${payload.event_type}:${payload.status}`)).toEqual([
      'task:started',
      'task:progress',
      'task:completed',
      'tool:completed'
    ])
  })

  it('does not let a background task event acknowledge the turn being sent', async () => {
    vi.useFakeTimers()

    try {
      const { chat, controller, emit } = await attached()

      await controller.send('design something')

      // A `protein_design.progress` broadcast reaches every subscribed
      // session. Treating it as the server answering *this* send would disarm
      // the watchdog guarding a turn nothing has acknowledged.
      emit(progress('progress', 1))
      expect(controller.active).toBe(true)

      vi.advanceTimersByTime(DEFAULT_WATCHDOG_MS)

      expect(text(chat)).toContain('turn produced no response')
      expect(controller.active).toBe(false)
    } finally {
      vi.useRealTimers()
    }
  })
})
