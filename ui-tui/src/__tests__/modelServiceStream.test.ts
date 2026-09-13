import type { AssistantMessage, AssistantMessageEvent } from '@earendil-works/pi-ai'

import { afterEach, describe, expect, it, vi } from 'vitest'

import type { Reply, StreamParams, TimeoutParams, WireEvent } from '../model-service/protocol.js'

import { classify, streamWithRetry } from '../model-service/stream.js'

const WINDOW = 200_000

function message(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
  return {
    api: 'anthropic-messages',
    content: [],
    model: 'claude',
    provider: 'anthropic',
    role: 'assistant',
    stopReason: 'stop',
    timestamp: 0,
    usage: {
      cacheRead: 0,
      cacheWrite: 0,
      cost: { cacheRead: 0, cacheWrite: 0, input: 0, output: 0, total: 0 },
      input: 0,
      output: 0,
      totalTokens: 0
    },
    ...overrides
  } as AssistantMessage
}

function failure(text: string): AssistantMessage {
  return message({ errorMessage: text, stopReason: 'error' })
}

/** A text delta as pi emits one: the `partial` snapshot is what travel strips. */
function textDelta(text: string): AssistantMessageEvent {
  return { contentIndex: 0, delta: text, partial: message(), type: 'text_delta' }
}

/** One attempt: a text delta, then the terminal event the script names. */
function attemptOf(final: AssistantMessage, text: string): AssistantMessageEvent[] {
  const terminal: AssistantMessageEvent =
    final.stopReason === 'error' || final.stopReason === 'aborted'
      ? { error: final, reason: final.stopReason, type: 'error' }
      : { message: final, reason: 'stop', type: 'done' }
  return [textDelta(text), terminal]
}

function params(maxRetries?: number, timeouts?: TimeoutParams): StreamParams {
  return {
    context: { messages: [] },
    model: 'claude',
    provider: 'anthropic',
    ...(maxRetries === undefined ? {} : { retry: { maxRetries } }),
    ...(timeouts === undefined ? {} : { timeouts })
  }
}

interface Options {
  maxRetries?: number
  signal?: AbortSignal
  timeouts?: TimeoutParams
}

/** Runs one request against `make`, returning every event line it wrote. */
async function drive(
  make: (index: number, signal: AbortSignal) => AsyncIterable<AssistantMessageEvent>,
  options: Options = {}
): Promise<WireEvent[]> {
  const written: WireEvent[] = []
  const write = (reply: Reply) => {
    if ('event' in reply) {
      written.push(reply.event as WireEvent)
    }
  }
  const signal = options.signal ?? new AbortController().signal
  await streamWithRetry(1, params(options.maxRetries, options.timeouts), WINDOW, signal, write, make)
  return written
}

/** Runs the scripted attempts, returning every event line the request wrote. */
async function run(
  script: AssistantMessageEvent[][],
  maxRetries?: number,
  signal: AbortSignal = new AbortController().signal
): Promise<WireEvent[]> {
  return drive(
    async function* (index) {
      for (const event of script[Math.min(index, script.length - 1)]) {
        yield event
      }
    },
    { maxRetries, signal }
  )
}

/** An attempt that yields what it is given and then goes quiet for good. */
function stalling(...events: AssistantMessageEvent[]) {
  return async function* (): AsyncGenerator<AssistantMessageEvent> {
    for (const event of events) {
      yield event
    }
    await new Promise(() => {})
  }
}

afterEach(() => {
  vi.useRealTimers()
})

describe('model-service stream retry', () => {
  it('announces a transient failure and lets the next attempt end the request', async () => {
    vi.useFakeTimers()
    const script = [attemptOf(failure('503 service unavailable'), 'half'), attemptOf(message(), 'whole')]

    const pending = run(script, 3)
    await vi.advanceTimersByTimeAsync(2000)
    const events = await pending

    expect(events.map(event => event.type)).toEqual(['text_delta', 'retry', 'text_delta', 'done'])
    // pi's own numbering: the retry about to run, out of the retries allowed.
    expect(events[1]).toEqual({
      attempt: 1,
      delayMs: 2000,
      errorMessage: '503 service unavailable',
      maxAttempts: 3,
      type: 'retry'
    })
    // The first attempt's failure never ended the request; the second one's
    // success did, and its text is what followed the retry.
    expect(events[2]).toMatchObject({ delta: 'whole' })
  })

  it('ends at the first failure when the request carries no budget', async () => {
    const events = await run([attemptOf(failure('503 service unavailable'), 'half')])

    expect(events.map(event => event.type)).toEqual(['text_delta', 'error'])
    expect(events[1]).toMatchObject({ overflow: false, retryable: true })
  })

  it('does not repeat a failure pi calls deterministic', async () => {
    const script = [
      attemptOf(failure('invalid request: the tool schema is malformed'), 'half'),
      attemptOf(message(), 'whole')
    ]

    const events = await run(script, 3)

    expect(events.map(event => event.type)).toEqual(['text_delta', 'error'])
    expect(events[1]).toMatchObject({ retryable: false })
  })

  it('ends the request as aborted when the backoff sleep is cancelled', async () => {
    vi.useFakeTimers()
    const controller = new AbortController()
    const script = [attemptOf(failure('503 service unavailable'), 'half'), attemptOf(message(), 'whole')]

    const pending = run(script, 3, controller.signal)
    await vi.advanceTimersByTimeAsync(1)
    controller.abort()
    const events = await pending

    expect(events.map(event => event.type)).toEqual(['text_delta', 'retry', 'error'])
    expect(events[2]).toMatchObject({ reason: 'aborted' })
  })

  it('writes a terminal error when an attempt ends without one', async () => {
    const events = await run([[textDelta('half')]])

    expect(events.map(event => event.type)).toEqual(['text_delta', 'error'])
    expect((events[1] as { error: AssistantMessage }).error.errorMessage).toMatch(/without a terminal event/)
  })
})

describe('model-service stream deadlines', () => {
  it('ends an attempt that never produces a first event', async () => {
    vi.useFakeTimers()
    const pending = drive(stalling(), { timeouts: { firstTokenMs: 5000 } })
    await vi.advanceTimersByTimeAsync(5000)
    const events = await pending

    expect(events.map(event => event.type)).toEqual(['error'])
    // pi's own classifier reads the wording, which is why it must say "timed
    // out": the retry budget covers a stalled stream only if this is transient.
    expect(events[0]).toMatchObject({ code: 'timeout', retryable: true })
    expect((events[0] as { error: AssistantMessage }).error.errorMessage).toBe(
      'timed out waiting for the first event after 5s'
    )
  })

  it('ends an attempt that goes quiet mid-stream, keeping what it wrote', async () => {
    vi.useFakeTimers()
    const pending = drive(stalling(textDelta('half')), { timeouts: { firstTokenMs: 60_000, idleMs: 2000 } })
    await vi.advanceTimersByTimeAsync(2000)
    const events = await pending

    expect(events.map(event => event.type)).toEqual(['text_delta', 'error'])
    expect((events[1] as { error: AssistantMessage }).error.errorMessage).toBe(
      'timed out after 2s of silence mid-stream'
    )
    expect(events[1]).toMatchObject({ code: 'timeout', retryable: true })
  })

  it('retries a stalled attempt under the same budget', async () => {
    vi.useFakeTimers()
    const stalled = stalling(textDelta('half'))
    const pending = drive(
      (index, _signal) =>
        index === 0
          ? stalled()
          : (async function* () {
              for (const event of attemptOf(message(), 'whole')) {
                yield event
              }
            })(),
      { maxRetries: 1, timeouts: { idleMs: 2000 } }
    )
    await vi.advanceTimersByTimeAsync(2000) // the deadline
    await vi.advanceTimersByTimeAsync(2000) // pi's backoff
    const events = await pending

    expect(events.map(event => event.type)).toEqual(['text_delta', 'retry', 'text_delta', 'done'])
    expect(events[1]).toMatchObject({ errorMessage: 'timed out after 2s of silence mid-stream' })
  })

  it('never cuts a stream that keeps delivering', async () => {
    vi.useFakeTimers()
    const pending = drive(
      async function* () {
        for (const text of ['a', 'b', 'c']) {
          await new Promise(resolve => setTimeout(resolve, 1500))
          yield textDelta(text)
        }
        yield attemptOf(message(), 'end')[1]
      },
      { timeouts: { firstTokenMs: 2000, idleMs: 2000 } }
    )
    await vi.advanceTimersByTimeAsync(10_000)
    const events = await pending

    // Four gaps of 1.5s against a 2s budget: longer in total than any deadline,
    // and not one silence over one. A long reply is not a fault.
    expect(events.map(event => event.type)).toEqual(['text_delta', 'text_delta', 'text_delta', 'done'])
  })

  it('aborts the attempt it ended and not the request', async () => {
    vi.useFakeTimers()
    const request = new AbortController()
    const seen: AbortSignal[] = []
    const pending = drive(
      (_index, attemptSignal) => {
        seen.push(attemptSignal)
        return stalling()()
      },
      { maxRetries: 1, signal: request.signal, timeouts: { firstTokenMs: 1000 } }
    )
    await vi.advanceTimersByTimeAsync(1000)
    await vi.advanceTimersByTimeAsync(2000)
    await vi.advanceTimersByTimeAsync(1000)
    await pending

    expect(seen).toHaveLength(2)
    expect(seen[0].aborted).toBe(true)
    expect(seen[1].aborted).toBe(true)
    // The request outlived both attempts it ended: a deadline is one try's.
    expect(request.signal.aborted).toBe(false)
  })
})

describe('model-service error classification', () => {
  const errorEvent = (final: AssistantMessage): WireEvent =>
    classify({ error: final, reason: 'error', type: 'error' }, WINDOW)

  it('marks a context overflow with pi own reading of the message', () => {
    expect(errorEvent(failure('prompt is too long: 213462 tokens > 200000 maximum'))).toMatchObject({
      overflow: true
    })
    expect(
      errorEvent(failure("Requested token count exceeds the model's maximum context length of 131072 tokens"))
    ).toMatchObject({ overflow: true })
  })

  it('does not read a throttle as an overflow', () => {
    // Bedrock's own wording for a throttle would match pi's "too many tokens"
    // overflow pattern; its exclusion list is what keeps the two apart.
    expect(errorEvent(failure('Throttling error: Too many tokens, please wait before trying again.'))).toMatchObject({
      overflow: false,
      retryable: false
    })
    expect(errorEvent(failure('429 rate limit exceeded'))).toMatchObject({ overflow: false, retryable: true })
  })

  it('reads a silent overflow from usage past the window', () => {
    const silent = message({
      usage: {
        cacheRead: 0,
        cacheWrite: 0,
        cost: { cacheRead: 0, cacheWrite: 0, input: 0, output: 0, total: 0 },
        input: WINDOW + 1,
        output: 0,
        totalTokens: WINDOW + 1
      }
    })

    // A `done` event carries no verdict: the field is on failures only.
    expect(classify({ message: silent, reason: 'stop', type: 'done' }, WINDOW)).not.toHaveProperty('overflow')
  })

  it('leaves a non-terminal event untouched', () => {
    expect(classify({ contentIndex: 0, delta: 'x', type: 'text_delta' }, WINDOW)).toEqual({
      contentIndex: 0,
      delta: 'x',
      type: 'text_delta'
    })
  })
})
