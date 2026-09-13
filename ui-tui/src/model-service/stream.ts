// One `stream` request's attempts, and what each event carries when it travels.
//
// The retry loop is pi-ai's own `retryAssistantCall`: the policy, the
// classifier that decides a failure is worth repeating, and the backoff are
// all pi's, so a change to any of them arrives with a version bump rather than
// as an edit here. What this module adds is the wire: every non-terminal event
// is written as it arrives, an attempt that will be run again ends in a
// `retry` event rather than its own `done` or `error`, and only the attempt
// the loop settles on ends the request.
//
// The reading of a failure is pi's too -- `isRetryableAssistantError` for
// whether to repeat it and `isContextOverflow` for whether the window is what
// refused it -- so the caller decides with pi's own verdicts and no table of
// its own.
//
// The two gap deadlines are enforced here as well, because pi's own option is
// a whole-call one and these are per silence. An expired budget aborts that
// attempt and fails it with wording pi reads as transient, which puts a
// stalled stream under the same retry budget as a refused one -- it is the
// commonest failure a retry exists for, and it used to be the one failure
// nothing repeated.

import type { AssistantMessage, AssistantMessageEvent, RetryPolicy } from '@earendil-works/pi-ai'

import { isContextOverflow, isRetryableAssistantError, ModelsError, retryAssistantCall } from '@earendil-works/pi-ai'

import type { Reply, RequestId, StreamParams, TimeoutParams, WireEvent } from './protocol.js'

import { stripPartial } from './protocol.js'

/** The `AssistantMessage` a failure that never reached the model layer looks like. */
export function errorMessage(params: Pick<StreamParams, 'model' | 'provider'>, error: unknown): AssistantMessage {
  return {
    api: 'openai-completions',
    content: [],
    errorMessage: error instanceof Error ? error.message : String(error),
    model: params.model,
    provider: params.provider,
    role: 'assistant',
    stopReason: 'error',
    timestamp: Date.now(),
    usage: {
      cacheRead: 0,
      cacheWrite: 0,
      cost: { cacheRead: 0, cacheWrite: 0, input: 0, output: 0, total: 0 },
      input: 0,
      output: 0,
      totalTokens: 0
    }
  }
}

/**
 * pi's own reading of a failed turn, carried on the event the caller sees:
 * whether it is worth retrying, whether the context window is what refused it,
 * and -- when a `ModelsError` is what failed -- its code, so a caller
 * classifies a missing credential without matching text.
 *
 * `contextWindow` is passed to pi's `isContextOverflow` so the checks that need
 * it can run. Only failures are classified here, so the silent kind it also
 * knows -- a backend that accepts an oversized request and answers anyway, or
 * truncates it and stops at zero output -- is not reported: those arrive as a
 * `done`, which carries no verdict. Every provider that refuses outright is.
 */
export function classify(event: WireEvent, contextWindow: number, cause?: unknown): WireEvent {
  if (event.type !== 'error') {
    return event
  }
  const overflow = isContextOverflow(event.error, contextWindow)
  const retryable = isRetryableAssistantError(event.error)
  return cause instanceof ModelsError
    ? { ...event, code: cause.code, overflow, retryable }
    : { ...event, overflow, retryable }
}

/**
 * pi's retry policy for one request, from the budget the caller sent.
 *
 * `maxAgentDelayMs` is the cap pi applies to the doubling backoff. pi-ai 0.85
 * ignores the field and a later one reads it; sending it now is what makes the
 * upgrade a version bump rather than an edit here.
 */
export function retryPolicy(retry: StreamParams['retry']): RetryPolicy & { maxAgentDelayMs: number } {
  const maxRetries = retry?.maxRetries ?? 0
  return { baseDelayMs: 2000, enabled: maxRetries > 0, maxAgentDelayMs: 60_000, maxRetries }
}

/** What a budget answers with when it expires before the stream does. */
const TIMED_OUT = Symbol('timed out')

/** A whole number of seconds where the budget is one, so the wording reads. */
function seconds(ms: number): string {
  return `${Number((ms / 1000).toFixed(3))}s`
}

/**
 * `promise`, or {@link TIMED_OUT} when `ms` passes first.
 *
 * The losing promise is not left unhandled: both settlements are attached
 * before the race, so a rejection arriving after the budget expired is
 * swallowed rather than reported as unhandled.
 */
function withDeadline<T>(promise: Promise<T>, ms: number): Promise<T | typeof TIMED_OUT> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => resolve(TIMED_OUT), ms)
    promise.then(
      value => {
        clearTimeout(timer)
        resolve(value)
      },
      error => {
        clearTimeout(timer)
        reject(error)
      }
    )
  })
}

/** The budget covering this gap: the first event's, or every silence after. */
function budgetFor(timeouts: TimeoutParams | undefined, first: boolean): number | undefined {
  return first ? timeouts?.firstTokenMs : timeouts?.idleMs
}

/** What an expired budget says, in wording pi's own classifier calls transient. */
function timedOutText(ms: number, first: boolean): string {
  return first
    ? `timed out waiting for the first event after ${seconds(ms)}`
    : `timed out after ${seconds(ms)} of silence mid-stream`
}

/** One attempt's terminal event, held until the retry loop says it is the last. */
interface Held {
  event: WireEvent
  message: AssistantMessage
}

/**
 * The event that ends the request, for the message the retry loop settled on.
 *
 * Normally the held one: the same object came out of the stream that produced
 * it. The exception is an abort during the backoff sleep, which pi normalises
 * into an aborted copy of the failure it was about to retry -- no stream
 * produced that, so it becomes an event here.
 */
function terminalEvent(result: AssistantMessage, held: Held | undefined, contextWindow: number): WireEvent {
  if (held?.message === result) {
    return held.event
  }
  return classify(
    { error: result, reason: result.stopReason === 'aborted' ? 'aborted' : 'error', type: 'error' },
    contextWindow
  )
}

/**
 * Run one request's attempts, writing what each one produces.
 *
 * `attempt` is called once per try, 0-indexed, and returns that try's events;
 * the signal it is handed is that attempt's own, so an expired deadline ends
 * one attempt without ending the request. The request's `signal` aborts the
 * attempt in flight and the backoff sleep alike, which is why an abort
 * mid-wait still ends the request rather than the next attempt starting.
 */
export async function streamWithRetry(
  id: RequestId,
  params: StreamParams,
  contextWindow: number,
  signal: AbortSignal,
  write: (reply: Reply) => void,
  attempt: (index: number, signal: AbortSignal) => AsyncIterable<AssistantMessageEvent>
): Promise<void> {
  let held: Held | undefined
  let index = 0

  /** One attempt, bounded per gap. Returns what ends it. */
  const produce = async (): Promise<AssistantMessage> => {
    held = undefined
    // The attempt's own controller, so a deadline ends this try and not the
    // request. The outer signal is relayed into it: an abort must reach the
    // stream, which only ever sees this one.
    const controller = new AbortController()
    const relay = () => controller.abort()
    if (signal.aborted) {
      controller.abort()
    } else {
      signal.addEventListener('abort', relay, { once: true })
    }
    const iterator = attempt(index++, controller.signal)[Symbol.asyncIterator]()
    let first = true
    try {
      for (;;) {
        const budget = budgetFor(params.timeouts, first)
        const step = iterator.next()
        const next = budget === undefined ? await step : await withDeadline(step, budget)
        if (next === TIMED_OUT) {
          // Told to stop before the caller is told anything: an abandoned
          // stream on a real provider is still being generated, and still
          // being billed.
          controller.abort()
          void Promise.resolve(iterator.return?.()).catch(() => {})
          const failure = errorMessage(params, timedOutText(budget as number, first))
          held = {
            event: { ...classify({ error: failure, reason: 'error', type: 'error' }, contextWindow), code: 'timeout' },
            message: failure
          }
          return failure
        }
        if (next.done === true) {
          break
        }
        first = false
        const wire = classify(stripPartial(next.value), contextWindow)
        if (wire.type === 'done') {
          held = { event: wire, message: wire.message }
        } else if (wire.type === 'error') {
          held = { event: wire, message: wire.error }
        } else {
          write({ event: wire, id })
        }
      }
    } finally {
      signal.removeEventListener('abort', relay)
    }
    if (!held) {
      const failure = errorMessage(params, 'stream ended without a terminal event')
      held = { event: classify({ error: failure, reason: 'error', type: 'error' }, contextWindow), message: failure }
    }
    return held.message
  }

  const result = await retryAssistantCall(produce, retryPolicy(params.retry), signal, {
    onRetryScheduled: (retryAttempt, maxAttempts, delayMs, failure) => {
      write({ event: { attempt: retryAttempt, delayMs, errorMessage: failure, maxAttempts, type: 'retry' }, id })
    }
  })
  write({ event: terminalEvent(result, held, contextWindow), id })
}
