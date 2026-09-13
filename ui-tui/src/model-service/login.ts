// pi's own login flow, run over the wire, with its questions answered from
// outside this process.
//
// The flow is pi-ai's unchanged: `models.login` drives it and every step it
// reports arrives here as an `AuthEvent` to show (`notify`) or an `AuthPrompt`
// to answer (`ask`). Both travel as `login_prompt` events on the request's own
// id, so a caller sees the device code, the sign-in URL and the login-method
// menu as pi produces them.
//
// A prompt is answered by a later `login_answer` request naming this login,
// which is what lets the answer come from a person at the other end of the pipe
// rather than from a `mode` decided before the flow started. `mode` still
// short-circuits the login-method menu, because the CLI knows which method it
// was asked for and there is nobody to ask.
//
// One answer is owed at a time: pi asks, waits for the string, and only then
// asks again, so the pending resolver is keyed by the login alone.

import type { AuthEvent, AuthInteraction, AuthPrompt } from '@earendil-works/pi-ai'

import type { LoginParams, Reply, RequestId } from './protocol.js'

/** The one thing a login needs from pi's model collection. */
export interface LoginModels {
  login(provider: string, type: 'oauth', interaction: AuthInteraction): Promise<{ type: string }>
}

export type WriteReply = (reply: Reply) => void

/** Resolvers for the answers still owed, by the login that is waiting. */
const pending = new Map<RequestId, (answer: string) => void>()

/** A prompt as it travels: everything but the `AbortSignal`, which is not JSON. */
export function askable(prompt: AuthPrompt): AuthPrompt {
  const { signal: _signal, ...rest } = prompt

  return rest as AuthPrompt
}

/**
 * Hand a waiting login the string it asked for: an option id for a `select`,
 * the pasted code for a `manual_code`.
 *
 * False when nothing is waiting under that id -- the login ended, or the prompt
 * was resolved by pi's own callback server winning the race.
 */
export function answerLogin(id: RequestId, answer: string): boolean {
  const resolve = pending.get(id)

  if (resolve === undefined) {
    return false
  }

  resolve(answer)

  return true
}

/** Whether an answer is owed for this login. For tests and for `login_answer`. */
export function awaitingAnswer(id: RequestId): boolean {
  return pending.has(id)
}

/**
 * The promise pi waits on while the answer travels.
 *
 * Either signal ends it: `prompt.signal` is pi cancelling one step (its
 * callback server answered the browser login before the code was pasted), and
 * the request's own signal is the whole login being aborted.
 */
function awaited(id: RequestId, prompt: AuthPrompt, signal: AbortSignal): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const ended = () => new Error(`the login ended before its ${prompt.type} step was answered`)

    // A signal that has already fired never fires again, so a prompt raised
    // after its login was aborted would otherwise wait for an answer nobody
    // can send.
    if (signal.aborted || prompt.signal?.aborted) {
      reject(ended())

      return
    }

    // Settling removes both listeners and only this prompt's resolver: a
    // listener left on the request's signal outlives the prompt, and when the
    // signal fires later it would delete whatever prompt is pending by then.
    const settle = () => {
      prompt.signal?.removeEventListener('abort', cancel)
      signal.removeEventListener('abort', cancel)

      if (pending.get(id) === answer) {
        pending.delete(id)
      }
    }
    const answer = (value: string) => {
      settle()
      resolve(value)
    }
    const cancel = () => {
      settle()
      reject(ended())
    }

    pending.set(id, answer)
    prompt.signal?.addEventListener('abort', cancel, { once: true })
    signal.addEventListener('abort', cancel, { once: true })
  })
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Run one login and write everything it produces.
 *
 * `mode` answers the login-method menu without a round trip, for a caller that
 * was told which method to use. Without it the menu travels like any other
 * prompt and the login waits for `login_answer`. A prompt neither an OAuth
 * flow's menu nor its paste-the-code fallback -- an API-key flow asking for the
 * key -- fails the login rather than waiting for an answer no caller sends.
 */
export async function runLogin(
  id: RequestId,
  params: LoginParams,
  models: LoginModels,
  signal: AbortSignal,
  write: WriteReply
): Promise<void> {
  const interaction: AuthInteraction = {
    notify: (event: AuthEvent) => write({ event: { notify: event, type: 'login_prompt' }, id }),
    prompt: (prompt: AuthPrompt) => {
      write({ event: { ask: askable(prompt), type: 'login_prompt' }, id })

      if (params.mode && prompt.type === 'select' && prompt.options.some(option => option.id === params.mode)) {
        return Promise.resolve(params.mode)
      }

      if (prompt.type === 'select' || prompt.type === 'manual_code') {
        return awaited(id, prompt, signal)
      }

      return Promise.reject(new Error(`this login needs a ${prompt.type} answer the model service cannot give`))
    },
    signal
  }

  try {
    const credential = await models.login(params.provider, 'oauth', interaction)

    write({ id, result: { provider: params.provider, type: credential.type } })
  } catch (error) {
    write({ error: { code: 'login_failed', message: message(error) }, id })
  } finally {
    pending.delete(id)
  }
}
