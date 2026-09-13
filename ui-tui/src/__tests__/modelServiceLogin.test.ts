// One `login` request: what travels, and what the flow waits for.
//
// pi's own flow is faked, because the real one talks to a vendor. The fake is
// shaped like pi's: it prompts through the interaction it is handed and returns
// a credential, so what is under test is the wire — which steps are written,
// which of them the flow stops on, and that a `login_answer` is what lets it go
// on.

import type { AuthInteraction, AuthPrompt } from '@earendil-works/pi-ai'

import { describe, expect, it } from 'vitest'

import type { LoginModels } from '../model-service/login.js'
import type { LoginEvent, Reply } from '../model-service/protocol.js'

import { answerLogin, awaitingAnswer, runLogin } from '../model-service/login.js'

const ID = 7

/** pi's own login-method menu, as `openai-codex` asks it. */
const MENU: AuthPrompt = {
  message: 'Select OpenAI Codex login method:',
  options: [
    { id: 'browser', label: 'Browser login (default)' },
    { id: 'device_code', label: 'Device code login (headless)' }
  ],
  type: 'select'
}

interface Run {
  done: Promise<void>
  /** Every reply the request wrote, in order. */
  replies: Reply[]
}

/**
 * Run one login against a scripted flow.
 *
 * `script` is what pi's own `login` would do with the interaction: prompt,
 * notify, and return. Whatever it returns is the credential.
 */
function run(
  script: (interaction: AuthInteraction) => Promise<{ type: string }>,
  params: { mode?: string; provider: string } = { provider: 'openai-codex' },
  signal: AbortSignal = new AbortController().signal
): Run {
  const replies: Reply[] = []
  const models: LoginModels = {
    login: (_provider, _type, interaction) => script(interaction)
  }
  const done = runLogin(ID, params, models, signal, reply => replies.push(reply))

  return { done, replies }
}

/** The login steps among the replies, in order. */
function steps(replies: Reply[]): LoginEvent[] {
  return replies
    .filter((reply): reply is { event: LoginEvent; id: number } => 'event' in reply)
    .map(reply => reply.event)
}

function asked(step: LoginEvent): AuthPrompt | undefined {
  return 'ask' in step ? step.ask : undefined
}

/** Let the flow reach its prompt before the test answers it. */
async function settle(times = 4): Promise<void> {
  for (let i = 0; i < times; i += 1) {
    await Promise.resolve()
  }
}

describe('model-service login', () => {
  it('forwards the login-method menu and waits for an answer to it', async () => {
    let chosen = ''
    const login = run(async interaction => {
      chosen = await interaction.prompt(MENU)

      return { type: 'oauth' }
    })

    await settle()

    // The menu travelled as pi wrote it, options and all.
    expect(asked(steps(login.replies)[0]!)).toEqual(MENU)
    // And nothing has been decided: the flow is waiting on this side.
    expect(chosen).toBe('')
    expect(awaitingAnswer(ID)).toBe(true)

    expect(answerLogin(ID, 'device_code')).toBe(true)
    await login.done

    expect(chosen).toBe('device_code')
    expect(login.replies.at(-1)).toEqual({ id: ID, result: { provider: 'openai-codex', type: 'oauth' } })
    expect(awaitingAnswer(ID)).toBe(false)
  })

  it('answers the menu from `mode` without waiting for anything', async () => {
    let chosen = ''
    const login = run(
      async interaction => {
        chosen = await interaction.prompt(MENU)

        return { type: 'oauth' }
      },
      { mode: 'device_code', provider: 'openai-codex' }
    )

    await login.done

    expect(chosen).toBe('device_code')
    // The step is still reported -- a caller shows what pi asked -- but no
    // answer was ever owed.
    expect(asked(steps(login.replies)[0]!)).toEqual(MENU)
    expect(awaitingAnswer(ID)).toBe(false)
  })

  it('forwards the menu when `mode` names an option it does not offer', async () => {
    const login = run(
      async interaction => {
        await interaction.prompt(MENU)

        return { type: 'oauth' }
      },
      { mode: 'carrier-pigeon', provider: 'openai-codex' }
    )

    await settle()

    expect(awaitingAnswer(ID)).toBe(true)
    expect(answerLogin(ID, 'browser')).toBe(true)
    await login.done
  })

  it('waits for the pasted authorization code, and shows what came before it', async () => {
    let pasted = ''
    const login = run(async interaction => {
      interaction.notify({ type: 'auth_url', url: 'https://auth.test/start' })
      pasted = await interaction.prompt({ message: 'Paste the code', type: 'manual_code' })

      return { type: 'oauth' }
    })

    await settle()

    const shown = steps(login.replies)

    expect(shown[0]).toEqual({ notify: { type: 'auth_url', url: 'https://auth.test/start' }, type: 'login_prompt' })
    expect(asked(shown[1]!)).toEqual({ message: 'Paste the code', type: 'manual_code' })
    expect(answerLogin(ID, 'code-from-the-browser')).toBe(true)
    await login.done

    expect(pasted).toBe('code-from-the-browser')
  })

  it('drops the prompt when pi resolves that step itself', async () => {
    // The browser login races its callback server against the paste prompt and
    // aborts the prompt when the callback wins.
    const perPrompt = new AbortController()
    const login = run(async interaction => {
      const pending = interaction.prompt({ message: 'Paste the code', signal: perPrompt.signal, type: 'manual_code' })

      perPrompt.abort()
      await expect(pending).rejects.toThrow(/before its manual_code step was answered/)

      return { type: 'oauth' }
    })

    await login.done

    expect(awaitingAnswer(ID)).toBe(false)
    // The prompt travelled without its signal, which is not JSON.
    expect(asked(steps(login.replies)[0]!)).toEqual({ message: 'Paste the code', type: 'manual_code' })
    expect(login.replies.at(-1)).toEqual({ id: ID, result: { provider: 'openai-codex', type: 'oauth' } })
  })

  it('fails the login when the whole request is aborted', async () => {
    const controller = new AbortController()
    const login = run(
      async interaction => {
        await interaction.prompt(MENU)

        return { type: 'oauth' }
      },
      { provider: 'openai-codex' },
      controller.signal
    )

    await settle()
    controller.abort()
    await login.done

    expect(login.replies.at(-1)).toMatchObject({
      error: { code: 'login_failed', message: expect.stringContaining('before its select step was answered') },
      id: ID
    })
    // Nothing is owed any more, so a late answer reaches nothing.
    expect(answerLogin(ID, 'browser')).toBe(false)
  })

  it("keeps a later prompt answerable when an earlier prompt's signal fires late", async () => {
    // pi's browser login answers its paste prompt and then aborts that prompt's
    // signal on the way out. The listener the answered prompt had left on it
    // used to delete whichever resolver was pending by then -- the next
    // prompt's -- so the menu that followed could never be answered.
    const first = new AbortController()
    const login = run(async interaction => {
      const code = await interaction.prompt({ message: 'Paste the code', signal: first.signal, type: 'manual_code' })
      const menu = interaction.prompt(MENU)

      first.abort()

      return { type: `${code}:${await menu}` }
    })

    await settle()
    expect(answerLogin(ID, 'xyz')).toBe(true)
    await settle()

    expect(awaitingAnswer(ID)).toBe(true)
    expect(answerLogin(ID, 'oauth')).toBe(true)
    await login.done

    expect(login.replies.at(-1)).toEqual({ id: ID, result: { provider: 'openai-codex', type: 'xyz:oauth' } })
  })

  it('rejects a prompt raised after the login was already aborted', async () => {
    const controller = new AbortController()
    const login = run(
      async interaction => {
        controller.abort()
        await interaction.prompt(MENU)

        return { type: 'oauth' }
      },
      { provider: 'openai-codex' },
      controller.signal
    )

    await login.done

    expect(login.replies.at(-1)).toMatchObject({
      error: { code: 'login_failed', message: expect.stringContaining('before its select step was answered') }
    })
    expect(awaitingAnswer(ID)).toBe(false)
  })

  it('refuses a prompt no caller answers rather than hanging on it', async () => {
    const login = run(async interaction => {
      await interaction.prompt({ message: 'Your API key', type: 'secret' })

      return { type: 'oauth' }
    })

    await login.done

    expect(login.replies.at(-1)).toMatchObject({
      error: { code: 'login_failed', message: expect.stringContaining('secret answer the model service cannot give') }
    })
  })

  it('answers nothing for a login that is not waiting', () => {
    expect(answerLogin('no-such-login', 'browser')).toBe(false)
  })
})
