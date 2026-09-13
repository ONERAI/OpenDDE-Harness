// pi's /login and /logout, stage by stage: the authentication method, the
// provider, then the key form, the endpoint form or the sign-in itself.
import { describe, expect, it } from 'vitest'

import { rpcErrorFromFrame } from '../rpc/index.js'
import { createAuthSelector } from '../selectors/authSelector.js'
import { createHarness, declaredRow, fakeLease, KEYS, providerRow, settle } from './selectorHarness.js'

const OPENAI = providerRow({ is_current: true })
/** One the config declares: an address and the wire it serves, and no key required. */
const LOCAL = declaredRow()
/** One of pi's own with nothing supplying its key, which is the only row the key form insists on. */
const NEEDS_A_KEY = providerRow({ authenticated: false, is_current: false, key_env: null, needs_api_key: true })
const GPT5 = 'openai/gpt-5'

function options(providers = [OPENAI], model = GPT5) {
  return { model, provider: 'openai', providers }
}

function build(
  results: Record<string, unknown> = {},
  _scope: 'default' | 'session' = 'session',
  sessionId: null | string = 'tui:abc',
  entry: { entry?: 'login' | 'logout'; entryProvider?: string } = {}
) {
  const harness = createHarness({ 'model.options': options(), ...results })
  const handle = fakeLease(sessionId)
  const selector = createAuthSelector({
    deps: harness.deps,
    entry: entry.entry ?? 'login',
    lease: handle.lease,
    ...(entry.entryProvider ? { entryProvider: entry.entryProvider } : {})
  })

  return {
    handle,
    harness,
    screen: () => harness.draw(selector.component),
    selector,
    view: selector.focus as { handleInput(data: string): void }
  }
}

const called = (harness: ReturnType<typeof createHarness>, method: string) =>
  harness.transport.calls.filter(call => call.method === method)

/** The id the stage minted for its sign-in: what its pushed steps must carry. */
const loginIdOf = (harness: ReturnType<typeof createHarness>) =>
  (called(harness, 'model.login')[0]?.params as { login_id: string }).login_id

function rpcError(code: number, message = 'nope', data?: unknown) {
  return rpcErrorFromFrame({ code, message, ...(data === undefined ? {} : { data }) })
}

/** pi's first two steps for a provider that signs in: "Sign in with an account",
 *  then the one provider under it. */
function enterUnconfigured(view: { handleInput(data: string): void }): void {
  view.handleInput(KEYS.enter)
  view.handleInput(KEYS.enter)
}

/** pi's first two steps for a provider that takes a key: "Sign in with an API
 *  key", then the one provider under it. */
function enterKeyed(view: { handleInput(data: string): void }): void {
  view.handleInput(KEYS.down)
  view.handleInput(KEYS.enter)
  view.handleInput(KEYS.enter)
}

describe("pi's /login lists", () => {
  it("opens on pi's authentication-method question, both options in pi's order", async () => {
    const { screen } = build()

    await settle()

    const shown = screen()

    expect(shown).toContain('Select authentication method:')
    expect(shown.indexOf('Sign in with an account')).toBeLessThan(shown.indexOf('Sign in with an API key'))
  })

  it("lists the key providers under the API-key method, with pi's status markers and the endpoint row last", async () => {
    const { screen, view } = build({ 'model.options': options([OPENAI, NEEDS_A_KEY, LOCAL]) })

    await settle()
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)

    const shown = screen()

    expect(shown).toContain('Select provider to configure:')
    expect(shown).toContain('✓ env: OPENAI_API_KEY')
    expect(shown).toContain('• unconfigured')
    expect(shown.indexOf('Local llama.cpp')).toBeLessThan(shown.indexOf('OpenAI Compatible'))
  })

  it('keeps the endpoint row last whatever the query matches', async () => {
    const ROUTER = providerRow({ authenticated: false, is_current: false, name: 'OpenRouter', slug: 'openrouter' })
    const AZURE = providerRow({
      authenticated: false,
      is_current: false,
      name: 'Azure OpenAI',
      slug: 'azure-openai-responses'
    })
    const { screen, view } = build({ 'model.options': options([OPENAI, ROUTER, AZURE]) })

    await settle()
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)
    for (const char of 'open') {
      view.handleInput(char)
    }

    const shown = screen()

    expect(shown.indexOf('Azure OpenAI')).toBeLessThan(shown.indexOf('OpenAI Compatible'))
  })
})

describe('declaring an OpenAI-compatible endpoint', () => {
  /** What the endpoint form is filled in with. The wire is not among the fields:
   *  the row declares the one a relay or a self-hosted server implements. */
  const TYPED = {
    api_key: 'sk-typed-into-the-form',
    base_url: 'http://127.0.0.1:8000/v1',
    provider: 'my-vllm'
  }

  /** The row the gateway answers a declaration with: configured, and serving
   *  what the endpoint published. */
  const DECLARED = declaredRow({
    api: 'openai-completions',
    authenticated: true,
    base_url: TYPED.base_url,
    model_labels: { 'my-vllm/qwen3-32b': { label: 'qwen3-32b' }, 'my-vllm/qwen3-8b': { label: 'qwen3-8b' } },
    models: ['my-vllm/qwen3-32b', 'my-vllm/qwen3-8b'],
    name: 'my-vllm',
    slug: 'my-vllm',
    total_models: 2
  })

  /** The API-key method → the last row of its list, which is this form. */
  function openTheForm(view: { handleInput(data: string): void }): void {
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)
    for (const char of 'compatible') {
      view.handleInput(char)
    }
    view.handleInput(KEYS.enter)
  }

  /** The three fields that matter, in the order the form asks for them; the
   *  model ids are left empty, the way an OpenRouter-like relay is declared. */
  function fillTheForm(view: { handleInput(data: string): void }, modelIds = ''): void {
    for (const value of [TYPED.provider, TYPED.base_url, TYPED.api_key, modelIds]) {
      for (const char of value) {
        view.handleInput(char)
      }

      if (value !== modelIds) {
        view.handleInput(KEYS.tab)
      }
    }
  }

  it('asks for the address and the key, and reads the models from the endpoint', async () => {
    const { handle, harness, screen, view } = build({
      'model.declare_provider': { discovered: 2, provider: DECLARED },
      'model.options': options([OPENAI, LOCAL])
    })

    await settle()
    openTheForm(view)

    expect(screen()).toContain('Add an OpenAI-compatible endpoint')
    expect(screen()).toContain('Model ids')
    expect(screen()).toContain('discovered from GET /models')

    fillTheForm(view)
    view.handleInput(KEYS.enter)
    await settle(10)

    expect(called(harness, 'model.declare_provider')[0]?.params).toEqual({
      api_key: TYPED.api_key,
      base_url: TYPED.base_url,
      model: '',
      provider: TYPED.provider,
      session_id: 'tui:abc'
    })
    // Connected: pi's sentence, and the selector closes on it.
    expect(harness.printed.map(line => line.text)).toContain(`Logged in to my-vllm (2 models from ${TYPED.base_url})`)
    expect(handle.closes).toEqual(['done'])
    expect(harness.printed.some(line => line.text.includes(TYPED.api_key))).toBe(false)
  })

  it('sends ids typed by hand for an endpoint that publishes none', async () => {
    const { harness, view } = build({
      'model.declare_provider': { discovered: 0, provider: DECLARED },
      'model.options': options([OPENAI, LOCAL])
    })

    await settle()
    openTheForm(view)
    fillTheForm(view, 'qwen3-32b, qwen3-8b')
    view.handleInput(KEYS.enter)
    await settle(10)

    expect((called(harness, 'model.declare_provider')[0]?.params as { model: string }).model).toBe(
      'qwen3-32b, qwen3-8b'
    )
  })

  it('masks the key while it is typed', async () => {
    const { screen, view } = build({ 'model.options': options([OPENAI, LOCAL]) })

    await settle()
    openTheForm(view)
    fillTheForm(view)

    expect(screen()).not.toContain(TYPED.api_key)
    expect(screen()).toContain('•••')
  })

  it("stays on the form with the gateway's own sentence when it refuses", async () => {
    const refused = 'my-vllm lists no models (it publishes no models at GET /models). Name the ids it serves'
    const { harness, screen, view } = build({
      'model.declare_provider': rpcError(-32011, 'config_validation_error', { detail: refused }),
      'model.options': options([OPENAI, LOCAL])
    })

    await settle()
    openTheForm(view)
    fillTheForm(view)
    view.handleInput(KEYS.enter)
    await settle(10)

    expect(called(harness, 'model.declare_provider')).toHaveLength(1)
    expect(screen()).toContain(refused)
    expect(screen()).toContain('Model ids')
    expect(screen()).not.toContain(TYPED.api_key)
  })
})

describe('key form', () => {
  it('requires a key when nothing in the environment supplies one', async () => {
    const { harness, screen, view } = build({ 'model.options': options([NEEDS_A_KEY], '') })

    await settle()
    enterKeyed(view)

    expect(screen()).toContain('Connect OpenAI')

    // Straight to Save with nothing typed.
    view.handleInput(KEYS.enter)
    await settle()

    expect(called(harness, 'model.save_key')).toHaveLength(0)
    expect(screen()).toContain('needs an API key')
  })

  it('masks what is typed and never writes it to the screen', async () => {
    const unconfigured = providerRow({ authenticated: false, is_current: false })
    const { screen, view } = build({ 'model.options': options([unconfigured], '') })

    await settle()
    enterKeyed(view)

    for (const char of 'sk-secret-123') {
      view.handleInput(char)
    }

    expect(screen()).not.toContain('sk-secret-123')
    expect(screen()).toContain('•••')
  })

  it('requires the address and the wire together for a provider the config declares', async () => {
    const { harness, screen, view } = build({ 'model.options': options([LOCAL], '') })

    await settle()
    enterKeyed(view)

    expect(screen()).toContain('Base URL')
    expect(screen()).toContain('optional for this provider')

    view.handleInput(KEYS.enter)
    await settle()

    expect(called(harness, 'model.save_key')).toHaveLength(0)
    expect(screen()).toContain('needs a base URL')

    view.handleInput(KEYS.tab)

    for (const char of 'http://127.0.0.1:8080') {
      view.handleInput(char)
    }

    view.handleInput(KEYS.enter)
    await settle()

    expect(called(harness, 'model.save_key')).toHaveLength(0)
    expect(screen()).toContain('needs the wire it serves')
  })

  it("sends the key, the address and the wire, then closes with pi's sentence when the provider authenticates", async () => {
    const { handle, harness, view } = build({
      'model.options': options([LOCAL], ''),
      'model.save_key': { provider: declaredRow({ authenticated: true }) }
    })

    await settle()
    enterKeyed(view)

    for (const char of 'sk-abc') {
      view.handleInput(char)
    }

    view.handleInput(KEYS.tab)

    for (const char of 'http://localhost:1234') {
      view.handleInput(char)
    }

    view.handleInput(KEYS.tab)

    for (const char of 'openai-completions') {
      view.handleInput(char)
    }

    view.handleInput(KEYS.enter)
    await settle(10)

    expect(called(harness, 'model.save_key')[0]?.params).toEqual({
      api: 'openai-completions',
      api_key: 'sk-abc',
      base_url: 'http://localhost:1234',
      session_id: 'tui:abc',
      slug: 'llama-cpp'
    })
    expect(harness.printed.map(line => line.text)).toContain('Logged in to Local llama.cpp')
    expect(handle.closes).toEqual(['done'])
  })

  it("asks one of pi's own for a key alone: it carries its own address and wire", async () => {
    const { harness, screen, view } = build({
      'model.options': options([NEEDS_A_KEY], ''),
      'model.save_key': { provider: providerRow({ authenticated: true }) }
    })

    await settle()
    enterKeyed(view)

    expect(screen()).not.toContain('Base URL')

    for (const char of 'sk-abc') {
      view.handleInput(char)
    }

    view.handleInput(KEYS.enter)
    await settle(10)

    expect(called(harness, 'model.save_key')[0]?.params).toEqual({
      api_key: 'sk-abc',
      session_id: 'tui:abc',
      slug: 'openai'
    })
  })

  it('keeps the form with a warning when the gateway does not accept the credential', async () => {
    const { handle, screen, view } = build({
      'model.options': options([NEEDS_A_KEY], ''),
      'model.save_key': { provider: providerRow({ authenticated: false, warning: 'the key was refused' }) }
    })

    await settle()
    enterKeyed(view)

    for (const char of 'sk-abc') {
      view.handleInput(char)
    }

    view.handleInput(KEYS.enter)
    await settle(10)

    expect(screen()).toContain('the key was refused')
    expect(handle.closes).toEqual([])
  })
})

describe('sign-in', () => {
  // pi offers this one a sign-in and nothing else, so there is no method to
  // choose: entering it starts the flow.
  const oauth = providerRow({
    auth_methods: ['oauth'],
    auth_type: 'oauth',
    authenticated: false,
    is_current: false,
    key_label: null,
    name: 'Claude'
  })
  const connected = providerRow({
    auth_methods: ['oauth'],
    auth_type: 'oauth',
    authenticated: true,
    is_current: false,
    key_label: null,
    name: 'Claude'
  })

  /** pi's own login-method menu, as `model.login` pushes it. */
  const menu = (loginId: string) => ({
    login_id: loginId,
    provider: 'openai',
    step: {
      message: 'Select OpenAI Codex login method:',
      options: [
        { id: 'browser', label: 'Browser login (default)' },
        { id: 'device_code', label: 'Device code login (headless)' }
      ],
      type: 'select' as const
    }
  })

  const deviceCode = (loginId: string) => ({
    login_id: loginId,
    provider: 'openai',
    step: { type: 'device_code' as const, userCode: 'ABCD-1234', verificationUri: 'https://example.test/device' }
  })

  /** A sign-in the test finishes by hand, so the stage can be driven first. */
  function pendingLogin() {
    let settle: (value: unknown) => void = () => {}
    const answered = new Promise(resolve => {
      settle = resolve
    })

    return {
      finish: (row: Record<string, unknown> = connected) => settle({ login_id: 'login-1', provider: row }),
      login: () => answered
    }
  }

  function startLogin(results: Record<string, unknown> = {}) {
    const built = build({ 'model.options': options([oauth], ''), ...results })

    return built
  }

  it('runs the sign-in here rather than handing over the terminal', async () => {
    const pending = pendingLogin()
    const { harness, screen, view } = startLogin({ 'model.login': pending.login })

    await settle()
    enterUnconfigured(view)
    await settle()

    expect(screen()).toContain('Sign in to Claude')
    expect(called(harness, 'model.login')[0]?.params).toEqual({
      login_id: expect.any(String),
      provider: 'openai',
      session_id: 'tui:abc'
    })
  })

  it("shows pi's own method menu from a pushed step and answers with the default", async () => {
    const pending = pendingLogin()
    const { harness, screen, view } = startLogin({
      'model.login': pending.login,
      'model.login_answer': { answered: true }
    })

    await settle()
    enterUnconfigured(view)
    await settle()
    harness.pushLoginStep(menu(loginIdOf(harness)))

    expect(screen()).toContain('Select OpenAI Codex login method:')
    expect(screen()).toContain('Browser login (default)')
    expect(screen()).toContain('Device code login (headless)')

    // Enter takes the first option, which is pi's own default.
    view.handleInput(KEYS.enter)
    await settle()

    expect(called(harness, 'model.login_answer')[0]?.params).toEqual({
      answer: 'browser',
      login_id: loginIdOf(harness)
    })
  })

  it('answers the menu with the option the arrows moved to', async () => {
    const pending = pendingLogin()
    const { harness, view } = startLogin({
      'model.login': pending.login,
      'model.login_answer': { answered: true }
    })

    await settle()
    enterUnconfigured(view)
    await settle()
    harness.pushLoginStep(menu(loginIdOf(harness)))
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)
    await settle()

    expect(called(harness, 'model.login_answer')[0]?.params).toEqual({
      answer: 'device_code',
      login_id: loginIdOf(harness)
    })
  })

  it("shows the device code and its URL the way pi's own dialog does", async () => {
    const pending = pendingLogin()
    const { harness, screen, view } = startLogin({ 'model.login': pending.login })

    await settle()
    enterUnconfigured(view)
    await settle()
    harness.pushLoginStep(deviceCode(loginIdOf(harness)))

    const shown = screen()

    // pi's three lines, in pi's order and with pi's words.
    expect(shown).toContain('https://example.test/device')
    expect(shown).toContain(process.platform === 'darwin' ? 'Cmd+click to open' : 'Ctrl+click to open')
    expect(shown).toContain('Enter code: ABCD-1234')
    expect(shown).toContain('Waiting for authentication...')
    expect(shown.indexOf('https://example.test/device')).toBeLessThan(shown.indexOf('Enter code:'))
  })

  it("shows a sign-in URL and its instructions the way pi's dialog does", async () => {
    const pending = pendingLogin()
    const { harness, screen, view } = startLogin({ 'model.login': pending.login })

    await settle()
    enterUnconfigured(view)
    await settle()
    harness.pushLoginStep({
      login_id: loginIdOf(harness),
      provider: 'openai',
      step: { instructions: 'Finish in the browser', type: 'auth_url' as const, url: 'https://auth.test/start' }
    })

    const shown = screen()

    expect(shown).toContain('https://auth.test/start')
    expect(shown).toContain(process.platform === 'darwin' ? 'Cmd+click to open' : 'Ctrl+click to open')
    expect(shown).toContain('Finish in the browser')
  })

  it('asks for a pasted code when the browser callback did not win', async () => {
    const pending = pendingLogin()
    const { harness, screen, view } = startLogin({
      'model.login': pending.login,
      'model.login_answer': { answered: true }
    })

    await settle()
    enterUnconfigured(view)
    await settle()
    harness.pushLoginStep({
      login_id: loginIdOf(harness),
      provider: 'openai',
      step: { message: 'Paste the authorization code', placeholder: 'code', type: 'manual_code' as const }
    })

    expect(screen()).toContain('Paste the authorization code')

    for (const char of 'xyz') {
      view.handleInput(char)
    }

    view.handleInput(KEYS.enter)
    await settle()

    expect(called(harness, 'model.login_answer')[0]?.params).toEqual({ answer: 'xyz', login_id: loginIdOf(harness) })
  })

  it("closes with pi's sentence once the provider is connected", async () => {
    const pending = pendingLogin()
    const { handle, harness, view } = startLogin({
      'model.login': pending.login,
      'model.login_answer': { answered: true }
    })

    await settle()
    enterUnconfigured(view)
    await settle()
    harness.pushLoginStep(menu(loginIdOf(harness)))
    view.handleInput(KEYS.enter)
    await settle()
    pending.finish()
    await settle(8)

    // pi's own sentence for a finished sign-in, and the selector closes on it.
    expect(harness.printed.map(line => line.text)).toContain('Logged in to Claude')
    expect(handle.closes).toEqual(['done'])
  })

  it('cancels the sign-in on escape and goes back to the list', async () => {
    const pending = pendingLogin()
    const { harness, screen, view } = startLogin({
      'model.login': pending.login,
      'model.login_cancel': { cancelled: true }
    })

    await settle()
    enterUnconfigured(view)
    await settle()
    harness.pushLoginStep(menu(loginIdOf(harness)))
    view.handleInput(KEYS.escape)
    await settle()

    expect(called(harness, 'model.login_cancel')[0]?.params).toEqual({ login_id: loginIdOf(harness) })
    expect(screen()).toContain('Select provider to configure:')
  })

  it('cancels a sign-in no step has arrived for yet, and ignores the step when it comes', async () => {
    const pending = pendingLogin()
    const { harness, screen, view } = startLogin({
      'model.login': pending.login,
      'model.login_cancel': { cancelled: true }
    })

    await settle()
    enterUnconfigured(view)
    await settle()
    // Escape before any step: the id is this stage's own, so there is a flow
    // to cancel by name already.
    view.handleInput(KEYS.escape)
    await settle()

    expect(called(harness, 'model.login_cancel')[0]?.params).toEqual({ login_id: loginIdOf(harness) })

    harness.pushLoginStep(menu(loginIdOf(harness)))
    await settle()

    expect(called(harness, 'model.login_cancel')).toHaveLength(1)
    expect(screen()).not.toContain('Select OpenAI Codex login method:')
  })

  it("shows only its own sign-in's steps", async () => {
    // A sign-in left on Esc earlier for the same provider may still push: its
    // steps carry another id and are not this stage's to show or to cancel.
    const pending = pendingLogin()
    const { harness, screen, view } = startLogin({ 'model.login': pending.login })

    await settle()
    enterUnconfigured(view)
    await settle()
    harness.pushLoginStep(menu('somebody-elses'))

    expect(screen()).not.toContain('Select OpenAI Codex login method:')

    harness.pushLoginStep(menu(loginIdOf(harness)))

    expect(screen()).toContain('Select OpenAI Codex login method:')
  })

  it('cancels the sign-in when the picker is closed over it', async () => {
    const pending = pendingLogin()
    const { harness, selector, view } = startLogin({
      'model.login': pending.login,
      'model.login_cancel': { cancelled: true }
    })

    await settle()
    enterUnconfigured(view)
    await settle()
    harness.pushLoginStep(menu(loginIdOf(harness)))
    // Closed from outside -- Ctrl+C on the app, another selector taking the
    // slot -- rather than by Esc in the view.
    selector.dispose()
    await settle()

    expect(called(harness, 'model.login_cancel')[0]?.params).toEqual({ login_id: loginIdOf(harness) })
  })

  it("opens the browser for a sign-in URL, as pi's dialog does, and not for a device code", async () => {
    const pending = pendingLogin()
    const { harness, view } = startLogin({ 'model.login': pending.login })

    await settle()
    enterUnconfigured(view)
    await settle()
    harness.pushLoginStep(deviceCode(loginIdOf(harness)))

    // A device code is entered on another machine; nothing to open here.
    expect(harness.opened).toEqual([])

    harness.pushLoginStep({
      login_id: loginIdOf(harness),
      provider: 'openai',
      step: { type: 'auth_url' as const, url: 'https://auth.test/start' }
    })

    expect(harness.opened).toEqual(['https://auth.test/start'])
  })

  it("shows the gateway's own sentence on the provider list when a sign-in fails", async () => {
    const { harness, screen, view } = startLogin({
      'model.login': rpcError(-32011, 'config_validation_error', {
        detail: 'the sign-in did not finish (login_failed).'
      })
    })

    await settle()
    enterUnconfigured(view)
    await settle(8)

    // pi's own sentence, with the gateway's own detail after it.
    expect(screen()).toContain('Failed to login to Claude: the sign-in did not finish (login_failed).')
    expect(called(harness, 'model.login')).toHaveLength(1)
  })
})

describe("pi's authentication-method step", () => {
  /** One of pi's providers that takes either a subscription or a key. */
  const both = providerRow({
    auth_methods: ['oauth', 'key'],
    auth_type: 'key',
    authenticated: false,
    is_current: false,
    key_env: null,
    key_label: 'xAI API key',
    login_label: 'Sign in with SuperGrok or X Premium',
    name: 'xAI',
    slug: 'xai'
  })

  /** `/login xai`: pi asks the provider's own question when it offers both. */
  const named = (results: Record<string, unknown>, provider = 'xai') =>
    build(results, 'session', 'tui:abc', { entryProvider: provider })

  it("shows pi's two options in pi's order, the subscription first", async () => {
    const { screen } = named({ 'model.options': options([both], '') })

    await settle()

    const shown = screen()

    expect(shown).toContain('Select authentication method for xAI:')
    // pi's own label for this provider, not a sentence of ours. Cut where
    // pi-tui's own list cuts a long label, which is where pi's own cuts it too.
    expect(shown).toContain('Sign in with SuperGrok or X Pr')
    expect(shown).toContain('Sign in with an API key')
    expect(shown.indexOf('Sign in with SuperGrok')).toBeLessThan(shown.indexOf('Sign in with an API key'))
  })

  it('takes the first option into the sign-in and the second into the key form', async () => {
    const pending = new Promise(() => {})
    const first = named({ 'model.login': () => pending, 'model.options': options([both], '') })

    await settle()
    first.view.handleInput(KEYS.enter)
    await settle()

    expect(first.screen()).toContain('Sign in to xAI')

    const second = named({ 'model.options': options([both], '') })

    await settle()
    second.view.handleInput(KEYS.down)
    second.view.handleInput(KEYS.enter)
    await settle()

    expect(second.screen()).toContain('Connect xAI')
    expect(second.screen()).toContain('API key')
  })

  it('does not ask when pi offers one way in', async () => {
    const keyOnly = providerRow({ authenticated: false, is_current: false, key_env: null, needs_api_key: true })
    const { screen } = named({ 'model.options': options([keyOnly], '') }, 'openai')

    await settle()

    expect(screen()).toContain('Connect OpenAI')
    expect(screen()).not.toContain('Select authentication method')
  })
})

describe("pi's /login", () => {
  const both = providerRow({
    auth_methods: ['oauth', 'key'],
    auth_type: 'key',
    authenticated: false,
    is_current: false,
    key_env: null,
    login_label: 'Sign in with SuperGrok or X Premium',
    name: 'xAI',
    slug: 'xai'
  })
  const codex = providerRow({
    auth_methods: ['oauth'],
    auth_type: 'oauth',
    authenticated: false,
    is_current: false,
    key_label: null,
    name: 'OpenAI Codex',
    slug: 'openai-codex'
  })

  const openLogin = (results: Record<string, unknown> = {}, provider?: string) =>
    build({ 'model.options': options([both, codex], ''), ...results }, 'session', 'tui:abc', {
      entry: 'login',
      ...(provider ? { entryProvider: provider } : {})
    })

  it("asks pi's question first, then lists the providers of that method", async () => {
    const { screen, view } = openLogin()

    await settle()

    expect(screen()).toContain('Select authentication method:')
    expect(screen()).toContain('Sign in with an account')
    expect(screen()).toContain('Sign in with an API key')

    view.handleInput(KEYS.enter)
    await settle()

    // pi's own title, and only the providers that sign in.
    expect(screen()).toContain('Select provider to configure:')
    expect(screen()).toContain('OpenAI Codex')
    expect(screen()).toContain('xAI')
    // Filtered to one method, so pi shows no method label.
    expect(screen()).not.toContain('[subscription]')
  })

  it("offers our own endpoint under the key method, and pi's providers with it", async () => {
    const { screen, view } = openLogin()

    await settle()
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)
    await settle()

    expect(screen()).toContain('Select provider to configure:')
    expect(screen()).toContain('xAI')
    // The one thing here that is not pi's.
    expect(screen()).toContain('OpenAI Compatible')
    // Codex takes no key, so pi does not list it under this method.
    expect(screen()).not.toContain('OpenAI Codex')
  })

  it('goes straight to the sign-in for a provider it was given', async () => {
    const pending = new Promise(() => {})
    const { harness, screen } = openLogin({ 'model.login': () => pending }, 'openai-codex')

    await settle()

    expect(screen()).toContain('Sign in to OpenAI Codex')
    expect(called(harness, 'model.login')[0]?.params).toEqual({
      login_id: expect.any(String),
      provider: 'openai-codex',
      session_id: 'tui:abc'
    })
  })

  it("asks pi's question for a named provider that offers both", async () => {
    const { screen } = openLogin({}, 'xai')

    await settle()

    expect(screen()).toContain('Select authentication method for xAI:')
  })

  it('lists a provider the gateway could not ask pi about under its own way in', async () => {
    // No model service to read pi's methods from: the row carries none, and its
    // own `auth_type` is the one way in the gate would judge a key by. The same
    // rule the wizard applies, so both lists hold the same rows offline.
    const unasked = providerRow({
      auth_methods: [],
      auth_type: 'key',
      authenticated: false,
      is_current: false,
      key_env: null,
      name: 'DeepSeek',
      slug: 'deepseek'
    })
    const { screen, view } = build({ 'model.options': options([unasked, codex], '') })

    await settle()
    view.handleInput(KEYS.down)
    view.handleInput(KEYS.enter)
    await settle()

    expect(screen()).toContain('Select provider to configure:')
    expect(screen()).toContain('DeepSeek')
    expect(screen()).not.toContain('OpenAI Codex')
  })
})

describe("pi's /logout", () => {
  const signedIn = providerRow({
    auth_methods: ['oauth'],
    auth_type: 'oauth',
    authenticated: true,
    is_current: false,
    key_label: null,
    name: 'OpenAI Codex',
    slug: 'openai-codex'
  })
  const keyed = providerRow({ authenticated: true, is_current: false, key_env: null, name: 'OpenAI', slug: 'openai' })

  const openLogout = (results: Record<string, unknown> = {}, rows = [signedIn]) =>
    build({ 'model.options': options(rows, ''), ...results }, 'session', 'tui:abc', { entry: 'logout' })

  it('forgets the chosen credential and says so the way pi says it', async () => {
    const { handle, harness, view } = openLogout({ 'model.logout': { forgotten: true } })

    await settle()
    view.handleInput(KEYS.enter)
    await settle()

    // pi's logout removes the credential, not the provider: `model.logout`, not
    // `model.disconnect`.
    expect(called(harness, 'model.logout')[0]?.params).toEqual({ session_id: 'tui:abc', slug: 'openai-codex' })
    expect(called(harness, 'model.disconnect')).toHaveLength(0)
    expect(harness.printed.map(line => line.text)).toContain('Logged out of OpenAI Codex')
    expect(handle.closes).toEqual(['done'])
  })

  it("lists stored keys as well as sign-ins, the way pi's auth store does", async () => {
    const { screen } = openLogout({}, [signedIn, keyed])

    await settle()

    const shown = screen()

    expect(shown).toContain('Select provider to logout:')
    expect(shown).toContain('OpenAI Codex')
    expect(shown).toContain('signed in')
    expect(shown).toContain('API key')
  })

  it("leaves out a key the environment supplies, and says pi's sentence when nothing is stored", async () => {
    // Nothing is stored for it, so there is nothing a logout could forget.
    const fromEnv = providerRow({ authenticated: true, is_current: false, key_env: 'OPENAI_API_KEY' })
    const { screen } = openLogout({}, [fromEnv])

    await settle()

    expect(screen()).toContain('No providers logged in. Use /login first.')
    expect(screen()).not.toContain('OpenAI')
  })

  it('stays on the list when the gateway had nothing to forget, or refused', async () => {
    const { handle, harness, screen, view } = openLogout({ 'model.logout': { forgotten: false } })

    await settle()
    view.handleInput(KEYS.enter)
    await settle()

    expect(screen()).toContain('nothing is stored for OpenAI Codex')
    expect(handle.closes).toEqual([])

    harness.transport.results['model.logout'] = rpcError(-32011, 'config_validation_error', {
      detail: 'could not sign OpenAI Codex out: the store could not be written'
    })
    view.handleInput(KEYS.enter)
    await settle()

    expect(screen()).toContain('could not sign OpenAI Codex out')
    expect(handle.closes).toEqual([])
  })
})

describe('leaving the picker', () => {
  it('closes on Ctrl+C from the first stage without calling anything', async () => {
    const { handle, harness, view } = build()

    await settle()
    view.handleInput(KEYS.ctrlC)

    expect(handle.closes).toEqual(['cancel'])
    expect(called(harness, 'config.set')).toHaveLength(0)
  })

  it('clears a typed secret when the selector is disposed', async () => {
    const unconfigured = providerRow({ authenticated: false, is_current: false })
    const { screen, selector, view } = build({ 'model.options': options([unconfigured], '') })

    await settle()
    enterKeyed(view)

    for (const char of 'sk-abc') {
      view.handleInput(char)
    }

    expect(screen()).toContain('•')
    selector.dispose()
    expect(screen()).not.toContain('•')
  })
})
