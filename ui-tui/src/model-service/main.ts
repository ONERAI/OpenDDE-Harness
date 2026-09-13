// The model service: pi-ai in a child process, driven over stdio.
//
// Python keeps the agent loop and speaks JSON lines to this process; the
// model layer is `@earendil-works/pi-ai` unmodified. See protocol.ts for the
// wire shape.
//
// Auth starts as whatever pi-ai resolves on its own (environment API keys
// through the default auth context). A `configure` request replaces that with
// this project's own configuration: a persistent credential store at a named
// path, the providers pi has no built-in factory for, and static keys for the
// ones it does. `login` runs pi's own OAuth flow over the same pipe and
// `login_answer` hands it what it asked for; `logout` deletes what it stored.
//
// `models` answers what the *configured* set serves; `providers` how each of
// them can be signed in to, in pi's own words; `catalog` answers what pi's own
// built-in rows say about one bare model id, whoever is configured, which is
// how a declared relay's model is sized before there is a configuration at all.
//
// Retry is pi's too: a `stream` request carrying a budget runs
// `retryAssistantCall` around the whole attempt, so a transient failure is
// announced as a `retry` event and the next attempt's events follow it. Only
// the last attempt's `done` or `error` reaches the caller. A request may also
// bound each attempt per gap (`timeouts`), which is what puts a stalled stream
// under that same budget rather than leaving it the one failure nothing
// repeats.
//
//   OPENDDE_MODEL_SERVICE_FAUX=1   also register four scripted providers,
//                                  `faux/echo` (instant), `faux-slow/echo`
//                                  (20 tokens/s), `faux-codex/echo` (shaped
//                                  like a Codex Responses model, so the
//                                  compaction path is reachable offline) and
//                                  `faux-flaky/echo` (fails to order, so the
//                                  retry path is), plus the `debug` request.
//                                  For offline tests.

import type {
  Api,
  AssistantMessage,
  Context,
  FauxProviderHandle,
  FauxResponseStep,
  Model,
  ModelCost,
  MutableModels,
  SimpleStreamOptions
} from '@earendil-works/pi-ai'

import {
  fauxAssistantMessage,
  fauxProvider,
  fauxText,
  fauxThinking,
  fauxToolCall,
  ModelsError
} from '@earendil-works/pi-ai'
import { registerBunOAuthFlows } from '@earendil-works/pi-ai/bun-oauth'
import { builtinModels } from '@earendil-works/pi-ai/providers/all'
import { createInterface } from 'node:readline'

import type { FileCredentialStore } from './credential-store.js'
import type { FileModelsStore } from './models-store.js'
import type {
  CompactParams,
  ConfigureParams,
  RefreshParams,
  LoginParams,
  LogoutParams,
  Reply,
  RequestId,
  StreamParams
} from './protocol.js'

import { ceilingFor } from './ceiling.js'
import { fauxCompaction, fauxResponsesPayload, replayOnPayload, runCompaction } from './compaction.js'
import { configure } from './configure.js'
import { staleCatalogs } from './discover.js'
import { answerLogin, runLogin } from './login.js'
import { BadRequest, parseRequest } from './protocol.js'
import { classify, errorMessage, streamWithRetry } from './stream.js'

let models: MutableModels = builtinModels()
/**
 * pi's own rows, kept apart from `models` so `catalog` survives a `configure`.
 *
 * `configure` replaces the provider set with this project's own -- built-in
 * vendors get a key, everything else is declared outright -- so after one the
 * configured collection can no longer answer "what does pi say about this
 * model id". That question is asked *while* the configuration is being built
 * (a declared relay serving `deepseek-chat` is sized from the vendor's row),
 * so it is answered from a collection nothing reconfigures. Built on first ask.
 */
let catalogue: MutableModels | undefined
let store: FileCredentialStore | undefined
let modelsStore: FileModelsStore | undefined
const faux = new Map<string, FauxProviderHandle>()
const running = new Map<RequestId, AbortController>()
/** Lines still queued plus async replies still owed. Nothing exits while this is above zero. */
let busy = 0
/** Faux mode only: how many compactions have been scripted, and the last patched payload. */
let fauxCompactions = 0
let lastPayload: { id: RequestId; payload: unknown; skipped?: string } | undefined

function write(reply: Reply): void {
  process.stdout.write(`${JSON.stringify(reply)}\n`)
}

/** One model as the Python side reads it: every fact it decides anything with. */
interface ModelRow {
  contextWindow: number
  /** pi's own rates, per *million* tokens (`calculateCost` divides). */
  cost: ModelCost
  id: string
  input: ('image' | 'text')[]
  maxTokens: number
  name: string
  provider: string
  reasoning: boolean
}

function row(model: Model<Api>): ModelRow {
  return {
    contextWindow: model.contextWindow,
    cost: model.cost,
    id: model.id,
    input: model.input,
    maxTokens: model.maxTokens,
    name: model.name,
    provider: model.provider,
    reasoning: model.reasoning
  }
}

/**
 * How one provider can be signed in to, as pi declares it.
 *
 * pi keeps both methods on the provider (`auth.apiKey`, `auth.oauth`) with the
 * labels it shows for each: `oauth.loginLabel` is the sentence its own selector
 * puts on the subscription option ("Sign in with SuperGrok or X Premium"), and
 * `apiKey.name` is what it calls the key ("xAI API key"). They travel as pi
 * wrote them: a caller offering the choice offers pi's words, not a second set
 * of ours.
 *
 * `keyLogin` is whether a key can be typed at all. pi's ambient providers -- the
 * AWS chain, Google's application default credentials -- carry `apiKey` auth
 * with no `login`, because there is no string for anyone to paste.
 */
interface ProviderAuthRow {
  id: string
  /** pi's own two method names, in pi's own spelling. */
  methods: ('api_key' | 'oauth')[]
  keyLogin: boolean
  /** `apiKey.name`, e.g. "xAI API key". */
  keyName?: string
  /** `oauth.loginLabel`, e.g. "Sign in with SuperGrok or X Premium". */
  loginLabel?: string
  name: string
  /** `oauth.name`, e.g. "xAI (Grok/X subscription)". */
  oauthName?: string
  /** pi's own flag: access through the sign-in is backed by a subscription. */
  subscription?: boolean
}

function providerAuthRows(): ProviderAuthRow[] {
  return models.getProviders().map(provider => {
    const { apiKey, oauth } = provider.auth
    return {
      id: provider.id,
      keyLogin: typeof apiKey?.login === 'function',
      methods: [...(oauth ? (['oauth'] as const) : []), ...(apiKey ? (['api_key'] as const) : [])],
      name: provider.name,
      ...(apiKey?.name ? { keyName: apiKey.name } : {}),
      ...(oauth?.loginLabel ? { loginLabel: oauth.loginLabel } : {}),
      ...(oauth?.name ? { oauthName: oauth.name } : {}),
      ...(oauth?.isSubscription === undefined ? {} : { subscription: oauth.isSubscription })
    }
  })
}

/** Every built-in row for one bare model id, across pi's own providers. */
function catalogRows(id: string): ModelRow[] {
  catalogue ??= builtinModels()
  return catalogue
    .getModels()
    .filter(model => model.id === id)
    .map(row)
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function lastUserText(context: Context): string {
  for (let i = context.messages.length - 1; i >= 0; i--) {
    const message = context.messages[i]
    if (message.role !== 'user') {
      continue
    }
    if (typeof message.content === 'string') {
      return message.content
    }
    return message.content.map(block => (block.type === 'text' ? block.text : '')).join('')
  }
  return ''
}

/** A scripted reply exercising every event type: thinking, several text deltas, one tool call. */
function echoResponse(context: Context): AssistantMessage {
  const text = lastUserText(context)
  return fauxAssistantMessage(
    [
      fauxThinking(`The user said: ${text}. Reply by echoing it and calling the echo tool.`),
      fauxText(`Echo: ${text}. `.repeat(6)),
      fauxToolCall('echo', { text })
    ],
    { stopReason: 'toolUse' }
  )
}

function registerFaux(): void {
  const definitions = [{ id: 'echo', name: 'Faux echo', reasoning: true }]
  const fast = fauxProvider({ provider: 'faux', models: definitions })
  const slow = fauxProvider({ provider: 'faux-slow', models: definitions, tokensPerSecond: 20 })
  // A third one the compaction path recognizes: it gates on the model's api
  // and host, so only a Codex-shaped model reaches `compact` and `replay`.
  // The provider id stays faux, which is what keeps the real one intact.
  const codex = fauxProvider({ api: 'openai-codex-responses', models: definitions, provider: 'faux-codex' })
  for (const model of codex.models) {
    model.baseUrl = 'https://chatgpt.com/backend-api'
  }
  // A fourth that fails on request, which is the only way the retry loop is
  // reachable with nothing to call. See `fauxStep`.
  const flaky = fauxProvider({ models: definitions, provider: 'faux-flaky' })
  faux.clear()
  for (const handle of [fast, slow, codex, flaky]) {
    models.setProvider(handle.provider)
    faux.set(handle.provider.id, handle)
  }
}

/**
 * The faux script for one attempt: one response, consumed by that attempt.
 *
 * `faux-flaky` reads its prompt as the script -- `fail:<n>` fails the first n
 * attempts with wording pi's own classifier calls transient, `fatal` fails
 * every attempt with wording it does not, and the default is one transient
 * failure. Every other faux provider echoes, whichever attempt this is.
 */
function fauxStep(provider: string, context: Context, attempt: number): FauxResponseStep {
  if (provider !== 'faux-flaky') {
    return echoResponse(context)
  }
  const prompt = lastUserText(context)
  const fatal = prompt.includes('fatal')
  const failures = fatal ? Number.POSITIVE_INFINITY : Number(/fail:(\d+)/.exec(prompt)?.[1] ?? 1)
  if (attempt >= failures) {
    return echoResponse(context)
  }
  return fauxAssistantMessage([], {
    errorMessage: fatal ? 'invalid request: the tool schema is malformed' : '503 service unavailable',
    stopReason: 'error'
  })
}

/**
 * Resolve auth before the stream rather than inside it.
 *
 * pi's `streamSimple` runs auth resolution inside `lazyStream`, which flattens
 * whatever it throws into a message string (`api/lazy.js`
 * `createSetupErrorMessage`): the `ModelsError` and its code are gone by the
 * time the failure is an event. Asking first is what keeps the code. The
 * second resolve the stream then does is free -- an OAuth refresh runs under
 * the store lock and the fresh token is already committed.
 */
async function requireAuth(model: Model<Api>, provider: string, signal: AbortSignal): Promise<void> {
  if (!(await models.getAuth(model, { signal }))) {
    // The same refusal `applyAuth` would raise, raised where its code survives.
    throw new ModelsError('auth', `Provider is not configured: ${provider}`)
  }
}

async function stream(id: RequestId, params: StreamParams): Promise<void> {
  const model = models.getModel(params.provider, params.model)
  if (!model) {
    write({ error: { code: 'model_not_found', message: `no model ${params.provider}/${params.model}` }, id })
    return
  }
  // A model nobody sized: see ceiling.ts for what is sent and what is refused.
  const ceiling = ceilingFor(model, params.options?.maxTokens)
  if (ceiling.refusal) {
    write({ error: { code: 'no_max_tokens', message: ceiling.refusal }, id })
    return
  }
  const controller = new AbortController()
  running.set(id, controller)
  let ended = false
  try {
    await requireAuth(model, params.provider, controller.signal)
    // A replacement history from a previous `compact` replaces this request's
    // input, which pi asks for right before it sends: see compaction.ts.
    const onPayload = replayOnPayload(model, params.replay, params.context)
    // No signal here: each attempt is handed its own, so a deadline ends one
    // try rather than the request. See `streamWithRetry`.
    const options: SimpleStreamOptions = { ...(params.options ?? {}), ...(onPayload ? { onPayload } : {}) }
    if (onPayload && faux.has(params.provider)) {
      // The faux provider sends nothing, so nothing would call the hook. Run
      // it here instead, against the request pi would have built, so an
      // offline test can see the replay reach it. Only `faux-codex` is shaped
      // like a model that compacts; on the others the hook is right to decline.
      lastPayload = { id, payload: await onPayload(fauxResponsesPayload(model, params.context), model) }
    } else if (params.replay && faux.has(params.provider)) {
      lastPayload = {
        id,
        payload: null,
        skipped: `${params.provider}/${params.model} has no server-side compaction, so nothing was replayed`
      }
    }
    await streamWithRetry(id, params, model.contextWindow, controller.signal, write, (attempt, attemptSignal) => {
      // Scripted per attempt, not per request: the faux provider serves one
      // queued response per call, so a retried request needs one more.
      faux.get(params.provider)?.appendResponses([fauxStep(params.provider, params.context, attempt)])
      return models.streamSimple(ceiling.model, params.context, { ...options, signal: attemptSignal })
    })
    ended = true
  } catch (error) {
    if (!ended) {
      const failure = { error: errorMessage(params, error), reason: 'error' as const, type: 'error' as const }
      write({ event: classify(failure, model.contextWindow, error), id })
      ended = true
    }
  } finally {
    running.delete(id)
    exitIfDone()
  }
}

/**
 * `compact`: OpenAI's own server-side compaction, replying with the opaque
 * replacement history to send back as `replay` on later turns.
 *
 * The caller decides when this happens and what it covers -- this process
 * keeps no session state. A faux provider answers with a scripted history
 * instead, offline.
 */
async function compact(id: RequestId, params: CompactParams): Promise<void> {
  const controller = new AbortController()
  running.set(id, controller)
  try {
    const outcome = faux.has(params.provider)
      ? fauxCompaction(models, params, ++fauxCompactions)
      : await runCompaction(models, params, controller.signal)
    write({ id, result: { items: outcome.items, usage: outcome.usage } })
  } catch (error) {
    const bad = error instanceof BadRequest ? error : undefined
    write({ error: { code: bad?.code ?? 'compaction_failed', message: bad?.message ?? message(error) }, id })
  } finally {
    running.delete(id)
    exitIfDone()
  }
}

/**
 * Ask the declared endpoints what they serve. pi's own `Models.refresh`, paced
 * the way pi paces its catalogs: a list checked within the window is left as
 * it is unless `force` says otherwise (the moment an endpoint is declared).
 */
async function refresh(id: RequestId, params: RefreshParams): Promise<void> {
  busy++
  try {
    const dynamic = models
      .getProviders()
      .filter(provider => provider.refreshModels !== undefined)
      .map(provider => provider.id)
    const selected = params.providers ? dynamic.filter(candidate => params.providers?.includes(candidate)) : dynamic
    const providers = params.force ? selected : await staleCatalogs(modelsStore, selected)
    const result =
      providers.length === 0
        ? { aborted: false, errors: new Map<string, Error>() }
        : await models.refresh({ allowNetwork: true, force: true, providers })
    write({
      id,
      result: {
        aborted: result.aborted,
        errors: Object.fromEntries([...result.errors].map(([provider, error]) => [provider, message(error)]))
      }
    })
  } catch (error) {
    write({ error: { code: 'refresh_failed', message: message(error) }, id })
  } finally {
    busy--
    exitIfDone()
  }
}

async function applyConfigure(id: RequestId, params: ConfigureParams): Promise<void> {
  busy++
  try {
    const configured = await configure(params, id)
    models = configured.models
    store = configured.store
    modelsStore = configured.modelsStore
    // The lists the declared endpoints published last time, from the models
    // store: on screen at once, and asked for again only by `refresh`.
    await models.refresh({ allowNetwork: false })
    if (process.env.OPENDDE_MODEL_SERVICE_FAUX === '1') {
      registerFaux()
    }
    write({
      id,
      result: {
        models: models.getModels().length,
        providers: models
          .getProviders()
          .map(provider => provider.id)
          .sort()
      }
    })
  } catch (error) {
    const bad = error instanceof BadRequest ? error : new BadRequest('configure_failed', message(error), id)
    write({ error: { code: bad.code, message: bad.message }, id })
  } finally {
    busy--
    exitIfDone()
  }
}

/**
 * Run pi's own OAuth flow and persist what it returns.
 *
 * The flow itself, and how its prompts are answered, is login.ts; this is the
 * request's lifetime around it. The controller goes into `running`, so an
 * `abort` naming this id cancels the sign-in -- which is what a caller that
 * walked away from the menu sends.
 */
async function login(id: RequestId, params: LoginParams): Promise<void> {
  const controller = new AbortController()
  running.set(id, controller)
  try {
    await runLogin(id, params, models, controller.signal, write)
  } finally {
    running.delete(id)
    exitIfDone()
  }
}

/**
 * Forget one provider's stored credential -- the sign-out.
 *
 * Through the store rather than the file, so the delete is serialised with the
 * writes a refresh or a login may be making at the same moment, and so it is
 * the same writer that put the credential there. `forgotten` says whether
 * there was one, which is what lets a caller report "signed out" rather than
 * "nothing was signed in" without reading the file itself.
 */
async function logout(id: RequestId, params: LogoutParams): Promise<void> {
  busy++
  try {
    if (!store) {
      write({
        error: { code: 'not_configured', message: 'logout needs a configure first: there is no credential store' },
        id
      })
      return
    }
    const held = (await store.list()).some(entry => entry.providerId === params.provider)
    if (held) {
      await store.delete(params.provider)
    }
    write({ id, result: { forgotten: held, provider: params.provider } })
  } catch (error) {
    write({ error: { code: 'logout_failed', message: message(error) }, id })
  } finally {
    busy--
    exitIfDone()
  }
}

/** What is stored and what each provider would resolve, with no secret in either. */
async function auth(id: RequestId): Promise<void> {
  busy++
  try {
    const stored = store ? [...(await store.list())] : []
    const providers = await Promise.all(
      models.getProviders().map(async provider => {
        try {
          const check = await models.checkAuth(provider.id)
          return check
            ? { configured: true, id: provider.id, source: check.source, type: check.type }
            : { configured: false, id: provider.id }
        } catch (error) {
          return { configured: false, error: message(error), id: provider.id }
        }
      })
    )
    write({ id, result: { providers, stored } })
  } catch (error) {
    write({ error: { code: 'auth_failed', message: message(error) }, id })
  } finally {
    busy--
    exitIfDone()
  }
}

/**
 * One request. Returns a promise only for `configure`, which the line loop
 * waits on: what it replaces -- the provider set, the credential store -- is
 * what every later request reads, so those must not overtake it. Streams stay
 * concurrent, which is the whole point of multiplexing by id.
 */
function handle(line: string): Promise<void> | void {
  let request
  try {
    request = parseRequest(line)
  } catch (error) {
    const bad = error instanceof BadRequest ? error : new BadRequest('internal', String(error))
    write({ error: { code: bad.code, message: bad.message }, id: bad.id })
    return
  }
  switch (request.method) {
    case 'models':
      write({ id: request.id, result: models.getModels().map(row) })
      return
    case 'providers':
      write({ id: request.id, result: providerAuthRows() })
      return
    case 'refresh':
      void refresh(request.id, request.params)
      return
    case 'catalog':
      write({ id: request.id, result: catalogRows(request.params.id) })
      return
    case 'abort':
      running.get(request.params.id)?.abort()
      write({ id: request.id, result: { aborted: running.has(request.params.id) } })
      return
    case 'compact':
      void compact(request.id, request.params)
      return
    case 'debug':
      if (process.env.OPENDDE_MODEL_SERVICE_FAUX !== '1') {
        write({
          error: { code: 'not_faux', message: 'debug is answered only under OPENDDE_MODEL_SERVICE_FAUX=1' },
          id: request.id
        })
        return
      }
      write({ id: request.id, result: { lastPayload: lastPayload ?? null } })
      return
    case 'auth':
      void auth(request.id)
      return
    case 'configure':
      return applyConfigure(request.id, request.params)
    case 'login':
      void login(request.id, request.params)
      return
    case 'login_answer':
      write({ id: request.id, result: { answered: answerLogin(request.params.loginId, request.params.answer) } })
      return
    case 'logout':
      void logout(request.id, request.params)
      return
    case 'stream':
      void stream(request.id, request.params)
      return
  }
}

// pi loads each OAuth flow through a variable specifier, so that a bundler
// cannot follow the import into its Node-only callback servers and PKCE. This
// file *is* a bundle, and the import it cannot follow resolves next to the
// output -- where there is nothing -- so the first `login` fails on a missing
// module. Registering the statically-linked flows is pi's own answer to that;
// its name says Bun because a Bun binary hit this first.
registerBunOAuthFlows()

if (process.env.OPENDDE_MODEL_SERVICE_FAUX === '1') {
  registerFaux()
}

process.on('uncaughtException', error => {
  process.stderr.write(`model-service: uncaught ${error.stack ?? error}\n`)
})
process.on('unhandledRejection', reason => {
  process.stderr.write(`model-service: unhandled rejection ${String(reason)}\n`)
})

const lines = createInterface({ crlfDelay: Infinity, input: process.stdin })
// One line at a time, in the order they arrived: a `configure` holds the queue
// until it has replaced the provider set. Everything else settles at once.
let queue: Promise<unknown> = Promise.resolve()
lines.on('line', line => {
  if (line.trim().length === 0) {
    return
  }
  busy++
  queue = queue
    .then(() => handle(line))
    .catch(error => process.stderr.write(`model-service: handling failed ${String(error)}\n`))
    .finally(() => {
      busy--
      exitIfDone()
    })
})
// stdin closed: the parent is gone or done. Work already in flight finishes
// (its output is still wanted, e.g. a piped one-shot run), then exit.
let closing = false
function exitIfDone(): void {
  if (closing && running.size === 0 && busy === 0) {
    process.exit(0)
  }
}
lines.on('close', () => {
  closing = true
  exitIfDone()
})
