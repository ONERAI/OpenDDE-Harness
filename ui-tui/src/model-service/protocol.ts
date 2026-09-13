// The model service's wire contract: one JSON object per line, in both
// directions. Requests carry an `id` the reply echoes; a `stream` request is
// answered by one `event` line per pi-ai AssistantMessageEvent and ends with
// the `done` or `error` event. The events are pi-ai's own types, minus the
// `partial` field: that is the whole message so far, repeated on every delta,
// and the terminal event carries the full message anyway. An `error` event
// also carries `retryable`, pi's own classification of the failure, `overflow`,
// pi's own reading of whether the context window is what refused it, and `code`
// when the failure is a pi `ModelsError` -- "auth" for a credential that is
// missing, "oauth" for a refresh that failed.
//
// A `stream` request carrying `retry` runs pi's own retry loop inside the
// service: a failed attempt pi calls transient is announced by a `retry` event
// and followed by the next attempt's events, from a fresh `start`. Only the
// last attempt's `done` or `error` is written, so one request still ends once.
// `timeouts` bounds each attempt per gap; an expired one is a transient
// failure like any other, so the same retry budget covers a stalled stream.
//
// `login` is answered the same way: one `login_prompt` event per step of the
// OAuth flow, then the request's result. A step pi is waiting on an answer for
// arrives as an `ask` and is answered by a `login_answer` request naming that
// login -- the option id for a menu, the pasted code for the browser flow's
// fallback. `logout` is its opposite and takes one round trip: the provider's
// entry is deleted from the credential store.
//
// `compact` asks OpenAI for a server-side replacement history and replies with
// it in one result; a later `stream` carrying that history in `replay` is sent
// with it in place of the conversation. See compaction.ts.

import type {
  AssistantMessageEvent,
  AuthEvent,
  AuthPrompt,
  Context,
  SimpleStreamOptions,
  Tool
} from '@earendil-works/pi-ai'
import type { ResponseItem } from 'pi-codex-compact/src/compaction.ts'

export type RequestId = number | string

export type Request =
  | { id: RequestId; method: 'abort'; params: { id: RequestId } }
  | { id: RequestId; method: 'auth' }
  | { id: RequestId; method: 'catalog'; params: CatalogParams }
  | { id: RequestId; method: 'compact'; params: CompactParams }
  | { id: RequestId; method: 'configure'; params: ConfigureParams }
  | { id: RequestId; method: 'debug' }
  | { id: RequestId; method: 'login'; params: LoginParams }
  | { id: RequestId; method: 'login_answer'; params: LoginAnswerParams }
  | { id: RequestId; method: 'logout'; params: LogoutParams }
  | { id: RequestId; method: 'models' }
  | { id: RequestId; method: 'providers' }
  | { id: RequestId; method: 'refresh'; params: RefreshParams }
  | { id: RequestId; method: 'stream'; params: StreamParams }

/**
 * One model id to look up in pi's own built-in catalogue.
 *
 * Independent of `configure`: the rows are pi's, whoever is configured, which
 * is what makes this answerable while the configuration is still being built
 * (a declared provider's limits are read from the vendor row for the same
 * model id -- see `providers/pi_auth._model_entry`).
 */
export interface CatalogParams {
  /** The bare model id, as the endpoint serves it: no provider prefix. */
  id: string
}

export interface StreamParams {
  context: Context
  model: string
  options?: Pick<SimpleStreamOptions, 'maxTokens' | 'reasoning' | 'sessionId' | 'temperature' | 'toolChoice'>
  provider: string
  /** A replacement history from a previous `compact`, sent instead of the conversation. */
  replay?: ReplayHistory
  /** How many times a failed attempt may be run again. Absent is none. */
  retry?: RetryParams
  /** Per-gap deadlines for each attempt. Absent, or a missing half, is none. */
  timeouts?: TimeoutParams
}

/** The caller's retry budget for one request; pi's own policy is built from it. */
export interface RetryParams {
  maxRetries: number
}

/**
 * How long one attempt may go without an event before it is ended.
 *
 * Two gaps, not a total: `firstTokenMs` covers the silence before the first
 * event (a model with hidden reasoning thinks for minutes before it) and
 * `idleMs` every silence after. A stream that keeps delivering is never cut --
 * a long reply is not a fault. An expired budget fails the attempt with
 * wording pi's own classifier calls transient, so the retry budget covers it.
 */
export interface TimeoutParams {
  firstTokenMs?: number
  idleMs?: number
}

/** The opaque replacement history one `compact` returned, as it travels back. */
export interface ReplayHistory {
  items: ResponseItem[]
}

export interface CompactParams {
  context: Context
  model: string
  provider: string
  /** Thinking effort for the compaction request, so it mirrors the turns it replaces. */
  reasoning?: string
  /** An earlier replacement history, compacted again together with what followed it. */
  replay?: ReplayHistory
  sessionId?: string
  tools?: Tool[]
}

/** One model a custom provider serves. Everything but the id is optional; see main.ts for the defaults. */
/** pi's four per-million rates, for a model no catalogue prices. */
export interface ConfigureCost {
  cacheRead: number
  cacheWrite: number
  input: number
  output: number
}

export interface ConfigureModel {
  /** Overrides the provider's `api` for this one model (a relay serving two wires). */
  api?: string
  /**
   * pi's own compatibility overrides for this model, in pi's own shape
   * (`OpenAICompletionsCompat` and its siblings): what pi would otherwise
   * detect from the address. Passed through untouched -- the caller validates
   * the keys against the wire, and pi ignores what it has no field for.
   */
  compat?: Record<string, unknown>
  contextWindow?: number
  cost?: ConfigureCost
  id: string
  input?: ('image' | 'text')[]
  maxTokens?: number
  name?: string
  reasoning?: boolean
}

/** A provider pi-ai has no built-in factory for: a relay, a self-hosted server, a gateway. */
export interface ConfigureProvider {
  api: string
  apiKey?: string
  baseUrl?: string
  /** The compatibility block every model this endpoint publishes travels with (a declared row carries its own). */
  compat?: Record<string, unknown>
  headers?: Record<string, string>
  id: string
  models: ConfigureModel[]
  name?: string
}

/** Ask the dynamic providers -- the declared endpoints -- for their lists again. */
export interface RefreshParams {
  /** Fetch even a list checked within the freshness window. */
  force?: boolean
  /** Only these providers; every dynamic one when absent. */
  providers?: string[]
}

export interface ConfigureParams {
  /** Static keys for pi's built-in providers, by pi provider id. Never written to disk. */
  apiKeys: Record<string, string>
  /** Where the persistent credential store lives. */
  credentials: string
  providers: ConfigureProvider[]
}

export interface LoginParams {
  /** Answers pi's own login-method menu without a round trip: "device_code" or
   *  "browser". Absent forwards the menu as an `ask` and waits for a
   *  `login_answer`. */
  mode?: string
  provider: string
}

/** One answer for a login that is waiting on a prompt. */
export interface LoginAnswerParams {
  /** What pi asked for: the option id of a `select`, the code of a `manual_code`. */
  answer: string
  /** The id of the `login` request whose prompt this answers. */
  loginId: RequestId
}

/** Which provider to forget. Signing out is deleting its entry from the store. */
export interface LogoutParams {
  provider: string
}

/**
 * One step of a login flow: something to show (`notify`), or something pi
 * asked for (`ask`). An `ask` is answered by a `login_answer` request naming
 * the login -- see login.ts.
 */
export type LoginEvent = { ask: AuthPrompt; type: 'login_prompt' } | { notify: AuthEvent; type: 'login_prompt' }

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never

export type WireEvent = DistributiveOmit<AssistantMessageEvent, 'partial'> & {
  /** pi's `ModelsErrorCode` when a `ModelsError` is the failure; absent otherwise. */
  code?: string
  /** pi's `isContextOverflow` on the failed message: the window is what refused it. */
  overflow?: boolean
  retryable?: boolean
}

/**
 * One attempt failed and another is scheduled -- pi's own retry loop, as an
 * event. `attempt` is the retry about to run, 1-indexed, out of `maxAttempts`
 * retries; the initial call is neither. The next attempt's events follow from a
 * fresh `start`, so a caller showing live text discards what this one streamed.
 */
export interface RetryEvent {
  attempt: number
  delayMs: number
  errorMessage: string
  maxAttempts: number
  type: 'retry'
}

export type Reply =
  | { error: { code: string; message: string }; id: RequestId | null }
  | { event: LoginEvent | RetryEvent | WireEvent; id: RequestId }
  | { id: RequestId; result: unknown }

export class BadRequest extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly id: RequestId | null = null
  ) {
    super(message)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isId(value: unknown): value is RequestId {
  return (typeof value === 'number' && Number.isFinite(value)) || (typeof value === 'string' && value.length > 0)
}

function positive(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined
}

/** The same object without the keys nothing set, so the parsed request is the request. */
function defined<T extends object>(value: T): T {
  return Object.fromEntries(Object.entries(value).filter(([, v]) => v !== undefined)) as T
}

function strings(value: unknown): Record<string, string> {
  if (!isRecord(value)) {
    return {}
  }
  return Object.fromEntries(Object.entries(value).filter(([, v]) => typeof v === 'string')) as Record<string, string>
}

/**
 * A declared model's rates. All four or none: pi reads four zeroes as "no
 * price for this model", so a half-filled cost would report a wrong figure
 * rather than an unknown one.
 */
function parseCost(raw: unknown): ConfigureCost | undefined {
  if (!isRecord(raw)) {
    return undefined
  }
  const rate = (value: unknown): number =>
    typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0
  const cost = {
    cacheRead: rate(raw.cacheRead),
    cacheWrite: rate(raw.cacheWrite),
    input: rate(raw.input),
    output: rate(raw.output)
  }
  return cost.cacheRead || cost.cacheWrite || cost.input || cost.output ? cost : undefined
}

function parseModel(raw: unknown, id: RequestId, provider: string): ConfigureModel {
  if (!isRecord(raw) || typeof raw.id !== 'string' || raw.id.length === 0) {
    throw new BadRequest('invalid_params', `provider ${provider} declares a model without an id`, id)
  }
  const input = Array.isArray(raw.input)
    ? raw.input.filter((kind): kind is 'image' | 'text' => kind === 'text' || kind === 'image')
    : undefined
  return defined({
    api: typeof raw.api === 'string' ? raw.api : undefined,
    compat: isRecord(raw.compat) ? raw.compat : undefined,
    contextWindow: positive(raw.contextWindow),
    cost: parseCost(raw.cost),
    id: raw.id,
    input: input && input.length > 0 ? input : undefined,
    maxTokens: positive(raw.maxTokens),
    name: typeof raw.name === 'string' ? raw.name : undefined,
    reasoning: typeof raw.reasoning === 'boolean' ? raw.reasoning : undefined
  })
}

function parseProvider(raw: unknown, id: RequestId): ConfigureProvider {
  if (!isRecord(raw) || typeof raw.id !== 'string' || raw.id.length === 0) {
    throw new BadRequest('invalid_params', 'a configured provider carries a string id', id)
  }
  if (typeof raw.api !== 'string' || raw.api.length === 0) {
    throw new BadRequest('invalid_params', `provider ${raw.id} declares no api`, id)
  }
  if (!Array.isArray(raw.models)) {
    throw new BadRequest('invalid_params', `provider ${raw.id} declares no models array`, id)
  }
  return defined({
    api: raw.api,
    apiKey: typeof raw.apiKey === 'string' && raw.apiKey.length > 0 ? raw.apiKey : undefined,
    baseUrl: typeof raw.baseUrl === 'string' && raw.baseUrl.length > 0 ? raw.baseUrl : undefined,
    compat: isRecord(raw.compat) ? raw.compat : undefined,
    headers: isRecord(raw.headers) ? strings(raw.headers) : undefined,
    id: raw.id,
    models: raw.models.map(model => parseModel(model, id, raw.id as string)),
    name: typeof raw.name === 'string' ? raw.name : undefined
  })
}

/**
 * A replacement history as it came back over the wire. The items are OpenAI's
 * own, opaque to us beyond each carrying a `type`, so that is all we check.
 */
function parseReplay(raw: unknown, id: RequestId): ReplayHistory | undefined {
  if (raw === undefined) {
    return undefined
  }
  if (!isRecord(raw) || !Array.isArray(raw.items)) {
    throw new BadRequest('invalid_params', 'replay carries an `items` array', id)
  }
  if (!raw.items.every(item => isRecord(item) && typeof item.type === 'string')) {
    throw new BadRequest('invalid_params', 'every replay item is an object with a string `type`', id)
  }
  return { items: raw.items as ResponseItem[] }
}

/** The caller's retry budget, or undefined when it named none. */
function parseRetry(raw: unknown, id: RequestId): RetryParams | undefined {
  if (raw === undefined) {
    return undefined
  }
  if (!isRecord(raw) || typeof raw.maxRetries !== 'number' || !Number.isInteger(raw.maxRetries) || raw.maxRetries < 0) {
    throw new BadRequest('invalid_params', 'retry carries a non-negative integer `maxRetries`', id)
  }
  return { maxRetries: raw.maxRetries }
}

/** The per-gap deadlines, or undefined when the request named none. */
function parseTimeouts(raw: unknown, id: RequestId): TimeoutParams | undefined {
  if (raw === undefined) {
    return undefined
  }
  if (!isRecord(raw)) {
    throw new BadRequest('invalid_params', 'timeouts is an object', id)
  }
  const parsed = defined({ firstTokenMs: positive(raw.firstTokenMs), idleMs: positive(raw.idleMs) })
  for (const key of ['firstTokenMs', 'idleMs'] as const) {
    if (raw[key] !== undefined && parsed[key] === undefined) {
      throw new BadRequest('invalid_params', `timeouts.${key} is a positive number of milliseconds`, id)
    }
  }
  return Object.keys(parsed).length > 0 ? parsed : undefined
}

/** The tools the compacted turn had, in pi's own shape: the request repeats them. */
function parseTools(raw: unknown, id: RequestId): Tool[] | undefined {
  if (raw === undefined) {
    return undefined
  }
  if (!Array.isArray(raw)) {
    throw new BadRequest('invalid_params', 'tools is an array', id)
  }
  return raw.map(tool => {
    if (!isRecord(tool) || typeof tool.name !== 'string' || !isRecord(tool.parameters)) {
      throw new BadRequest('invalid_params', 'a tool carries a string name and an object `parameters`', id)
    }
    return {
      description: typeof tool.description === 'string' ? tool.description : '',
      name: tool.name,
      parameters: tool.parameters
    } as unknown as Tool
  })
}

function parseCompact(params: Record<string, unknown>, id: RequestId): CompactParams {
  if (typeof params.provider !== 'string' || typeof params.model !== 'string') {
    throw new BadRequest('invalid_params', 'compact needs string provider and model', id)
  }
  if (!isRecord(params.context) || !Array.isArray(params.context.messages)) {
    throw new BadRequest('invalid_params', 'compact needs context.messages', id)
  }
  return defined({
    context: params.context as unknown as Context,
    model: params.model,
    provider: params.provider,
    reasoning: typeof params.reasoning === 'string' ? params.reasoning : undefined,
    replay: parseReplay(params.replay, id),
    sessionId: typeof params.sessionId === 'string' ? params.sessionId : undefined,
    tools: parseTools(params.tools, id)
  })
}

function parseConfigure(params: Record<string, unknown>, id: RequestId): ConfigureParams {
  if (typeof params.credentials !== 'string' || params.credentials.length === 0) {
    throw new BadRequest('invalid_params', 'configure names the credential file in `credentials`', id)
  }
  const providers = Array.isArray(params.providers) ? params.providers : []
  const parsed = providers.map(provider => parseProvider(provider, id))
  const seen = new Set<string>()
  for (const provider of parsed) {
    if (seen.has(provider.id)) {
      throw new BadRequest('invalid_params', `provider ${provider.id} is configured twice`, id)
    }
    seen.add(provider.id)
  }
  return { apiKeys: strings(params.apiKeys), credentials: params.credentials, providers: parsed }
}

/** One inbound line as a typed request, or a `BadRequest` saying what is wrong. */
export function parseRequest(line: string): Request {
  let raw: unknown
  try {
    raw = JSON.parse(line)
  } catch (error) {
    throw new BadRequest('parse_error', `not JSON: ${(error as Error).message}`)
  }
  if (!isRecord(raw)) {
    throw new BadRequest('invalid_request', 'a request is a JSON object')
  }
  const id = isId(raw.id) ? raw.id : null
  if (id === null) {
    throw new BadRequest('invalid_request', 'a request carries a number or string id')
  }
  const params = isRecord(raw.params) ? raw.params : {}
  switch (raw.method) {
    case 'models':
      return { id, method: 'models' }
    case 'providers':
      return { id, method: 'providers' }
    case 'refresh': {
      const providers = Array.isArray(params.providers)
        ? params.providers.filter((item): item is string => typeof item === 'string' && item.length > 0)
        : undefined
      return { id, method: 'refresh', params: defined({ force: params.force === true ? true : undefined, providers }) }
    }
    case 'auth':
      return { id, method: 'auth' }
    case 'catalog': {
      if (typeof params.id !== 'string' || params.id.length === 0) {
        throw new BadRequest('invalid_params', 'catalog names the model id to look up', id)
      }
      return { id, method: 'catalog', params: { id: params.id } }
    }
    case 'configure':
      return { id, method: 'configure', params: parseConfigure(params, id) }
    case 'login': {
      if (typeof params.provider !== 'string' || params.provider.length === 0) {
        throw new BadRequest('invalid_params', 'login names a provider', id)
      }
      const mode = typeof params.mode === 'string' ? params.mode : undefined
      return { id, method: 'login', params: defined({ mode, provider: params.provider }) }
    }
    case 'login_answer': {
      if (!isId(params.loginId)) {
        throw new BadRequest('invalid_params', 'login_answer names the id of the login it answers', id)
      }
      if (typeof params.answer !== 'string') {
        throw new BadRequest('invalid_params', 'login_answer carries the answer as a string', id)
      }
      return { id, method: 'login_answer', params: { answer: params.answer, loginId: params.loginId } }
    }
    case 'logout': {
      if (typeof params.provider !== 'string' || params.provider.length === 0) {
        throw new BadRequest('invalid_params', 'logout names a provider', id)
      }
      return { id, method: 'logout', params: { provider: params.provider } }
    }
    case 'abort': {
      if (!isId(params.id)) {
        throw new BadRequest('invalid_params', 'abort names the id of the stream to abort', id)
      }
      return { id, method: 'abort', params: { id: params.id } }
    }
    case 'compact':
      return { id, method: 'compact', params: parseCompact(params, id) }
    case 'debug':
      return { id, method: 'debug' }
    case 'stream': {
      if (typeof params.provider !== 'string' || typeof params.model !== 'string') {
        throw new BadRequest('invalid_params', 'stream needs string provider and model', id)
      }
      if (!isRecord(params.context) || !Array.isArray(params.context.messages)) {
        throw new BadRequest('invalid_params', 'stream needs context.messages', id)
      }
      return {
        id,
        method: 'stream',
        params: defined({
          context: params.context as unknown as Context,
          model: params.model,
          options: isRecord(params.options) ? (params.options as StreamParams['options']) : undefined,
          provider: params.provider,
          replay: parseReplay(params.replay, id),
          retry: parseRetry(params.retry, id),
          timeouts: parseTimeouts(params.timeouts, id)
        })
      }
    }
    default:
      throw new BadRequest('method_not_found', `unknown method ${JSON.stringify(raw.method)}`, id)
  }
}

/** The event as it travels: pi-ai's event without its `partial` snapshot. */
export function stripPartial(event: AssistantMessageEvent): WireEvent {
  if ('partial' in event) {
    const { partial: _partial, ...rest } = event
    return rest
  }
  return event
}
