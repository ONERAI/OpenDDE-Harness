// What a declared endpoint serves, read from the endpoint itself.
//
// A relay, a gateway, a self-hosted server: every OpenAI-compatible endpoint
// publishes its models at `GET <baseUrl>/models`, and OpenRouter -- the relay
// pi ships a catalogue for -- publishes the richest shape of that list. A
// provider this config declares is treated the way pi treats OpenRouter: its
// address and key are the only things anyone types, and the models come from
// the list it publishes. This is pi-ai's own mechanism for a dynamic provider
// (`createProvider({ fetchModels })`): the fetched rows overlay the declared
// ones, `Models.refresh()` runs the fetch, and the models store keeps the last
// list across restarts.
//
// Two shapes are read. The minimal OpenAI one is `{data: [{id}]}`, and a row
// then knows nothing but its id. OpenRouter's shape adds `name`,
// `context_length`, `pricing` (per token, which pi keeps per million),
// `architecture.input_modalities`, `supported_parameters` and
// `top_provider.max_completion_tokens`; vLLM publishes `max_model_len`. Each
// is read when present. What a row leaves out is borrowed from pi's own
// catalogue for the same model id, the way a declared row borrows it: a relay
// fronting a vendor's model is sized by the vendor's row. What nobody knows
// stays zero -- pi reads a zero window as "do not trim" -- and nothing is
// invented here.

import type { Api, Model, ModelsStore, RefreshModelsContext } from '@earendil-works/pi-ai'

import { builtinModels } from '@earendil-works/pi-ai/providers/all'

import type { ConfigureProvider } from './protocol.js'

/** What `declaredAuth` resolves for an endpoint reached without a key. */
export const KEYLESS = 'unused'

/**
 * How long a declared endpoint's list is trusted before it is fetched again.
 * pi's remote catalogs skip the network for four hours after a check; an
 * endpoint's own list changes rarer than that, and the picker paints the
 * stored list either way -- this only decides whether the background refresh
 * asks at all.
 */
export const CATALOG_MAX_AGE_MS = 24 * 60 * 60 * 1000

/** The providers among `candidates` whose stored list is older than the window, or absent. */
export async function staleCatalogs(
  store: ModelsStore | undefined,
  candidates: readonly string[],
  now: number = Date.now()
): Promise<string[]> {
  const out: string[] = []
  for (const id of candidates) {
    const entry = await store?.read(id)
    if (entry?.checkedAt === undefined || now - entry.checkedAt >= CATALOG_MAX_AGE_MS) {
      out.push(id)
    }
  }
  return out
}

const DISCOVERY_TIMEOUT_MS = 10_000

type FetchLike = (url: string, init: RequestInit) => Promise<Response>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function number(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
    return value
  }
  if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value)) && Number(value) >= 0) {
    return Number(value)
  }
  return undefined
}

/** pi keeps rates per million tokens; OpenRouter publishes them per token. */
function perMillion(value: unknown): number {
  const rate = number(value)
  return rate === undefined ? 0 : rate * 1_000_000
}

let catalogue: ReturnType<typeof builtinModels> | undefined

/** What pi's own rows know about one model id: the same merge `pi_auth.merged_catalog_row` makes. */
interface CatalogFacts {
  contextWindow?: number
  input?: ('image' | 'text')[]
  maxTokens?: number
  reasoning?: boolean
}

/**
 * pi's catalogue rows for a published id. A leading *pi* provider id is
 * dropped first ("openai/gpt-5" on a relay asks about gpt-5), but not any
 * other prefix: a relay fronting OpenRouter names its models
 * "anthropic/claude-x", and that whole string is the id pi's OpenRouter rows
 * carry. Where several rows disagree about a limit the smallest positive one
 * wins, because it is the only one true of every row; reasoning is true if any
 * row says so and the modalities are unioned.
 */
export function catalogFacts(id: string): CatalogFacts {
  catalogue ??= builtinModels()
  const providers = new Set(catalogue.getProviders().map(provider => provider.id))
  const slash = id.indexOf('/')
  const key = slash > 0 && providers.has(id.slice(0, slash)) ? id.slice(slash + 1) : id
  const rows = catalogue.getModels().filter(model => model.id === key)
  const facts: CatalogFacts = {}
  for (const row of rows) {
    if (row.contextWindow > 0) {
      facts.contextWindow = Math.min(facts.contextWindow ?? row.contextWindow, row.contextWindow)
    }
    if (row.maxTokens > 0) {
      facts.maxTokens = Math.min(facts.maxTokens ?? row.maxTokens, row.maxTokens)
    }
    if (row.reasoning) {
      facts.reasoning = true
    }
    for (const kind of row.input) {
      if ((kind === 'text' || kind === 'image') && !(facts.input ??= []).includes(kind)) {
        facts.input.push(kind)
      }
    }
  }
  return facts
}

/** The rows of a `/models` reply, whichever of the two shapes it uses. */
export function modelRows(payload: unknown): Record<string, unknown>[] {
  const entries = Array.isArray(payload)
    ? payload
    : isRecord(payload) && Array.isArray(payload.data)
      ? payload.data
      : isRecord(payload) && Array.isArray(payload.models)
        ? payload.models
        : undefined
  if (!entries) {
    throw new Error('the reply is not a model list: expected {data: [...]}')
  }
  return entries.filter((entry): entry is Record<string, unknown> => isRecord(entry) && typeof entry.id === 'string')
}

/** One published row as pi's `Model`, under this provider. */
export function toDiscoveredModel(provider: ConfigureProvider, row: Record<string, unknown>): Model<Api> {
  const id = String(row.id)
  const architecture = isRecord(row.architecture) ? row.architecture : {}
  const modalities = Array.isArray(architecture.input_modalities) ? architecture.input_modalities : []
  const input = modalities.filter((mode): mode is 'image' | 'text' => mode === 'text' || mode === 'image')
  const pricing = isRecord(row.pricing) ? row.pricing : {}
  const parameters = Array.isArray(row.supported_parameters) ? row.supported_parameters : []
  const topProvider = isRecord(row.top_provider) ? row.top_provider : {}
  const known = catalogFacts(id)
  const published = {
    contextWindow: number(row.context_length) ?? number(row.context_window) ?? number(row.max_model_len),
    maxTokens:
      number(topProvider.max_completion_tokens) ??
      number(row.max_completion_tokens) ??
      number(row.max_output_tokens) ??
      number(row.max_tokens),
    reasoning: parameters.includes('reasoning') || parameters.includes('include_reasoning')
  }
  return {
    api: provider.api as Api,
    baseUrl: provider.baseUrl ?? '',
    // The wire's standard role and whatever the operator declared for the
    // whole endpoint, the same block every declared row carries.
    compat: provider.compat as Model<Api>['compat'],
    contextWindow: published.contextWindow ?? known.contextWindow ?? 0,
    cost: {
      cacheRead: perMillion(pricing.input_cache_read),
      cacheWrite: perMillion(pricing.input_cache_write),
      input: perMillion(pricing.prompt),
      output: perMillion(pricing.completion)
    },
    headers: provider.headers,
    id,
    input: input.length > 0 ? input : (known.input ?? ['text']),
    maxTokens: published.maxTokens ?? known.maxTokens ?? 0,
    name: typeof row.name === 'string' && row.name.trim() ? row.name : id,
    provider: provider.id,
    reasoning: published.reasoning || known.reasoning === true
  }
}

/**
 * The endpoint's list, minus the ids the operator declared by hand: a declared
 * row is the operator describing their own deployment (a window, a ceiling, a
 * name), and pi's overlay would replace it with the endpoint's row otherwise.
 */
export function discoveredModels(
  provider: ConfigureProvider,
  payload: unknown,
  declared: ReadonlySet<string>
): Model<Api>[] {
  const seen = new Set<string>()
  const out: Model<Api>[] = []
  for (const row of modelRows(payload)) {
    const id = String(row.id)
    if (declared.has(id) || seen.has(id)) {
      continue
    }
    seen.add(id)
    out.push(toDiscoveredModel(provider, row))
  }
  return out
}

/** pi's `fetchModels` for a declared provider. */
export function discoverer(
  provider: ConfigureProvider,
  fetchImpl: FetchLike = (url, init) => fetch(url, init)
): (context: RefreshModelsContext) => Promise<readonly Model<Api>[]> {
  const declared = new Set(provider.models.map(model => model.id))
  return async context => {
    if (!provider.baseUrl) {
      throw new Error(`provider ${provider.id} has no address to list models from`)
    }
    const key =
      context.credential?.type === 'api_key' && context.credential.key !== KEYLESS
        ? context.credential.key
        : (provider.apiKey ?? '')
    const headers: Record<string, string> = { accept: 'application/json', ...(provider.headers ?? {}) }
    if (key) {
      headers.authorization = `Bearer ${key}`
    }
    const signal = AbortSignal.any([context.signal, AbortSignal.timeout(DISCOVERY_TIMEOUT_MS)])
    const response = await fetchImpl(`${provider.baseUrl.replace(/\/+$/, '')}/models`, { headers, signal })
    if (!response.ok) {
      throw new Error(`${provider.id} answered ${response.status} to GET /models`)
    }
    return discoveredModels(provider, await response.json(), declared)
  }
}
