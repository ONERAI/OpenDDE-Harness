// Turning one `configure` request into a live pi-ai `Models` collection.
//
// Two kinds of provider arrive. A built-in one (pi already knows its address,
// its wire and its models) only needs a credential, which goes into the store
// under pi's own provider id. Everything else -- a relay, a self-hosted
// server, a gateway -- is declared outright: id, address, wire, models, and
// the key to send. Neither kind is probed: the caller states the wire, the
// same rule the Python side follows.

import type { Api, Model, ProviderAuth, ProviderStreams } from '@earendil-works/pi-ai'

import { createProvider } from '@earendil-works/pi-ai'
import { anthropicMessagesApi } from '@earendil-works/pi-ai/api/anthropic-messages.lazy'
import { azureOpenAIResponsesApi } from '@earendil-works/pi-ai/api/azure-openai-responses.lazy'
import { googleGenerativeAIApi } from '@earendil-works/pi-ai/api/google-generative-ai.lazy'
import { mistralConversationsApi } from '@earendil-works/pi-ai/api/mistral-conversations.lazy'
import { openAICodexResponsesApi } from '@earendil-works/pi-ai/api/openai-codex-responses.lazy'
import { openAICompletionsApi } from '@earendil-works/pi-ai/api/openai-completions.lazy'
import { openAIResponsesApi } from '@earendil-works/pi-ai/api/openai-responses.lazy'
import { builtinModels } from '@earendil-works/pi-ai/providers/all'

import type { ConfigureModel, ConfigureParams, ConfigureProvider } from './protocol.js'

import { FileCredentialStore } from './credential-store.js'
import { discoverer, KEYLESS } from './discover.js'
import { FileModelsStore } from './models-store.js'
import { BadRequest } from './protocol.js'

/** The wires a declared provider may name. Anything else is refused by name. */
const APIS: Record<string, () => ProviderStreams> = {
  'anthropic-messages': anthropicMessagesApi,
  'azure-openai-responses': azureOpenAIResponsesApi,
  'google-generative-ai': googleGenerativeAIApi,
  'mistral-conversations': mistralConversationsApi,
  'openai-codex-responses': openAICodexResponsesApi,
  'openai-completions': openAICompletionsApi,
  'openai-responses': openAIResponsesApi
}

export const SUPPORTED_APIS = Object.keys(APIS).sort()

// What a keyless endpoint sends is `KEYLESS` (discover.ts): pi's API
// implementations require a key or an Authorization header and use that same
// word for the placeholder (`getClientApiKey` in api/openai-completions.js);
// resolving nothing instead would make a local server that needs no key look
// unconfigured, and it would never be reached at all.

/**
 * Auth for a declared provider: the stored credential first, then the key the
 * request carried, then the keyless placeholder. A declared provider is always
 * configured -- the caller named its address and its models, which is the
 * declaration -- so `auth` reports it as reachable and says which of the three
 * answered.
 */
function declaredAuth(provider: ConfigureProvider): ProviderAuth {
  return {
    apiKey: {
      name: provider.name ?? provider.id,
      resolve: async ({ credential, signal }) => {
        signal.throwIfAborted()
        if (credential?.key) {
          return { auth: { apiKey: credential.key }, source: 'stored credential' }
        }
        if (provider.apiKey) {
          return { auth: { apiKey: provider.apiKey }, source: 'configured' }
        }
        return { auth: { apiKey: KEYLESS }, source: 'keyless' }
      }
    }
  }
}

/**
 * A declared model as pi's `Model`.
 *
 * `contextWindow` and `maxTokens` stay at 0 when the caller does not know
 * them: pi reads 0 as "do not clamp", and a number invented here would be a
 * guess about somebody's own deployment. What a zero `maxTokens` must not do
 * is reach a request (pi would floor it at one token), which is why `stream`
 * in main.ts refuses such a model unless the request names its own cap.
 */
function toModel(provider: ConfigureProvider, model: ConfigureModel): Model<Api> {
  return {
    api: model.api ?? provider.api,
    baseUrl: provider.baseUrl ?? '',
    // pi keeps compatibility overrides per model (`getCompat` reads
    // `model.compat` and falls back to what it detects from the address), so a
    // row carries its own. The caller decides what a row's block holds -- for a
    // provider it declares rather than one of pi's, that includes the wire's
    // standard role, since pi's detection is a guess for a host it has no rule
    // for. Cast because pi types `compat` per wire and a row's wire is not
    // known here; pi reads only the fields its own wire declares.
    compat: model.compat as Model<Api>['compat'],
    contextWindow: model.contextWindow ?? 0,
    // Four zeroes is pi carrying no price for this model, which is what an
    // unpriced deployment is: the spend reads as unknown rather than as free.
    cost: model.cost ?? { cacheRead: 0, cacheWrite: 0, input: 0, output: 0 },
    headers: provider.headers,
    id: model.id,
    input: model.input ?? ['text'],
    maxTokens: model.maxTokens ?? 0,
    name: model.name ?? model.id,
    provider: provider.id,
    reasoning: model.reasoning ?? false
  }
}

function apiFor(api: string, providerId: string, requestId: BadRequest['id']): ProviderStreams {
  const factory = APIS[api]
  if (!factory) {
    throw new BadRequest(
      'invalid_params',
      `provider ${providerId} names api ${JSON.stringify(api)}; known apis are ${SUPPORTED_APIS.join(', ')}`,
      requestId
    )
  }
  return factory()
}

export interface Configured {
  models: ReturnType<typeof builtinModels>
  /** The last list each declared endpoint published, and when it was asked. */
  modelsStore: FileModelsStore
  store: FileCredentialStore
}

/** Where the last list each declared provider published is kept: beside the credential store. */
export function modelsStorePath(credentials: string): string {
  return credentials.replace(/[^/\\]*$/, 'pi-models-store.json')
}

/**
 * Build the collection this request describes. Called again on every
 * `configure`, so the previous call's declared providers are gone rather than
 * merged: the request states the whole set.
 */
export async function configure(params: ConfigureParams, requestId: BadRequest['id']): Promise<Configured> {
  const store = new FileCredentialStore(params.credentials)
  const modelsStore = new FileModelsStore(modelsStorePath(params.credentials))
  const models = builtinModels({ credentials: store, modelsStore })

  for (const provider of params.providers) {
    // A mixed-wire provider dispatches per model; one wire is one implementation.
    const wires = new Set(provider.models.map(model => model.api ?? provider.api))
    const api =
      wires.size > 1
        ? Object.fromEntries([...wires].map(wire => [wire, apiFor(wire, provider.id, requestId)]))
        : apiFor(provider.api, provider.id, requestId)
    models.setProvider(
      createProvider({
        api,
        auth: declaredAuth(provider),
        baseUrl: provider.baseUrl,
        // The endpoint's own list, the way pi's dynamic providers get theirs:
        // fetched by `Models.refresh()`, restored from the models store first,
        // overlaid on the rows the operator declared.
        fetchModels: discoverer(provider),
        headers: provider.headers,
        id: provider.id,
        models: provider.models.map(model => toModel(provider, model)),
        name: provider.name ?? provider.id
      })
    )
  }

  // The keys go into pi's own store, through pi's own serialised writer: this
  // is the one process writing that file, so a login or a refresh landing at
  // the same moment is queued behind these rather than overwritten by a
  // snapshot taken before it. A stored key whose provider no longer supplies
  // one is dropped for the same reason a key is written -- pi resolves the
  // store first, and a key removed from the config would otherwise go on being
  // sent. Sign-ins are pi's and are never touched here.
  for (const [providerId, key] of Object.entries(params.apiKeys)) {
    await store.modify(providerId, async () => ({ key, type: 'api_key' }))
  }
  for (const stored of await store.list()) {
    if (stored.type === 'api_key' && !(stored.providerId in params.apiKeys)) {
      await store.delete(stored.providerId)
    }
  }

  return { models, modelsStore, store }
}
