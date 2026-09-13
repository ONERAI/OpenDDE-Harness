// A declared endpoint's models come from the endpoint, the way pi treats
// OpenRouter: the address and key are typed, the list is published. The
// endpoint is faked with a stubbed fetch; what is under test is the reading of
// its two list shapes, the overlay pi builds from the fetch, the models store
// that keeps it, and the `refresh` request that runs it.

import type { ModelsStoreEntry } from '@earendil-works/pi-ai'

import { mkdtemp, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ConfigureProvider } from '../model-service/protocol.js'

import { configure, modelsStorePath } from '../model-service/configure.js'
import {
  CATALOG_MAX_AGE_MS,
  catalogFacts,
  discoveredModels,
  discoverer,
  KEYLESS,
  staleCatalogs,
  toDiscoveredModel
} from '../model-service/discover.js'
import { FileModelsStore } from '../model-service/models-store.js'
import { parseRequest } from '../model-service/protocol.js'

const RELAY: ConfigureProvider = {
  api: 'openai-completions',
  apiKey: 'sk-relay',
  baseUrl: 'https://relay.example/v1/',
  compat: { supportsDeveloperRole: false },
  id: 'yunjintao',
  models: [{ contextWindow: 200000, id: 'declared-by-hand', maxTokens: 8192 }]
}

/** OpenRouter's shape for one row, the richest a relay publishes. */
const OPENROUTER_ROW = {
  architecture: { input_modalities: ['text', 'image', 'file'], output_modalities: ['text'] },
  context_length: 400000,
  id: 'anthropic/claude-opus-5',
  name: 'Anthropic: Claude Opus 5',
  pricing: { completion: '0.000075', input_cache_read: '0.0000015', prompt: '0.000015' },
  supported_parameters: ['temperature', 'reasoning', 'tools'],
  top_provider: { max_completion_tokens: 64000 }
}

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { headers: { 'content-type': 'application/json' }, status })
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe("reading an endpoint's model list", () => {
  it("reads OpenRouter's shape: window, ceiling, modalities, reasoning and per-million prices", () => {
    const model = toDiscoveredModel(RELAY, OPENROUTER_ROW)

    expect(model).toEqual({
      api: 'openai-completions',
      baseUrl: 'https://relay.example/v1/',
      compat: { supportsDeveloperRole: false },
      contextWindow: 400000,
      cost: { cacheRead: 1.5, cacheWrite: 0, input: 15, output: 75 },
      headers: undefined,
      id: 'anthropic/claude-opus-5',
      input: ['text', 'image'],
      maxTokens: 64000,
      name: 'Anthropic: Claude Opus 5',
      provider: 'yunjintao',
      reasoning: true
    })
  })

  it('reads the minimal OpenAI shape as an id that knows nothing else', () => {
    const [model] = discoveredModels(RELAY, { data: [{ id: 'qwen3-32b', object: 'model' }] }, new Set())

    expect(model).toMatchObject({
      contextWindow: 0,
      cost: { cacheRead: 0, cacheWrite: 0, input: 0, output: 0 },
      id: 'qwen3-32b',
      input: ['text'],
      maxTokens: 0,
      name: 'qwen3-32b',
      reasoning: false
    })
  })

  it("reads vLLM's max_model_len as the window", () => {
    const [model] = discoveredModels(RELAY, { data: [{ id: 'qwen3-32b', max_model_len: 40960 }] }, new Set())

    expect(model?.contextWindow).toBe(40960)
    expect(model?.maxTokens).toBe(0)
  })

  it("borrows what pi's own catalogue knows about the same model id, the way a declared row does", () => {
    // A relay fronting OpenRouter names its models the way OpenRouter does, and
    // that whole string is the id pi's OpenRouter rows carry; a relay fronting
    // OpenAI names them "openai/gpt-5", and the pi provider prefix is dropped.
    expect(catalogFacts('anthropic/claude-opus-5')).toEqual({
      contextWindow: 1000000,
      input: ['text', 'image'],
      maxTokens: 64000,
      reasoning: true
    })
    // The smallest positive limit across pi's rows for the id: cloudflare's
    // 128k window for gpt-5, not OpenAI's 400k.
    expect(catalogFacts('openai/gpt-5')).toMatchObject({ contextWindow: 128000, maxTokens: 128000 })
    expect(catalogFacts('glm-4.6')).toEqual({})

    const [fronted, bare] = discoveredModels(
      RELAY,
      { data: [{ id: 'anthropic/claude-opus-5' }, { id: 'glm-4.6' }] },
      new Set()
    )
    expect(fronted).toMatchObject({
      contextWindow: 1000000,
      input: ['text', 'image'],
      maxTokens: 64000,
      reasoning: true
    })
    expect(bare).toMatchObject({ contextWindow: 0, maxTokens: 0 })
    // What the endpoint publishes wins over the catalogue.
    const [published] = discoveredModels(
      RELAY,
      { data: [{ id: 'anthropic/claude-opus-5', context_length: 200000 }] },
      new Set()
    )
    expect(published?.contextWindow).toBe(200000)
  })

  it('leaves a row the operator declared by hand to the declaration, and drops duplicates and junk', () => {
    const models = discoveredModels(
      RELAY,
      [{ id: 'declared-by-hand', context_length: 1 }, { id: 'a' }, { id: 'a' }, 'not a row', { name: 'no id' }],
      new Set(['declared-by-hand'])
    )

    expect(models.map(model => model.id)).toEqual(['a'])
  })

  it('refuses a reply that is not a list', () => {
    expect(() => discoveredModels(RELAY, { error: 'nope' }, new Set())).toThrow(/not a model list/)
  })
})

describe('fetching the list', () => {
  it('asks GET <baseUrl>/models with the resolved key, and no key for a keyless endpoint', async () => {
    const calls: [string, RequestInit][] = []
    const fetchImpl = async (url: string, init: RequestInit) => {
      calls.push([url, init])
      return response({ data: [{ id: 'a' }] })
    }
    const context = { allowNetwork: true, publish: async () => true, signal: new AbortController().signal }

    await discoverer(RELAY, fetchImpl)({ ...context, credential: { key: 'sk-stored', type: 'api_key' } })
    await discoverer(
      { ...RELAY, apiKey: undefined },
      fetchImpl
    )({
      ...context,
      credential: { key: KEYLESS, type: 'api_key' }
    })

    expect(calls[0]?.[0]).toBe('https://relay.example/v1/models')
    expect((calls[0]?.[1].headers as Record<string, string>).authorization).toBe('Bearer sk-stored')
    expect((calls[1]?.[1].headers as Record<string, string>).authorization).toBeUndefined()
  })

  it('reports an endpoint that refuses, with its status', async () => {
    const fetchImpl = async () => response({ detail: 'no' }, 404)
    const context = { allowNetwork: true, publish: async () => true, signal: new AbortController().signal }

    await expect(discoverer(RELAY, fetchImpl)(context)).rejects.toThrow('yunjintao answered 404 to GET /models')
  })
})

describe('a declared provider through configure and refresh', () => {
  it('serves its declared rows at once and the published ones after a refresh, kept in the models store', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => response({ data: [OPENROUTER_ROW, { id: 'declared-by-hand' }, { id: 'qwen3-32b' }] }))
    )
    const credentials = join(await mkdtemp(join(tmpdir(), 'model-service-discover-')), 'auth.json')

    const { models } = await configure({ apiKeys: {}, credentials, providers: [RELAY] }, 1)

    expect(models.getModels('yunjintao').map(model => model.id)).toEqual(['declared-by-hand'])

    const result = await models.refresh({ allowNetwork: true, force: true, providers: ['yunjintao'] })

    expect([...result.errors]).toEqual([])
    expect(models.getModels('yunjintao').map(model => model.id)).toEqual([
      'declared-by-hand',
      'anthropic/claude-opus-5',
      'qwen3-32b'
    ])
    // The hand-declared row keeps the operator's own limits.
    expect(models.getModel('yunjintao', 'declared-by-hand')?.contextWindow).toBe(200000)
    expect(models.getModel('yunjintao', 'anthropic/claude-opus-5')?.contextWindow).toBe(400000)
    const stored = JSON.parse(await readFile(modelsStorePath(credentials), 'utf8')) as Record<string, ModelsStoreEntry>
    expect(stored.yunjintao?.models.map(model => model.id)).toEqual(['anthropic/claude-opus-5', 'qwen3-32b'])

    // A new service over the same store shows the last list before any fetch.
    const again = await configure({ apiKeys: {}, credentials, providers: [RELAY] }, 2)
    await again.models.refresh({ allowNetwork: false })

    expect(again.models.getModels('yunjintao').map(model => model.id)).toEqual([
      'declared-by-hand',
      'anthropic/claude-opus-5',
      'qwen3-32b'
    ])
  })

  it('keeps the last list and reports the provider when the endpoint fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => response({ detail: 'down' }, 503))
    )
    const credentials = join(await mkdtemp(join(tmpdir(), 'model-service-discover-')), 'auth.json')
    const { models } = await configure({ apiKeys: {}, credentials, providers: [RELAY] }, 1)

    const result = await models.refresh({ allowNetwork: true, force: true })

    expect(result.errors.get('yunjintao')?.message).toContain('answered 503')
    expect(models.getModels('yunjintao').map(model => model.id)).toEqual(['declared-by-hand'])
  })

  it('parses the refresh request, with or without a provider list, and its force flag', () => {
    expect(
      parseRequest(JSON.stringify({ id: 1, method: 'refresh', params: { force: true, providers: ['a', 3, ''] } }))
    ).toEqual({
      id: 1,
      method: 'refresh',
      params: { force: true, providers: ['a'] }
    })
    expect(parseRequest(JSON.stringify({ id: 2, method: 'refresh', params: { force: 'yes' } }))).toEqual({
      id: 2,
      method: 'refresh',
      params: {}
    })
  })

  it('carries a provider-wide compat block through configure', () => {
    const request = parseRequest(
      JSON.stringify({
        id: 3,
        method: 'configure',
        params: {
          credentials: '/tmp/auth.json',
          providers: [
            { api: 'openai-completions', baseUrl: 'http://x', compat: { thinkingFormat: 'zai' }, id: 'r', models: [] }
          ]
        }
      })
    )

    expect(request.method === 'configure' && request.params.providers[0]?.compat).toEqual({ thinkingFormat: 'zai' })
  })
})

describe('pacing the refresh', () => {
  it('asks again only for a list older than the window, or never fetched', async () => {
    const store = new FileModelsStore(join(await mkdtemp(join(tmpdir(), 'models-store-')), 'pi-models-store.json'))
    const now = 1_700_000_000_000
    await store.write('fresh', { checkedAt: now - CATALOG_MAX_AGE_MS + 1, models: [] })
    await store.write('old', { checkedAt: now - CATALOG_MAX_AGE_MS, models: [] })
    await store.write('unchecked', { models: [] })

    expect(await staleCatalogs(store, ['fresh', 'old', 'unchecked', 'never'], now)).toEqual([
      'old',
      'unchecked',
      'never'
    ])
    // No store at all: every endpoint is asked.
    expect(await staleCatalogs(undefined, ['fresh'], now)).toEqual(['fresh'])
  })
})

describe('the models store', () => {
  it('round-trips an entry and deletes it, leaving no temp file', async () => {
    const path = join(await mkdtemp(join(tmpdir(), 'models-store-')), 'pi-models-store.json')
    const store = new FileModelsStore(path)
    const entry: ModelsStoreEntry = { checkedAt: 5, models: [] }

    expect(await store.read('r')).toBeUndefined()
    await Promise.all([store.write('r', entry), store.write('s', { models: [] })])
    expect(await store.read('r')).toEqual(entry)
    expect(Object.keys(JSON.parse(await readFile(path, 'utf8')) as object).sort()).toEqual(['r', 's'])

    await store.delete('r')
    expect(await store.read('r')).toBeUndefined()
    expect(await store.read('s')).toEqual({ models: [] })
  })
})
