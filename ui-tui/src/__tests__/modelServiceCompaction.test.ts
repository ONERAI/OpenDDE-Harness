import type { Api, AssistantMessage, Context, Model, Message, Tool } from '@earendil-works/pi-ai'

import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CompactParams, ReplayHistory } from '../model-service/protocol.js'

import { fauxCompaction, fauxResponsesPayload, replayOnPayload, runCompaction } from '../model-service/compaction.js'
import { BadRequest } from '../model-service/protocol.js'

type Record_ = Record<string, unknown>

function model(overrides: Partial<Model<Api>> = {}): Model<Api> {
  return {
    api: 'openai-responses',
    baseUrl: 'https://api.openai.com/v1',
    contextWindow: 400_000,
    cost: { cacheRead: 0, cacheWrite: 0, input: 0, output: 0 },
    id: 'gpt-5.4',
    input: ['text'],
    maxTokens: 128_000,
    name: 'GPT-5.4',
    provider: 'openai',
    reasoning: true,
    ...overrides
  }
}

function assistant(text: string): AssistantMessage {
  return {
    api: 'openai-responses',
    content: [{ text, type: 'text' }],
    model: 'gpt-5.4',
    provider: 'openai',
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
    }
  }
}

const messages: Message[] = [
  { content: 'first question', role: 'user', timestamp: 0 },
  assistant('an answer'),
  { content: 'second question', role: 'user', timestamp: 0 }
]

const context: Context = {
  messages,
  systemPrompt: 'be useful',
  tools: [{ description: 'read a file', name: 'read', parameters: { type: 'object' } } as unknown as Tool]
}

const replay: ReplayHistory = {
  items: [{ encrypted_content: 'EARLIER', type: 'compaction' }]
}

function params(overrides: Partial<CompactParams> = {}): CompactParams {
  return { context, model: 'gpt-5.4', provider: 'openai', reasoning: 'high', sessionId: 'session-1', ...overrides }
}

/** The `Models` reads the compaction path makes, with nothing behind them. */
function stubModels(resolved: Model<Api> | undefined, apiKey = 'sk-test') {
  return {
    getAuth: async () => (apiKey ? { auth: { apiKey } } : undefined),
    getModel: () => resolved
  }
}

/** One Responses compaction stream, as the API sends it: the opaque item among others. */
function compactionSse(encrypted: string): string {
  return [
    `data: ${JSON.stringify({
      item: { encrypted_content: 'REASONING', summary: [], type: 'reasoning' },
      type: 'response.output_item.done'
    })}`,
    '',
    `data: ${JSON.stringify({
      item: { content: [{ text: 'compacted', type: 'output_text' }], role: 'assistant', type: 'message' },
      type: 'response.output_item.done'
    })}`,
    '',
    `data: ${JSON.stringify({
      item: { encrypted_content: encrypted, type: 'compaction' },
      type: 'response.output_item.done'
    })}`,
    '',
    `data: ${JSON.stringify({
      response: { usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 } },
      type: 'response.completed'
    })}`,
    '',
    'data: [DONE]',
    ''
  ].join('\n')
}

describe('model-service compaction', () => {
  beforeEach(() => {
    // buildCodexIdentityHeaders reads (and creates) CODEX_HOME/installation_id.
    // Never the real one.
    vi.stubEnv('CODEX_HOME', mkdtempSync(join(tmpdir(), 'codex-home-')))
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    vi.unstubAllGlobals()
  })

  it('asks the Responses API to compact and keeps the user messages ahead of the opaque item', async () => {
    const fetched: { body: Record_; headers: Record<string, string>; url: string }[] = []
    vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
      fetched.push({
        body: JSON.parse(String(init.body)),
        headers: init.headers as Record<string, string>,
        url: String(url)
      })
      return new Response(compactionSse('V2_ENCRYPTED'), { status: 200 })
    })

    const outcome = await runCompaction(stubModels(model()), params())

    expect(fetched).toHaveLength(1)
    expect(fetched[0].url).toBe('https://api.openai.com/v1/responses')
    expect(fetched[0].headers['x-codex-beta-features']).toBe('remote_compaction_v2')
    expect(fetched[0].headers.authorization).toBe('Bearer sk-test')

    const body = fetched[0].body
    expect(body.model).toBe('gpt-5.4')
    expect(body.instructions).toBe('be useful')
    expect(body.reasoning).toEqual({ effort: 'high', summary: 'auto' })
    expect(body.tools).toEqual([
      { description: 'read a file', name: 'read', parameters: { type: 'object' }, type: 'function' }
    ])
    const sent = body.input as Record_[]
    expect(sent.at(-1)).toEqual({ type: 'compaction_trigger' })
    expect(sent.map(item => item.type)).toEqual(['message', 'message', 'message', 'compaction_trigger'])

    expect(outcome.items.map(item => item.type)).toEqual(['message', 'message', 'compaction'])
    expect(outcome.items.at(-1)).toEqual({ encrypted_content: 'V2_ENCRYPTED', type: 'compaction' })
    expect(outcome.items.slice(0, 2).map(item => (item as Record_).role)).toEqual(['user', 'user'])
    expect(outcome.usage?.totalTokens).toBe(12)
  })

  it('compacts an earlier replacement history together with what followed it', async () => {
    let sent: Record_[] = []
    vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
      sent = (JSON.parse(String(init.body)) as Record_).input as Record_[]
      return new Response(compactionSse('SECOND'), { status: 200 })
    })

    const outcome = await runCompaction(stubModels(model()), params({ replay }))

    expect(sent[0]).toEqual({ encrypted_content: 'EARLIER', type: 'compaction' })
    expect(outcome.items.at(-1)).toEqual({ encrypted_content: 'SECOND', type: 'compaction' })
  })

  it('refuses a model with no server-side compaction, before asking anything', async () => {
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)

    const chat = model({ api: 'openai-completions', id: 'gpt-4.1', provider: 'openai' })
    const refusal = await runCompaction(stubModels(chat), params({ model: 'gpt-4.1' })).catch(error => error)

    expect(refusal).toBeInstanceOf(BadRequest)
    expect((refusal as BadRequest).code).toBe('unsupported_model')
    expect(fetch).not.toHaveBeenCalled()

    const missing = await runCompaction(stubModels(undefined), params()).catch(error => error)
    expect((missing as BadRequest).code).toBe('model_not_found')

    const unauthenticated = await runCompaction(stubModels(model(), ''), params()).catch(error => error)
    expect((unauthenticated as BadRequest).code).toBe('auth')
  })

  it('replaces a Responses request input with the replayed history', () => {
    const patch = replayOnPayload(model(), replay, context)
    expect(patch).toBeDefined()

    const patched = patch?.(
      {
        input: [{ content: [{ text: 'stale', type: 'input_text' }], role: 'user', type: 'message' }],
        model: 'gpt-5.4',
        previous_response_id: 'resp_earlier',
        stream: true
      },
      model()
    ) as Record_

    expect(patched).toBeDefined()
    expect(patched.previous_response_id).toBeUndefined()
    expect('previous_response_id' in patched).toBe(false)
    expect(patched.stream).toBe(true)
    const input = patched.input as Record_[]
    expect(input[0]).toEqual({ encrypted_content: 'EARLIER', type: 'compaction' })
    expect(input.map(item => item.type)).toEqual(['compaction', 'message', 'message', 'message'])
    expect(JSON.stringify(input)).not.toContain('stale')
  })

  it('leaves a payload that is not a Responses request alone', () => {
    // A chat model is not a compaction model, whatever its payload looks like.
    expect(replayOnPayload(model({ api: 'openai-completions' }), replay, context)).toBeUndefined()
    // Nor is a relay pretending to serve the Responses wire.
    expect(replayOnPayload(model({ baseUrl: 'https://relay.example.com/v1' }), replay, context)).toBeUndefined()
    // And nothing to replay changes nothing.
    expect(replayOnPayload(model(), undefined, context)).toBeUndefined()
    expect(replayOnPayload(model(), { items: [] }, context)).toBeUndefined()

    // A supported model still leaves a payload with no request body shape alone.
    const patch = replayOnPayload(model(), replay, context)
    expect(patch?.({ prompt: 'legacy completions' }, model())).toBeUndefined()
    expect(patch?.('not an object', model())).toBeUndefined()
  })

  it('scripts a compaction offline in the same shape as the real one', () => {
    expect(() => fauxCompaction(stubModels(model({ api: 'faux' })), params(), 1)).toThrow(/server-side compaction/)

    const outcome = fauxCompaction(stubModels(model()), params(), 2)

    expect(outcome.items.map(item => item.type)).toEqual(['message', 'message', 'compaction'])
    expect(outcome.items.at(-1)).toEqual({ encrypted_content: 'faux-2', type: 'compaction' })

    const payload = fauxResponsesPayload(model(), context)
    const patched = replayOnPayload(model(), { items: outcome.items }, context)?.(payload, model()) as Record_
    expect((patched.input as Record_[])[2]).toEqual({ encrypted_content: 'faux-2', type: 'compaction' })
    expect('previous_response_id' in patched).toBe(false)
  })
})
