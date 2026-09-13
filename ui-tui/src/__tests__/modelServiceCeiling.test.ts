// What a request carries as its ceiling when nobody sized the model.
import type { Api, Model } from '@earendil-works/pi-ai'

import { describe, expect, it } from 'vitest'

import { ceilingFor, UNBOUNDED_WINDOW } from '../model-service/ceiling.js'

function model(overrides: Partial<Model<Api>> = {}): Model<Api> {
  return {
    api: 'openai-completions',
    baseUrl: 'https://relay.example/v1',
    contextWindow: 0,
    cost: { cacheRead: 0, cacheWrite: 0, input: 0, output: 0 },
    id: 'glm-4.6',
    input: ['text'],
    maxTokens: 0,
    name: 'glm-4.6',
    provider: 'custom',
    reasoning: false,
    ...overrides
  }
}

describe('the ceiling of an unsized model', () => {
  it('leaves a model alone once anything is known: a named ceiling, its own, or its window', () => {
    const unsized = model()

    expect(ceilingFor(unsized, 256)).toEqual({ model: unsized })
    expect(ceilingFor(model({ maxTokens: 8192 }), undefined).model.contextWindow).toBe(0)
    // A window alone is enough: pi's clamp then leaves the zero ceiling at
    // zero and the OpenAI wires send none.
    expect(ceilingFor(model({ contextWindow: 128000 }), undefined).model).toEqual(model({ contextWindow: 128000 }))
  })

  it('streams an OpenAI-shaped request under an unbounded window, so pi sends no ceiling', () => {
    // pi floors the ceiling at one token only when the window is unknown; an
    // unbounded window for the clamp's sake makes the zero drop out of the
    // request, and the server's own default applies -- the request every
    // OpenAI-compatible client sends for a model it has not sized.
    for (const api of ['openai-completions', 'openai-responses', 'openai-codex-responses', 'azure-openai-responses']) {
      const decision = ceilingFor(model({ api: api as Api }), undefined)

      expect(decision.refusal).toBeUndefined()
      expect(decision.model.contextWindow).toBe(UNBOUNDED_WINDOW)
      expect(decision.model.maxTokens).toBe(0)
    }
  })

  it('refuses on a wire that would send the zero as written, naming the fix', () => {
    const decision = ceilingFor(model({ api: 'anthropic-messages' }), undefined)

    expect(decision.refusal).toContain('custom/glm-4.6 declares no maxTokens and no contextWindow')
    expect(decision.refusal).toContain('ddeharness provider set')
  })
})
