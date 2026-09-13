import { describe, expect, it } from 'vitest'

import { BadRequest, parseRequest, stripPartial } from '../model-service/protocol.js'

describe('model-service protocol', () => {
  it('parses the three methods', () => {
    expect(parseRequest('{"id":1,"method":"models"}')).toEqual({ id: 1, method: 'models' })
    expect(parseRequest('{"id":"a","method":"abort","params":{"id":3}}')).toEqual({
      id: 'a',
      method: 'abort',
      params: { id: 3 }
    })
    const stream = parseRequest(
      '{"id":2,"method":"stream","params":{"provider":"faux","model":"echo","context":{"messages":[]},"options":{"reasoning":"low"}}}'
    )
    expect(stream).toEqual({
      id: 2,
      method: 'stream',
      params: { context: { messages: [] }, model: 'echo', options: { reasoning: 'low' }, provider: 'faux' }
    })
  })

  it('parses a catalogue lookup and refuses one that names no model', () => {
    expect(parseRequest('{"id":4,"method":"catalog","params":{"id":"deepseek-v4-flash"}}')).toEqual({
      id: 4,
      method: 'catalog',
      params: { id: 'deepseek-v4-flash' }
    })
    expect(() => parseRequest('{"id":4,"method":"catalog","params":{}}')).toThrow(/model id/)
    expect(() => parseRequest('{"id":4,"method":"catalog","params":{"id":""}}')).toThrow(/model id/)
  })

  it('refuses what it cannot answer, keeping the id when there is one', () => {
    expect(() => parseRequest('nope')).toThrow(BadRequest)
    expect(() => parseRequest('[1]')).toThrow(/JSON object/)
    expect(() => parseRequest('{"method":"models"}')).toThrow(/id/)
    const noModel = (() => {
      try {
        parseRequest('{"id":9,"method":"stream","params":{"provider":"x"}}')
      } catch (error) {
        return error as BadRequest
      }
    })()
    expect(noModel?.code).toBe('invalid_params')
    expect(noModel?.id).toBe(9)
    expect(() => parseRequest('{"id":1,"method":"fly"}')).toThrow(/unknown method/)
  })

  it('parses a configure request, filling in what a model leaves out', () => {
    const request = parseRequest(
      JSON.stringify({
        id: 4,
        method: 'configure',
        params: {
          apiKeys: { anthropic: 'sk-synthetic', bogus: 7 },
          credentials: '/tmp/auth.json',
          providers: [
            {
              api: 'openai-completions',
              baseUrl: 'http://127.0.0.1:8000/v1',
              headers: { 'X-Tenant': 'lab' },
              id: 'lab',
              models: [
                { contextWindow: 8192, id: 'a', maxTokens: 1024 },
                { api: 'openai-responses', id: 'b' }
              ]
            }
          ]
        }
      })
    )
    expect(request).toEqual({
      id: 4,
      method: 'configure',
      params: {
        // A non-string key is dropped rather than sent on as one.
        apiKeys: { anthropic: 'sk-synthetic' },
        credentials: '/tmp/auth.json',
        providers: [
          {
            api: 'openai-completions',
            baseUrl: 'http://127.0.0.1:8000/v1',
            headers: { 'X-Tenant': 'lab' },
            id: 'lab',
            // Nothing a request left out comes back as a key that was set.
            models: [
              { contextWindow: 8192, id: 'a', maxTokens: 1024 },
              { api: 'openai-responses', id: 'b' }
            ]
          }
        ]
      }
    })
  })

  it('refuses a configure request that cannot describe a provider', () => {
    const refused = (params: unknown) => {
      try {
        parseRequest(JSON.stringify({ id: 5, method: 'configure', params }))
      } catch (error) {
        return error as BadRequest
      }
      return undefined
    }
    expect(refused({ providers: [] })?.message).toMatch(/credentials/)
    expect(refused({ credentials: '/tmp/a', providers: [{ id: 'x', models: [] }] })?.message).toMatch(/no api/)
    expect(refused({ credentials: '/tmp/a', providers: [{ api: 'openai-completions', id: 'x' }] })?.message).toMatch(
      /no models array/
    )
    expect(
      refused({
        credentials: '/tmp/a',
        providers: [{ api: 'openai-completions', id: 'x', models: [{ name: 'nameless' }] }]
      })?.message
    ).toMatch(/without an id/)
    const twice = refused({
      credentials: '/tmp/a',
      providers: [
        { api: 'openai-completions', id: 'x', models: [] },
        { api: 'openai-completions', id: 'x', models: [] }
      ]
    })
    expect(twice?.message).toMatch(/configured twice/)
    expect(twice?.id).toBe(5)
  })

  it('parses login and auth', () => {
    expect(parseRequest('{"id":6,"method":"login","params":{"provider":"openai-codex","mode":"device_code"}}')).toEqual(
      { id: 6, method: 'login', params: { mode: 'device_code', provider: 'openai-codex' } }
    )
    expect(parseRequest('{"id":7,"method":"auth"}')).toEqual({ id: 7, method: 'auth' })
    expect(() => parseRequest('{"id":9,"method":"login","params":{}}')).toThrow(/names a provider/)
    // A login with no mode is the one whose menu travels, so the field is absent
    // rather than defaulted here.
    expect(parseRequest('{"id":8,"method":"login","params":{"provider":"anthropic"}}')).toEqual({
      id: 8,
      method: 'login',
      params: { provider: 'anthropic' }
    })
  })

  it('parses the request for pi own auth descriptors', () => {
    expect(parseRequest('{"id":5,"method":"providers"}')).toEqual({ id: 5, method: 'providers' })
  })

  it('parses an answer for a waiting login, and refuses one that names no login', () => {
    expect(parseRequest('{"id":3,"method":"login_answer","params":{"loginId":6,"answer":"browser"}}')).toEqual({
      id: 3,
      method: 'login_answer',
      params: { answer: 'browser', loginId: 6 }
    })
    // An empty answer is a real answer: a field submitted blank.
    expect(parseRequest('{"id":3,"method":"login_answer","params":{"loginId":"a","answer":""}}')).toEqual({
      id: 3,
      method: 'login_answer',
      params: { answer: '', loginId: 'a' }
    })
    expect(() => parseRequest('{"id":3,"method":"login_answer","params":{"answer":"x"}}')).toThrow(/id of the login/)
    expect(() => parseRequest('{"id":3,"method":"login_answer","params":{"loginId":6}}')).toThrow(/as a string/)
  })

  it('drops the partial snapshot and nothing else', () => {
    const partial = { content: [], role: 'assistant' } as never
    expect(stripPartial({ contentIndex: 0, delta: 'x', partial, type: 'text_delta' })).toEqual({
      contentIndex: 0,
      delta: 'x',
      type: 'text_delta'
    })
  })
})
