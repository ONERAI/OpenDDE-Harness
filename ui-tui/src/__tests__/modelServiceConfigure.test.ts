// What one `configure` request builds: the rows a declared provider serves, as
// pi will read them. The compatibility block is the interesting part -- pi
// attaches it per model and otherwise guesses from the address.

import { mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

import { configure } from '../model-service/configure.js'
import { parseRequest } from '../model-service/protocol.js'

async function credentials(): Promise<string> {
  // Named but never written: nothing here stores a credential.
  return join(await mkdtemp(join(tmpdir(), 'model-service-configure-')), 'auth.json')
}

describe('configure', () => {
  it("carries a declared model's compat into the provider pi is given", async () => {
    const { models } = await configure(
      {
        apiKeys: {},
        credentials: await credentials(),
        providers: [
          {
            api: 'openai-completions',
            baseUrl: 'http://127.0.0.1:8000/v1',
            id: 'lab',
            models: [
              { compat: { supportsDeveloperRole: false }, contextWindow: 8192, id: 'a', maxTokens: 1024 },
              { compat: { supportsDeveloperRole: true, thinkingFormat: 'zai' }, id: 'b' },
              { id: 'c' }
            ]
          }
        ]
      },
      1
    )

    expect(models.getModel('lab', 'a')?.compat).toEqual({ supportsDeveloperRole: false })
    expect(models.getModel('lab', 'b')?.compat).toEqual({ supportsDeveloperRole: true, thinkingFormat: 'zai' })
    // A row that carries none keeps pi's own detection, which is what a row
    // saying nothing means.
    expect(models.getModel('lab', 'c')?.compat).toBeUndefined()
  })

  it("writes the keys into pi's store and drops a stored key the config no longer supplies", async () => {
    // pi's store is the one writer of that file: a key removed from the config
    // would otherwise go on being resolved from the store, and a sign-in pi
    // wrote there is not this request's to touch.
    const path = await credentials()
    await writeFile(
      path,
      JSON.stringify({
        anthropic: { key: 'the-old-one', type: 'api_key' },
        'openai-codex': { access: 'a', expires: 1, refresh: 'r', type: 'oauth' }
      })
    )

    await configure({ apiKeys: { deepseek: 'sk-deepseek-synthetic' }, credentials: path, providers: [] }, 1)

    expect(JSON.parse(await readFile(path, 'utf8'))).toEqual({
      deepseek: { key: 'sk-deepseek-synthetic', type: 'api_key' },
      'openai-codex': { access: 'a', expires: 1, refresh: 'r', type: 'oauth' }
    })
  })

  it('parses a compat block through unchanged and drops one that is not an object', () => {
    const parsed = parseRequest(
      JSON.stringify({
        id: 7,
        method: 'configure',
        params: {
          credentials: '/tmp/auth.json',
          providers: [
            {
              api: 'openai-completions',
              baseUrl: 'http://127.0.0.1:8000/v1',
              id: 'lab',
              models: [
                { compat: { maxTokensField: 'max_tokens', supportsDeveloperRole: false }, id: 'a' },
                { compat: 'yes', id: 'b' }
              ]
            }
          ]
        }
      })
    )

    const models = parsed.method === 'configure' ? parsed.params.providers[0].models : []
    expect(models[0].compat).toEqual({ maxTokensField: 'max_tokens', supportsDeveloperRole: false })
    expect('compat' in models[1]).toBe(false)
  })
})
