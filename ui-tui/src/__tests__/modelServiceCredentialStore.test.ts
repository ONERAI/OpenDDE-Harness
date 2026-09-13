import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { FileCredentialStore } from '../model-service/credential-store.js'

describe('the model service credential store', () => {
  let directory: string
  let path: string

  beforeEach(async () => {
    directory = await mkdtemp(join(tmpdir(), 'pi-credentials-'))
    path = join(directory, 'nested', 'auth.json')
  })

  afterEach(async () => {
    await rm(directory, { force: true, recursive: true })
  })

  it('round-trips a credential through read, modify, list and delete', async () => {
    const store = new FileCredentialStore(path)
    expect(await store.read('anthropic')).toBeUndefined()
    expect(await store.list()).toEqual([])

    const written = await store.modify('anthropic', async () => ({ key: 'sk-synthetic', type: 'api_key' }))
    expect(written).toEqual({ key: 'sk-synthetic', type: 'api_key' })
    expect(await store.read('anthropic')).toEqual({ key: 'sk-synthetic', type: 'api_key' })

    await store.modify('openai-codex', async () => ({
      access: 'a',
      expires: 7,
      refresh: 'r',
      type: 'oauth'
    }))
    expect(await store.list()).toEqual([
      { providerId: 'anthropic', type: 'api_key' },
      { providerId: 'openai-codex', type: 'oauth' }
    ])

    await store.delete('anthropic')
    expect(await store.read('anthropic')).toBeUndefined()
    expect(await store.list()).toEqual([{ providerId: 'openai-codex', type: 'oauth' }])
  })

  it('shows the current credential to the modifier and leaves it alone on undefined', async () => {
    const store = new FileCredentialStore(path)
    await store.modify('openai', async () => ({ key: 'first', type: 'api_key' }))

    const seen: unknown[] = []
    const unchanged = await store.modify('openai', async current => {
      seen.push(current)
      return undefined
    })

    expect(seen).toEqual([{ key: 'first', type: 'api_key' }])
    expect(unchanged).toEqual({ key: 'first', type: 'api_key' })
    expect(await store.read('openai')).toEqual({ key: 'first', type: 'api_key' })
  })

  it('serialises concurrent writes so none is lost', async () => {
    const store = new FileCredentialStore(path)
    await Promise.all(
      ['a', 'b', 'c', 'd'].map(id => store.modify(id, async () => ({ key: `key-${id}`, type: 'api_key' })))
    )
    expect((await store.list()).map(entry => entry.providerId).sort()).toEqual(['a', 'b', 'c', 'd'])
  })

  it('writes the file no wider than the owner', async () => {
    const store = new FileCredentialStore(path)
    await store.modify('openai', async () => ({ key: 'sk-synthetic', type: 'api_key' }))
    expect((await stat(path)).mode & 0o777).toBe(0o600)
  })

  it('leaves no temporary file behind, so a reader sees one file or none', async () => {
    const store = new FileCredentialStore(path)
    await store.modify('openai', async () => ({ key: 'sk-synthetic', type: 'api_key' }))
    await store.modify('openai', async () => ({ key: 'sk-rotated', type: 'api_key' }))

    const { readdir } = await import('node:fs/promises')
    expect(await readdir(join(directory, 'nested'))).toEqual(['auth.json'])
    expect(JSON.parse(await readFile(path, 'utf8'))).toEqual({ openai: { key: 'sk-rotated', type: 'api_key' } })
  })

  it('refuses a file that is not a credential map rather than reading past it', async () => {
    const { mkdir } = await import('node:fs/promises')
    await mkdir(join(directory, 'nested'), { recursive: true })
    await writeFile(path, '{"openai": {"key": "x"}}')
    await expect(new FileCredentialStore(path).list()).rejects.toThrow(/not a credential/)

    await writeFile(path, 'not json at all')
    await expect(new FileCredentialStore(path).list()).rejects.toThrow(/not JSON/)
  })
})
