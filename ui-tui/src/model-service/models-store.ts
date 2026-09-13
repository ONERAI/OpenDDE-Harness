// A persistent `ModelsStore` for pi-ai: the last list each dynamic provider
// published, so a relay's models are on screen at the next start before the
// endpoint has been asked again.
//
// One JSON file keyed by provider id, written whole through a temp file and a
// rename, and serialised on one chain the way the credential store is: two
// providers refreshing at once would each write back the snapshot they read
// and one would lose its entry.

import type { ModelsStore, ModelsStoreEntry, ModelsStoreOperationOptions } from '@earendil-works/pi-ai'

import { randomBytes } from 'node:crypto'
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { basename, dirname, join } from 'node:path'

type Stored = Record<string, ModelsStoreEntry>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parse(path: string, text: string): Stored {
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch (error) {
    throw new Error(`${path} is not JSON: ${(error as Error).message}`)
  }
  if (!isRecord(raw)) {
    throw new Error(`${path} does not hold a models map`)
  }
  const stored: Stored = {}
  for (const [providerId, entry] of Object.entries(raw)) {
    if (!isRecord(entry) || !Array.isArray(entry.models)) {
      throw new Error(`${path} holds an entry for ${providerId} that is not a model list`)
    }
    stored[providerId] = entry as unknown as ModelsStoreEntry
  }
  return stored
}

export class FileModelsStore implements ModelsStore {
  private chain: Promise<unknown> = Promise.resolve()

  constructor(readonly path: string) {}

  async read(providerId: string, options?: ModelsStoreOperationOptions): Promise<ModelsStoreEntry | undefined> {
    options?.signal?.throwIfAborted()
    const entry = (await this.load())[providerId]
    return entry ? structuredClone(entry) : undefined
  }

  write(providerId: string, entry: ModelsStoreEntry, options?: ModelsStoreOperationOptions): Promise<void> {
    return this.serialise(async () => {
      const all = await this.load()
      all[providerId] = structuredClone(entry)
      await this.save(all)
    }, options)
  }

  delete(providerId: string, options?: ModelsStoreOperationOptions): Promise<void> {
    return this.serialise(async () => {
      const all = await this.load()
      if (!(providerId in all)) {
        return
      }
      delete all[providerId]
      await this.save(all)
    }, options)
  }

  private async load(): Promise<Stored> {
    let text: string
    try {
      text = await readFile(this.path, 'utf8')
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        return {}
      }
      throw error
    }
    return parse(this.path, text)
  }

  private async save(all: Stored): Promise<void> {
    const directory = dirname(this.path)
    await mkdir(directory, { mode: 0o700, recursive: true })
    const temp = join(directory, `.${basename(this.path)}.${randomBytes(6).toString('hex')}`)
    try {
      await writeFile(temp, `${JSON.stringify(all, null, 2)}\n`)
      await rename(temp, this.path)
    } catch (error) {
      await rm(temp, { force: true })
      throw error
    }
  }

  private serialise<T>(task: () => Promise<T>, options?: ModelsStoreOperationOptions): Promise<T> {
    const queued = this.chain.then(
      () => {
        options?.signal?.throwIfAborted()
        return task()
      },
      () => {
        options?.signal?.throwIfAborted()
        return task()
      }
    )
    this.chain = queued.catch(() => {})
    return queued
  }
}
