// A persistent `CredentialStore` for pi-ai: one JSON file, one writer at a time.
//
// pi-ai keys credentials by provider id and owns the read-modify-write pattern
// (`Models.getAuth()` refreshes a rotated OAuth token inside `modify`), so the
// store's only job is to make that serial and to land the bytes atomically at
// mode 0600. The whole file is rewritten on every write, so the lock is one
// chain for the file rather than one per provider id: two providers writing at
// once through separate chains would each write back the snapshot they read
// and one would lose its entry.

import type { AuthOperationOptions, Credential, CredentialInfo, CredentialStore } from '@earendil-works/pi-ai'

import { randomBytes } from 'node:crypto'
import { chmod, mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { basename, dirname, join } from 'node:path'

type Stored = Record<string, Credential>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** The file's contents as credentials, or an error saying what is wrong with it. */
function parse(path: string, text: string): Stored {
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch (error) {
    throw new Error(`${path} is not JSON: ${(error as Error).message}`)
  }
  if (!isRecord(raw)) {
    throw new Error(`${path} does not hold a credential map`)
  }
  const stored: Stored = {}
  for (const [providerId, credential] of Object.entries(raw)) {
    if (!isRecord(credential) || (credential.type !== 'api_key' && credential.type !== 'oauth')) {
      throw new Error(`${path} holds an entry for ${providerId} that is not a credential`)
    }
    stored[providerId] = credential as unknown as Credential
  }
  return stored
}

export class FileCredentialStore implements CredentialStore {
  /** Every write, and every read a write depends on, in the order they arrived. */
  private chain: Promise<unknown> = Promise.resolve()

  constructor(readonly path: string) {}

  async delete(providerId: string, options?: AuthOperationOptions): Promise<void> {
    await this.serialise(async () => {
      const all = await this.load()
      if (!(providerId in all)) {
        return
      }
      delete all[providerId]
      await this.save(all)
    }, options)
  }

  async list(options?: AuthOperationOptions): Promise<readonly CredentialInfo[]> {
    options?.signal?.throwIfAborted()
    return Object.entries(await this.load()).map(([providerId, credential]) => ({
      providerId,
      type: credential.type
    }))
  }

  modify(
    providerId: string,
    fn: (current: Credential | undefined) => Promise<Credential | undefined>,
    options?: AuthOperationOptions
  ): Promise<Credential | undefined> {
    return this.serialise(async () => {
      const all = await this.load()
      const current = all[providerId]
      const next = await fn(current)
      options?.signal?.throwIfAborted()
      if (next === undefined) {
        return current
      }
      all[providerId] = next
      await this.save(all)
      return next
    }, options)
  }

  async read(providerId: string, options?: AuthOperationOptions): Promise<Credential | undefined> {
    options?.signal?.throwIfAborted()
    return (await this.load())[providerId]
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

  /** Temp file, chmod, rename: a reader sees the old file or the new one, never a half-written one. */
  private async save(all: Stored): Promise<void> {
    const directory = dirname(this.path)
    await mkdir(directory, { mode: 0o700, recursive: true })
    const temp = join(directory, `.${basename(this.path)}.${randomBytes(6).toString('hex')}`)
    try {
      await writeFile(temp, `${JSON.stringify(all, null, 2)}\n`, { mode: 0o600 })
      // `mode` above is masked by the umask; this is what actually guarantees 0600.
      await chmod(temp, 0o600)
      await rename(temp, this.path)
    } catch (error) {
      await rm(temp, { force: true })
      throw error
    }
  }

  private serialise<T>(task: () => Promise<T>, options?: AuthOperationOptions): Promise<T> {
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
    // A failed task must not poison the queue behind it.
    this.chain = queued.catch(() => {})
    return queued
  }
}
