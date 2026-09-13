// Reading the task root. Everything here is bounded and read-only: the TUI
// never writes a snapshot, never reconciles one (that path checks a PID and
// writes a `failed` snapshot), and never walks into a task's candidates or
// structures. One level of directories, one file each.
//
// The task ID rule stops a path being *spelled* outside the root. It does
// nothing about a path that *resolves* outside it: an ordinary task directory
// can hold a `snapshot.json` symlinked anywhere on the host, and `open`,
// `stat` and `readFile` all follow it. So every file this module reads is
// resolved with `realpath` first and rejected unless it lands inside the
// resolved root, opened without following a final-component symlink, and
// checked to be a regular file on the descriptor that was actually opened.
//
// A name is only a promise until someone renames something above it, so the
// open is anchored to the task directory's own descriptor where the platform
// can do that. Where it cannot — macOS — the path is required to be one nobody
// else can rename through instead, and refused by name when it is not. See
// `openTaskFile`.

import type { Stats } from 'node:fs'
import type { FileHandle } from 'node:fs/promises'

import { constants as FS } from 'node:fs'
import { lstat, open, readdir, realpath } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join, resolve, sep } from 'node:path'

import type { SnapshotFacts } from './types.js'

import { isTaskId, parseSnapshot } from './types.js'

/** A snapshot bigger than this is a broken file, not a task state. */
export const MAX_SNAPSHOT_BYTES = 1024 * 1024

/** How far back `worker.log` is read for a tail. */
export const MAX_LOG_BYTES = 1024 * 1024

/** How many snapshots are read at once. */
export const READ_CONCURRENCY = 4

/** `O_NOFOLLOW` refuses a symlink at the final component, which is the one
 *  `realpath` cannot vouch for any more by the time we open. `O_NONBLOCK`
 *  means a FIFO with no writer fails or opens empty instead of parking the
 *  command forever. Neither exists on Windows, which this UI does not target. */
const O_NOFOLLOW = typeof FS.O_NOFOLLOW === 'number' ? FS.O_NOFOLLOW : 0
const O_NONBLOCK = typeof FS.O_NONBLOCK === 'number' ? FS.O_NONBLOCK : 0
const O_DIRECTORY = typeof FS.O_DIRECTORY === 'number' ? FS.O_DIRECTORY : 0
const OPEN_FLAGS = FS.O_RDONLY | O_NOFOLLOW | O_NONBLOCK
const OPEN_DIR_FLAGS = FS.O_RDONLY | O_DIRECTORY | O_NOFOLLOW

function errorCode(err: unknown): string | undefined {
  return typeof err === 'object' && err !== null ? (err as NodeJS.ErrnoException).code : undefined
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

/** `OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT` with `~` expanded, else the default. */
export function taskRoot(env: NodeJS.ProcessEnv = process.env): string {
  const raw = (env.OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT ?? '').trim()

  if (!raw) {
    return join(homedir(), '.opendde_harness', 'protein_design')
  }

  if (raw === '~') {
    return homedir()
  }

  return raw.startsWith('~/') ? join(homedir(), raw.slice(2)) : raw
}

/**
 * The task root with every symlink in it followed.
 *
 * The root itself may legitimately be a symlink, or sit under one — `/tmp` is
 * on plenty of systems. What matters is that it and the files under it resolve
 * to the same real directory. Null means it is not there, which is the same as
 * having started no designs.
 */
export async function resolveRoot(root: string): Promise<null | string> {
  try {
    return await realpath(resolve(root))
  } catch {
    return null
  }
}

/** Whether `target` is `root` itself or sits below it. Both must be real
 *  paths: this is a string test, and it is only sound on resolved ones. */
export function isInside(root: string, target: string): boolean {
  if (target === root) {
    return true
  }

  return target.startsWith(root.endsWith(sep) ? root : root + sep)
}

/** The task directories under `root`, one level deep. A symlinked directory is
 *  not a directory here: `Dirent` reports it as a link, and it is skipped. */
export async function listTaskIds(root: string): Promise<string[]> {
  try {
    const entries = await readdir(root, { withFileTypes: true })

    return entries.filter(entry => entry.isDirectory() && isTaskId(entry.name)).map(entry => entry.name)
  } catch (err) {
    if (errorCode(err) === 'ENOENT') {
      return []
    }

    throw err
  }
}

type OpenOutcome =
  | { handle: FileHandle; kind: 'ok'; mtimeMs: number; size: number }
  | { kind: 'gone' }
  | { kind: 'unavailable'; reason: string }

/**
 * Platforms where a descriptor can be named as a path again.
 *
 * Linux and Android publish every descriptor under `/proc/self/fd`, and a path
 * resolved through one of those entries starts at the object the descriptor
 * already refers to. That makes it an `openat`, which Node does not otherwise
 * expose. macOS's `/dev/fd` is not the same thing — `/dev/fd/N/child` does not
 * resolve through a directory descriptor there — and nothing else in Node
 * takes a directory descriptor, so there is no equivalent to fall back to.
 */
const ANCHORED_PLATFORMS = new Set(['android', 'linux'])

function descriptorPath(handle: FileHandle): null | string {
  return ANCHORED_PLATFORMS.has(process.platform) ? `/proc/self/fd/${handle.fd}` : null
}

/** The sticky bit, which `node:fs` does not name. On a directory it means only
 *  an entry's owner may rename or remove it, so a shared directory carrying it
 *  cannot be used to swap out something of ours. `/tmp` is the reason it has to
 *  be considered: it is world-writable on every Unix and is nobody's mistake. */
const S_ISVTX = 0o1000

/**
 * Whether a directory can be trusted to still be the one we looked at.
 *
 * The race the anchored open exists to close needs someone who can rename an
 * entry in one of the directories on the way to the file. On a platform with
 * no anchor, that is what gets checked instead: a directory nobody else can
 * write to cannot be swapped, so its name stays a promise for as long as the
 * read takes. Owned by us or by root, and not writable by group or others
 * unless the sticky bit takes that ability back.
 *
 * Returns the reason it cannot, naming the directory, or undefined.
 */
function vetDirectory(info: Stats, where: string, uid: number): string | undefined {
  if (info.isSymbolicLink()) {
    return `${where} is a symlink`
  }

  if (!info.isDirectory()) {
    return `${where} is not a directory`
  }

  if (info.uid !== uid && info.uid !== 0) {
    return `${where} is owned by uid ${info.uid}`
  }

  if ((info.mode & (FS.S_IWGRP | FS.S_IWOTH)) !== 0 && (info.mode & S_ISVTX) === 0) {
    return `${where} is writable by other users`
  }

  return undefined
}

/** Every directory from the filesystem root down to `target`, in that order. */
function ancestry(target: string): string[] {
  const chain: string[] = [sep]
  let at = ''

  for (const part of target.split(sep).filter(Boolean)) {
    at += sep + part
    chain.push(at)
  }

  return chain
}

/**
 * Whether the path down to `realRoot` is private enough to read by name.
 *
 * Walked in full on every read, and nothing about the answer is remembered.
 * The conclusion is about permissions, and permissions change without the
 * objects they sit on changing: a `chmod 0777` leaves every device and inode
 * on the path exactly as it was. A verdict cached against identity therefore
 * goes on saying "private" about a directory the whole world can now write,
 * which is how a foreign uid came to rename a task directory out from under a
 * read and serve a file from outside the root. There is nothing on this path
 * that cannot change, so there is nothing worth caching.
 *
 * The cost is a handful of `lstat` calls per read on the platforms that have
 * no anchored open. That is the price of the answer being current.
 */
async function vetRoot(realRoot: string, uid: number): Promise<string | undefined> {
  for (const component of ancestry(realRoot)) {
    let info: Stats

    try {
      info = await lstat(component)
    } catch (err) {
      return `${component} could not be checked: ${message(err)}`
    }

    const refusal = vetDirectory(info, component, uid)

    if (refusal) {
      return refusal
    }
  }

  return undefined
}

/**
 * Whether `dirPath` still names the directory `dir` holds open.
 *
 * The window between the last check and a name-based open is the one thing an
 * anchored open would close and this cannot. It is bounded twice: the walk
 * leaves nobody but this user and root able to rename a component, and this
 * reads the name back afterwards. Something that swapped the directory during
 * the window and left it swapped is caught here; something that swapped it and
 * put it back is not, and needs the write access the walk has just ruled out
 * for everyone but the user themselves.
 */
async function stillTheSameDirectory(dirPath: string, dir: FileHandle): Promise<boolean> {
  try {
    const [byName, byDescriptor] = await Promise.all([lstat(dirPath), dir.stat()])

    return byName.dev === byDescriptor.dev && byName.ino === byDescriptor.ino
  } catch {
    return false
  }
}

/** The path down to one task directory, vetted: the chain to the root, then
 *  the directory itself judged on the descriptor that is already open for it.
 *  Returns why it cannot be read by name, or undefined. */
async function vetByPath(realRoot: string, dirPath: string, dir: FileHandle): Promise<string | undefined> {
  const uid = process.getuid?.()

  if (uid === undefined) {
    return `${realRoot} cannot be vetted: this platform has no user ids`
  }

  return (await vetRoot(realRoot, uid)) ?? vetDirectory(await dir.stat(), dirPath, uid)
}

/**
 * Open one file in one task's directory, or say why not.
 *
 * `realpath` validates a *name*, and a name stops being a promise the moment
 * it returns: renaming `<root>/<task>` to a symlink between the check and the
 * open redirects every component after it, and `O_NOFOLLOW` guards only the
 * last one. So the task directory is opened first and everything else is done
 * against that descriptor:
 *
 * 1. The directory is opened with `O_DIRECTORY | O_NOFOLLOW`, so a task
 *    directory that is a symlink is refused rather than followed.
 * 2. Where that descriptor actually is, is read back from the descriptor and
 *    required to be inside the root. This is the containment check, and it is
 *    about an object rather than about a name.
 * 3. The file is opened from that descriptor, so no later rename of anything
 *    above it can change what is opened, and `O_NOFOLLOW` refuses a symlinked
 *    file. `O_NONBLOCK` keeps a FIFO from parking the caller, and the
 *    regular-file check is made on the descriptor, not on the name.
 *
 * Step 3 needs a descriptor-anchored open, and macOS has none. An identity
 * check either side of a name-based open is no substitute on its own:
 * swapping the directory for a symlink, opening through it and putting the
 * directory back leaves both samples identical and the descriptor pointing
 * outside the root. What is left is to make the swap impossible rather than to
 * detect it. The whole path from the filesystem root down is required to be
 * private — every component owned by this user or by root, none of them
 * writable by anyone else unless the sticky bit takes that back — and a path
 * nobody else can rename through is a path that still means what it meant when
 * it was resolved. Where that does not hold, the root is refused with the
 * component that spoiled it named, which is something its owner can fix. The
 * walk is redone on every read, because what it decides is about permissions
 * and permissions move without moving the objects they sit on.
 *
 * That draws the line where the threat actually is. The task root lives in the
 * user's own directory and is written by the harness's own worker, so winning
 * this race needs write access to the user's files, which is not a boundary
 * this check can defend anyway.
 */
/**
 * Why the task directory would not open.
 *
 * `O_DIRECTORY | O_NOFOLLOW` on a link to a directory answers ENOTDIR on Linux
 * and ELOOP elsewhere, and ENOTDIR is also what a plain file answers, so the
 * name is looked at without following it rather than guessed at from the errno.
 */
async function classifyDirectory(dirPath: string, err: unknown): Promise<OpenOutcome> {
  const code = errorCode(err)

  if (code === 'ENOENT' || code === 'ENOTDIR' || code === 'ELOOP') {
    try {
      if ((await lstat(dirPath)).isSymbolicLink()) {
        return { kind: 'unavailable', reason: 'the task directory is a symlink' }
      }
    } catch {
      // Gone between the open and the look. Gone is gone.
    }

    return { kind: 'gone' }
  }

  return { kind: 'unavailable', reason: `the task directory could not be opened: ${message(err)}` }
}

async function openTaskFile(realRoot: string, taskId: string, name: string, what: string): Promise<OpenOutcome> {
  if (!isTaskId(taskId)) {
    return { kind: 'unavailable', reason: 'invalid task id' }
  }

  const dirPath = join(realRoot, taskId)
  let dir: FileHandle

  try {
    dir = await open(dirPath, OPEN_DIR_FLAGS)
  } catch (err) {
    return { ...(await classifyDirectory(dirPath, err)) }
  }

  try {
    const anchor = descriptorPath(dir)
    let filePath: string

    if (anchor === null) {
      // No `openat` to be had. Containment comes from the path instead: the
      // walk down to the root says nobody else can rename a component of it,
      // and the task directory is judged on the descriptor just opened without
      // following a link, so `<root>/<id>` is inside the root by construction.
      const refusal = await vetByPath(realRoot, dirPath, dir)

      if (refusal) {
        return { kind: 'unavailable', reason: `${what} was not read: ${refusal}` }
      }

      filePath = join(dirPath, name)
    } else {
      let dirReal: string

      try {
        dirReal = await realpath(anchor)
      } catch (err) {
        // The descriptor is open; it is the facility for naming it that is
        // missing. Saying the task is gone would blame the wrong thing.
        return { kind: 'unavailable', reason: `${what} was not read: ${anchor} is unavailable: ${message(err)}` }
      }

      if (!isInside(realRoot, dirReal)) {
        return { kind: 'unavailable', reason: `${what} is in a directory outside the task root` }
      }

      filePath = join(anchor, name)
    }

    const handle = await open(filePath, OPEN_FLAGS)

    try {
      const info = await handle.stat()

      if (!info.isFile()) {
        await handle.close()

        return { kind: 'unavailable', reason: `${what} is not a regular file` }
      }

      // Only the branch that opened by name has a window to check.
      if (anchor === null && !(await stillTheSameDirectory(dirPath, dir))) {
        await handle.close()

        return {
          kind: 'unavailable',
          reason: `${what} was not read: the task directory moved while it was being opened`
        }
      }

      return { handle, kind: 'ok', mtimeMs: info.mtimeMs, size: info.size }
    } catch (err) {
      await handle.close().catch(() => {})

      return { kind: 'unavailable', reason: `${what} could not be read: ${message(err)}` }
    }
  } catch (err) {
    const code = errorCode(err)

    if (code === 'ENOENT' || code === 'ENOTDIR') {
      return { kind: 'gone' }
    }

    if (code === 'ELOOP') {
      return { kind: 'unavailable', reason: `${what} is a symlink` }
    }

    return { kind: 'unavailable', reason: `${what} could not be opened: ${message(err)}` }
  } finally {
    await dir.close().catch(() => {})
  }
}

/** Read at most `limit` bytes from the start of `handle`, plus one more so an
 *  overlong file is recognised as overlong rather than silently clipped. */
async function readBounded(handle: FileHandle, size: number, limit: number): Promise<Buffer> {
  const buffer = Buffer.alloc(Math.min(size, limit) + 1)
  const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0)

  return buffer.subarray(0, bytesRead)
}

export type SnapshotRead =
  /** `snapshot.json` is not there (yet, or any more). */
  | { kind: 'gone' }
  | { facts: SnapshotFacts; kind: 'ok'; mtimeMs: number; size: number }
  /** Same mtime and size as the last read: nothing to parse. */
  | { kind: 'unchanged' }
  | { kind: 'unavailable'; reason: string }

/** What the last successful read saw, so an unchanged file is skipped. */
export interface SnapshotStamp {
  mtimeMs?: number
  size?: number
}

/**
 * Read one task's snapshot. `realRoot` comes from {@link resolveRoot}.
 *
 * The file is opened once and every decision — the size cap, the unchanged
 * check and the stamp that is stored — is taken from that descriptor. Stat'ing
 * a name and then reading the same name again is two different files whenever
 * the worker replaces the snapshot in between, which it does by design: it
 * writes and renames.
 */
export async function readSnapshot(realRoot: string, taskId: string, seen: SnapshotStamp = {}): Promise<SnapshotRead> {
  if (!isTaskId(taskId)) {
    return { kind: 'unavailable', reason: 'invalid task id' }
  }

  const opened = await openTaskFile(realRoot, taskId, 'snapshot.json', 'snapshot.json')

  if (opened.kind !== 'ok') {
    return opened
  }

  const { handle, mtimeMs, size } = opened

  try {
    if (size > MAX_SNAPSHOT_BYTES) {
      return { kind: 'unavailable', reason: `snapshot.json is larger than ${MAX_SNAPSHOT_BYTES} bytes` }
    }

    if (seen.mtimeMs === mtimeMs && seen.size === size) {
      return { kind: 'unchanged' }
    }

    const bytes = await readBounded(handle, size, MAX_SNAPSHOT_BYTES)

    // The file grew past the cap between the stat and the read. Both figures
    // came from this one descriptor, so this is the file we would have parsed.
    if (bytes.length > MAX_SNAPSHOT_BYTES) {
      return { kind: 'unavailable', reason: `snapshot.json is larger than ${MAX_SNAPSHOT_BYTES} bytes` }
    }

    let parsed: unknown

    try {
      parsed = JSON.parse(bytes.toString('utf8'))
    } catch (err) {
      return { kind: 'unavailable', reason: `snapshot.json could not be parsed: ${message(err)}` }
    }

    const facts = parseSnapshot(parsed, taskId)

    if (!facts) {
      return { kind: 'unavailable', reason: 'snapshot.json does not describe this task' }
    }

    return { facts, kind: 'ok', mtimeMs, size: bytes.length }
  } catch (err) {
    return { kind: 'unavailable', reason: `snapshot.json could not be read: ${message(err)}` }
  } finally {
    await handle.close().catch(() => {})
  }
}

export interface LogTail {
  /** The log's size. Zero is an empty file; a positive size with no lines is a
   *  window that held no complete line, which is a very different thing. */
  bytes: number
  lines: string[]
  /** There is no readable `worker.log` for this task on this host. */
  missing: boolean
  /** Why there is none, when that is worth saying. */
  reason?: string
  /** The byte cap stopped us short of `maxLines`. */
  truncated: boolean
}

/**
 * The end of a task's worker log. `realRoot` comes from {@link resolveRoot}.
 *
 * Reads at most the last {@link MAX_LOG_BYTES}. A window that does not start at
 * the beginning of the file can start inside a line — and therefore inside a
 * UTF-8 sequence — so everything before the first newline in it is dropped
 * before the bytes are decoded. A window with no newline in it at all yields no
 * lines, which the caller reports as a read limit rather than as an empty file.
 */
export async function readLogTail(realRoot: string, taskId: string, maxLines: number): Promise<LogTail> {
  if (!isTaskId(taskId)) {
    throw new Error(`invalid protein-design task id: ${taskId}`)
  }

  const opened = await openTaskFile(realRoot, taskId, 'worker.log', 'worker.log')

  if (opened.kind === 'gone') {
    return { bytes: 0, lines: [], missing: true, truncated: false }
  }

  if (opened.kind === 'unavailable') {
    return { bytes: 0, lines: [], missing: true, reason: opened.reason, truncated: false }
  }

  const { handle, size } = opened

  try {
    const start = Math.max(0, size - MAX_LOG_BYTES)
    const length = size - start
    const buffer = Buffer.alloc(length)

    if (length > 0) {
      await handle.read(buffer, 0, length, start)
    }

    let from = 0

    if (start > 0) {
      const newline = buffer.indexOf(0x0a)

      from = newline === -1 ? length : newline + 1
    }

    const all = buffer.subarray(from).toString('utf8').split('\n')

    if (all.at(-1) === '') {
      all.pop()
    }

    const lines = all.slice(-maxLines)

    return { bytes: size, lines, missing: false, truncated: start > 0 && lines.length < maxLines }
  } finally {
    await handle.close().catch(() => {})
  }
}

/** Run `work` over `items`, at most `limit` at a time. */
export async function mapLimit<T>(items: readonly T[], limit: number, work: (item: T) => Promise<void>): Promise<void> {
  let next = 0

  const runner = async (): Promise<void> => {
    for (let index = next++; index < items.length; index = next++) {
      await work(items[index]!)
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, runner))
}
