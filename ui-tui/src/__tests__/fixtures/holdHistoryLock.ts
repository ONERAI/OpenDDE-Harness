// A second process holding the history lock, for the concurrency tests. It
// takes the same lock file production takes, says so, and holds it until the
// test removes a barrier file.
//
//   tsx fixtures/holdHistoryLock.ts <history file> <barrier file>
//
// Prints `HELD` once the lock is its, and `RELEASED` after it lets go.

import { existsSync, openSync, closeSync, unlinkSync } from 'node:fs'

const [file, barrier] = process.argv.slice(2)

if (!file || !barrier) {
  throw new Error('usage: holdHistoryLock.ts <history file> <barrier file>')
}

const lock = `${file}.lock`
const held = openSync(lock, 'wx', 0o600)

process.stdout.write('HELD\n')

const release = (): void => {
  closeSync(held)

  try {
    unlinkSync(lock)
  } catch {
    // Already gone is gone enough.
  }

  process.stdout.write('RELEASED\n')
  process.exit(0)
}

// Poll rather than watch: the barrier is removed by a process that may be
// blocked in a synchronous wait of its own, and this side must not care.
const poll = setInterval(() => {
  if (!existsSync(barrier)) {
    clearInterval(poll)
    release()
  }
}, 5)
