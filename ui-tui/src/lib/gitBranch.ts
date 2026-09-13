// The current git branch for the footer, read from `.git/HEAD` and refreshed by
// an fs watcher. Reading the file is what pi does before falling back to
// `git symbolic-ref`; the file answers in microseconds and is correct for every
// case the footer cares about (branch, detached HEAD, worktree, no repo).

import type { FSWatcher } from 'node:fs'

import { existsSync, readFileSync, statSync, watch } from 'node:fs'
import { basename, dirname, join, resolve } from 'node:path'

const DEBOUNCE_MS = 500

/** `.git/HEAD` for `cwd`'s repository, following a worktree's `.git` file. */
export function findHeadFile(cwd: string): null | string {
  let dir = resolve(cwd)

  for (;;) {
    const gitPath = join(dir, '.git')

    if (existsSync(gitPath)) {
      try {
        const stat = statSync(gitPath)

        if (stat.isDirectory()) {
          const head = join(gitPath, 'HEAD')

          return existsSync(head) ? head : null
        }

        if (stat.isFile()) {
          const content = readFileSync(gitPath, 'utf8').trim()

          if (!content.startsWith('gitdir: ')) {
            return null
          }

          const head = join(resolve(dir, content.slice('gitdir: '.length).trim()), 'HEAD')

          return existsSync(head) ? head : null
        }
      } catch {
        return null
      }
    }

    const parent = dirname(dir)

    if (parent === dir) {
      return null
    }

    dir = parent
  }
}

/** The branch name in a `HEAD` file's contents, or `detached` for a raw sha. */
export function parseHead(contents: string): null | string {
  const text = contents.trim()

  if (!text) {
    return null
  }

  const ref = /^ref:\s*refs\/heads\/(.+)$/.exec(text)

  return ref ? ref[1]!.trim() : 'detached'
}

/**
 * The branch, cached and kept current. `null` outside a repository.
 *
 * `dispose()` closes the watcher; without it the process would not exit.
 */
export class GitBranch {
  private branch: null | string = null
  private readonly headFile: null | string
  private timer: null | ReturnType<typeof setTimeout> = null
  private watcher: FSWatcher | null = null

  constructor(
    cwd: string,
    private readonly onChange: () => void = () => {},
    /** How long to let a burst of watcher events settle. Shortened by the
     *  tests, which cannot wait half a second per replacement. */
    private readonly debounceMs: number = DEBOUNCE_MS
  ) {
    this.headFile = findHeadFile(cwd)
    this.branch = this.read()
    this.watch()
  }

  get(): null | string {
    return this.branch
  }

  dispose(): void {
    if (this.timer) {
      clearTimeout(this.timer)
      this.timer = null
    }

    this.watcher?.close()
    this.watcher = null
  }

  private read(): null | string {
    if (!this.headFile) {
      return null
    }

    try {
      return parseHead(readFileSync(this.headFile, 'utf8'))
    } catch {
      return null
    }
  }

  private watch(): void {
    if (!this.headFile) {
      return
    }

    const head = basename(this.headFile)

    try {
      // The directory, not the file: a checkout writes a new HEAD and renames
      // it over the old one, and watching the file watches the inode that was
      // just unlinked. The first replacement would still wake this watcher and
      // the second one never would. The directory outlives every replacement.
      this.watcher = watch(dirname(this.headFile), (_event, name) => {
        // `name` is null on the platforms that do not report one; re-read then
        // rather than miss the change.
        if (name != null && basename(String(name)) !== head) {
          return
        }

        // Events arrive in bursts; coalesce them and re-read once they settle.
        if (this.timer) {
          clearTimeout(this.timer)
        }

        this.timer = setTimeout(() => {
          this.timer = null

          const next = this.read()

          if (next !== this.branch) {
            this.branch = next
            this.onChange()
          }
        }, this.debounceMs)

        this.timer.unref?.()
      })
    } catch {
      // No watcher: the branch shown is the one from startup.
    }
  }
}
