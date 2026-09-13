// Editor autocomplete: pi's `CombinedAutocompleteProvider` for slash commands
// and their arguments, with path completion routed through the gateway first.
//
// The gateway's `complete.path` handler is still a stub that answers with an
// empty list (opendde_harness/tui_rpc/methods/slash_routing.py). The TUI and the
// gateway share a filesystem, so an empty answer falls back to pi's own local
// walk rather than leaving the user with no completions; the day the handler
// grows real suggestions, they win without a change here.

import type { AutocompleteItem, AutocompleteSuggestions, SlashCommand } from '@earendil-works/pi-tui'

import { CombinedAutocompleteProvider } from '@earendil-works/pi-tui'
import { accessSync, constants } from 'node:fs'
import { delimiter, join } from 'node:path'

/** Characters that end a path token, as pi's provider reads them. */
const PATH_DELIMITERS = new Set([' ', '\t', '"', "'", '='])

function findLastDelimiter(text: string): number {
  for (let i = text.length - 1; i >= 0; i -= 1) {
    if (PATH_DELIMITERS.has(text[i] ?? '')) {
      return i
    }
  }

  return -1
}

/**
 * The `@`- or path-shaped token before the cursor, or null when the text is not
 * asking for a path. Mirrors `extractAtPrefix` / `extractPathPrefix` in pi's
 * provider, minus the quoting cases it handles on its own after we defer.
 */
export function pathPrefixBefore(text: string, force = false): null | string {
  const start = findLastDelimiter(text) + 1
  const token = text.slice(start)

  if (token.startsWith('@')) {
    return token
  }

  if (force) {
    return token
  }

  if (token.includes('/') || token.startsWith('.') || token.startsWith('~/')) {
    return token
  }

  return null
}

/** The word to ask the gateway about: the token without its `@` or quote. */
export function completionWord(prefix: string): string {
  if (prefix.startsWith('@"')) {
    return prefix.slice(2)
  }

  if (prefix.startsWith('@') || prefix.startsWith('"')) {
    return prefix.slice(1)
  }

  return prefix
}

export interface RemotePathCompleter {
  (word: string): Promise<AutocompleteItem[]>
}

/**
 * `fd` on PATH, which is what pi's `@` trigger walks the tree with. Without it
 * `@` finds nothing, so the app resolves it once at boot and passes it in.
 * Debian calls the binary `fdfind`.
 */
export function resolveFdPath(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform
): null | string {
  const names = platform === 'win32' ? ['fd.exe'] : ['fd', 'fdfind']

  for (const dir of (env.PATH ?? '').split(delimiter).filter(Boolean)) {
    for (const name of names) {
      const candidate = join(dir, name)

      try {
        accessSync(candidate, constants.X_OK)

        return candidate
      } catch {
        // Try the next directory.
      }
    }
  }

  return null
}

export interface HarnessAutocompleteOptions {
  /** Where relative paths are resolved from: the session's cwd. */
  basePath: string
  commands: (AutocompleteItem | SlashCommand)[]
  /** `complete.path` on the gateway, when the app wants to try it first. */
  completePath?: RemotePathCompleter
  /** Path to `fd`, for pi's `@` file search. */
  fdPath?: null | string
}

export class HarnessAutocompleteProvider extends CombinedAutocompleteProvider {
  private readonly completePath?: RemotePathCompleter

  constructor(options: HarnessAutocompleteOptions) {
    super(options.commands, options.basePath, options.fdPath ?? null)
    this.completePath = options.completePath
  }

  override async getSuggestions(
    lines: string[],
    cursorLine: number,
    cursorCol: number,
    options: { force?: boolean; signal: AbortSignal }
  ): Promise<AutocompleteSuggestions | null> {
    const suggestions = await this.remoteSuggestions(lines, cursorLine, cursorCol, options)

    return suggestions ?? super.getSuggestions(lines, cursorLine, cursorCol, options)
  }

  private async remoteSuggestions(
    lines: string[],
    cursorLine: number,
    cursorCol: number,
    options: { force?: boolean; signal: AbortSignal }
  ): Promise<AutocompleteSuggestions | null> {
    if (!this.completePath) {
      return null
    }

    const before = (lines[cursorLine] ?? '').slice(0, cursorCol)

    // A bare `/name` is a slash command, not a path; pi's provider owns it and
    // the argument completions that follow the first space.
    if (!before.startsWith('@') && !options.force && before.startsWith('/')) {
      return null
    }

    const prefix = pathPrefixBefore(before, options.force)

    if (prefix === null) {
      return null
    }

    try {
      const items = await this.completePath(completionWord(prefix))

      if (options.signal.aborted || items.length === 0) {
        return null
      }

      return { items, prefix }
    } catch {
      return null
    }
  }
}
