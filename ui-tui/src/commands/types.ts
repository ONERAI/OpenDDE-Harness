// What a slash command is, and what the app hands it.
//
// The context is deliberately an interface of small accessors rather than the
// app itself: commands are then testable against a handful of fakes, and the
// pieces WP2 owns (the transcript) and phase 2 adds (the pickers) are named here
// without this package depending on them.

import type { AutocompleteItem } from '@earendil-works/pi-tui'
import type { KeybindingsManager } from '@earendil-works/pi-tui'

import type { BusyInputMode } from '../components/queue.js'
import type { SystemTone } from '../components/systemLine.js'
import type { Gateway } from '../gateway.js'
import type { RendererMode } from '../renderer.js'
import type { CommandsCatalogResult } from '../rpc/index.js'
import type { LogTail } from '../tasks/files.js'
import type { TaskResolution } from '../tasks/monitor.js'
import type { TaskRecord } from '../tasks/types.js'

export interface TranscriptEntry {
  role: 'assistant' | 'user'
  text: string
}

/** The prompt `/retry` sends again, kept apart from the visible transcript. */
export interface RetryCandidate {
  /** Whether the gateway still holds an exchange for this prompt. False once
   *  `/undo` removed it, and false when the turn stored nothing because it
   *  failed before producing anything. Either way `/retry` must resend without
   *  undoing first, or it would take the exchange before this one instead. */
  onServer: boolean
  text: string
}

export interface TranscriptAccess {
  /** Drop the last *saved* user/assistant pair, after `session.undo` removed
   *  it server-side, along with anything shown after it. Returns whether an
   *  exchange was found: a transcript whose only echoes were never stored has
   *  nothing to drop, and saying so beats leaving deleted history on screen
   *  without a word. */
  dropLastExchange(): boolean
  /** What is on screen, oldest first. Feeds /copy and /history. */
  history(): TranscriptEntry[]
  /** The last prompt this session sent, or null before it has sent one. Held
   *  separately from the transcript because `/undo` takes the exchange off
   *  screen and `/retry` still has to resend that prompt, not the one before. */
  retryCandidate(): null | RetryCandidate
  /** One dim line. */
  print(text: string, tone?: SystemTone): void
  /** A wrapped block, optionally under a title. Commands never open a pager. */
  printBlock(text: string, title?: string): void
}

/** The model and session facts the footer and panel show. */
export interface SessionInfoView {
  /** Something compacts this session before its context window runs out — the
   *  backend server-side, or the context engine. The footer says `(auto)`. */
  autoCompact?: boolean
  /** What the session already holds, as the gateway estimated it when the
   *  session was opened. The footer's starting point until a turn reports a
   *  real figure: a resumed conversation is in the window from its first
   *  call, and opening it at nothing said the opposite.
   *
   *  Null when the gateway says nothing measures it — a compaction marker its
   *  backend replays stands in front of the stored history, so the messages
   *  before it are not what the next call sends. */
  contextTokens?: null | number
  contextWindow?: number
  effort?: string
  fast?: boolean
  model?: string
  provider?: string
  /** The model is billed by plan, so the cost beside it is not what this
   *  conversation costs. The footer says `(sub)`. */
  subscription?: boolean
  title?: string
}

export interface SessionAccess {
  /** Why a session switch cannot start right now, or null when it can. Says
   *  more than {@link SessionAccess.busy}: a running turn, a cancel still
   *  unwinding, a switch already in flight and messages still queued all
   *  refuse the switch, and they ask for different patience. */
  blockedReason(): null | string
  /** Adopt a session the server minted for us (a fork). Resolves once the new
   *  session is attached, so the command that asked cannot report a fork the
   *  UI is not yet on. The fork inherits its parent's model and provider, and
   *  the caller supplies only the title the server gave it. */
  adopt(id: string, title?: string): Promise<void>
  /** True while a turn is in flight; session switches wait for it. */
  busy(): boolean
  /** Start a fresh session, optionally titled. */
  create(title?: string): Promise<void>
  /** The active session key, or null before one is open. */
  id(): null | string
  info(): SessionInfoView
  patchInfo(patch: Partial<SessionInfoView>): void
  /** Open the session picker. */
  pick(): void
  /** Delete a session, switching away from it when it is the active one. */
  remove(id: string): Promise<boolean>
  /**
   * Adopt `id`. `stillCurrent` is checked at the commit point, so a picker that
   * closed — or a session opened after this one was asked for — cannot be
   * replaced by a late answer.
   */
  resume(id: string, stillCurrent?: () => boolean): Promise<void>
  /** Start a turn with this text. */
  send(text: string): void
}

export type DetailsMode = 'collapsed' | 'expanded' | 'hidden'
export type DetailsSection = 'thinking' | 'tools'

/** What a *collapsed* tool row looks like: one line with its output hidden, or
 *  the first lines of the result. `/quiet-tools` is the only thing that moves
 *  it, and it moves nothing the model is sent. */
export interface QuietToolsAccess {
  get(): boolean
  set(quiet: boolean): void
}

/** The protein-design task monitor, as the local `/tasks` and `/task` see it.
 *  Everything here reads the local task root; nothing starts, stops or asks a
 *  worker for anything. */
export interface TaskAccess {
  /** Put a task first in the bar. */
  focus(taskId: null | string): void
  /** Most recently observed first. */
  list(): TaskRecord[]
  /** The end of a task's worker log. */
  readLogTail(taskId: string, maxLines: number): Promise<LogTail>
  /** One pass over the task root. Callable while the bar is hidden. */
  refresh(): Promise<void>
  resolve(query: string): TaskResolution
  /** Why the task root could not be listed, or null. */
  rootFailure(): null | string
  rootPath(): string
  /** The session generation, checked before an awaited result is printed: the
   *  tasks are root-global, but the transcript they print into is not. */
  sessionEpoch(): number
  setVisible(visible: boolean): void
  visible(): boolean
}

/** The renderer `/fullscreen` drives. Absent when the app was constructed with
 *  a plain TUI rather than a renderer owner, as most tests are. */
export interface RendererAccess {
  mode(): RendererMode
  /** Switch modes. Returns null on success, or why it was refused. */
  switchTo(mode: RendererMode): null | string
}

export interface CommandContext {
  /** What Enter does while a turn is running, as the live queue is behaving.
   *  The gateway does not serve `busy`, so asking it would answer with the
   *  default rather than with the mode in force. */
  busyMode(): BusyInputMode
  /** The command catalog `commands.catalog` served at boot, for /help. */
  catalog(): CommandsCatalogResult | null
  gateway: Gateway
  keybindings: KeybindingsManager
  /** Open pi's sign-in flow: the authentication method, then the provider, then
   *  the login itself. With a provider named, straight to that one. */
  openLoginPicker(provider?: string): void
  /** Open pi's sign-out list: the sign-ins this machine holds. */
  openLogoutPicker(): void
  /** Open pi's model list, for this session or for the default; `query` is
   *  typed into its search first. */
  openModelPicker(scope: 'default' | 'session', query?: string): void
  /** Open pi's scoped-models list: which models `/model` shows under "scoped". */
  openScopedModelsPicker(): void
  /** Open the thinking-level picker. */
  openThinkingPicker(): void
  quietTools: QuietToolsAccess
  quit(): void
  /** Undefined when this build of the app owns no renderer of its own. */
  renderer?: RendererAccess
  session: SessionAccess
  /** Apply `display.busy_input_mode` to the live queue. */
  setBusyMode(mode: BusyInputMode): void
  /** Persist `tui.show_token_usage`. Resolves to whether the gateway applied
   *  it; the footer only changes when it did. */
  setShowTokenUsage(value: boolean): Promise<boolean>
  /** Append the help panel to the transcript. It never takes focus. */
  showHelp(): void
  /** Whether the footer is showing token counters right now. */
  showTokenUsage(): boolean
  tasks: TaskAccess
  transcript: TranscriptAccess
}

export interface Command {
  /** Shown after the name in the completion popup, e.g. `<level>`. */
  argumentHint?: string
  description: string
  /** Completions for this command's argument, driven by pi's autocomplete. */
  getArgumentCompletions?(prefix: string, ctx: CommandContext): AutocompleteItem[] | null
  /** Kept out of completion and /help; still runs when typed in full. */
  hidden?: boolean
  name: string
  run(ctx: CommandContext, arg: string): Promise<void> | void
}

/** `/name rest` split into its parts. `name` is lowercased. */
export interface ParsedCommand {
  arg: string
  name: string
  /** The whole line, as typed. */
  text: string
}

export function looksLikeCommand(text: string): boolean {
  return /^\/[^\s/]*(?:\s|$)/.test(text)
}

export function parseCommand(text: string): ParsedCommand {
  const [name = '', ...rest] = text.trim().slice(1).split(/\s+/)

  return { arg: rest.join(' '), name: name.toLowerCase(), text: text.trim() }
}

/** Turn a list of words into completion items. */
export function wordItems(words: readonly string[], prefix: string): AutocompleteItem[] {
  const token = prefix.trim().toLowerCase()

  return words.filter(word => word.startsWith(token)).map(word => ({ label: word, value: word }))
}
