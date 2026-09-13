// One tool call in the transcript: a panel whose background says what state
// the call is in, a line naming the call, a progress line while it runs and
// what it returned.
//
// Shaped after pi's `ToolExecutionComponent`, minus everything the gateway
// never sends: `tool.complete` carries only `{tool_call_id, result_preview,
// truncated}`, so there are no images, no structured details and no per-tool
// renderers.
//
// It carries no error flag either, and a preview is the tool's *output*, not
// its status: `read` returning a document that opens with "Error handling
// guide" succeeded. A completed call is therefore drawn as completed. The only
// failure this component paints is one it was told about — `abandon`, when the
// turn ended with no result. The day `ToolCompletePayload` grows an `is_error`,
// `complete` takes it and paints it.
//
// Collapsed, a call is *one line*: what was invoked, truncated to the width,
// with the expand key named at its end. A successful call's output is not shown
// at all — the panel says which file was read, and the file itself is one key
// away. That is the rule of the owner's `pi-fold` extension, brought to our own
// rows. A failure keeps its first error line under the invocation: a tool that
// failed has to be visible without being expanded first. `setQuiet(false)`
// gives a session the older behaviour back, the first {@link PREVIEW_LINES} of
// every result.
//
// None of this reaches the model. The panel draws `result_preview`; what the
// tool returned to the conversation was decided by the gateway before this
// component saw a word of it.

import type { Component, TUI } from '@earendil-works/pi-tui'

import { Box, Container, MouseRegion, Spacer, Text, truncateToWidth, visibleWidth } from '@earendil-works/pi-tui'

import type { JsonValue } from '../rpc/index.js'
import type { ColorScheme, Theme } from '../theme.js'
import type { CountdownClock } from './countdownTimer.js'

import { waitLine, waitPhrases } from '../content/charms.js'
import { sanitizeLine } from '../tasks/types.js'
import { CountdownTimer } from './countdownTimer.js'

/** Lines of result kept when the panel is collapsed and not quiet. */
const PREVIEW_LINES = 10

/** A call still pending after this long grows one quiet reassurance row. */
const REASSURE_AT_MS = 8_000

/** And after this long that same row is rewritten, once. There is no third. */
const REASSURE_AGAIN_MS = 18_000

/** Characters of rendered arguments kept when a tool names no display label. */
const ARG_PREVIEW_CHARS = 72

/** Columns the invocation itself must keep before the expand hint is worth the
 *  room: a row that is only a hint says nothing. pi-fold draws the same line. */
const MIN_INVOCATION = 8

type ToolState = 'error' | 'pending' | 'success'

const STATE_BG = {
  error: 'toolErrorBg',
  pending: 'toolPendingBg',
  success: 'toolSuccessBg'
} as const

/**
 * A whole escape sequence — OSC with either terminator, and CSI — in text the
 * tool wrote.
 *
 * {@link sanitizeLine} takes the control characters out; taking the sequence
 * out first is what keeps its `[0m` tail from being drawn as text.
 */
// eslint-disable-next-line no-control-regex
const ESCAPE_SEQUENCE = /\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]/g

/**
 * pi-tui's `truncateToWidth` brackets its ellipsis with full resets
 * (`\x1b[0m`), which would end the panel's background there and leave the rest
 * of the row in the terminal's own colours.
 *
 * Everything painted here is chalk's, which closes with targeted codes, and
 * every piece of foreign text has been through {@link inlineText} — so a full
 * reset in a truncated line was put there by the truncation itself. Downgraded
 * to "default foreground, bold off", the enclosing background survives.
 * pi-fold does this to pi's own rows for the same reason.
 */
// eslint-disable-next-line no-control-regex
const FULL_RESET = /\x1b\[0m/g

/** One line of a tool's own text: no sequences, no newlines, bounded. */
function inlineText(value: string): string {
  return sanitizeLine(value.replace(ESCAPE_SEQUENCE, ' '))
}

/** `text` cut to `width`, with the panel's background left intact. */
function fit(text: string, width: number): string {
  return truncateToWidth(text, width).replace(FULL_RESET, '\x1b[39m\x1b[22m')
}

/** One line that can be rewritten in place. `TruncatedText` truncates but has
 *  no setter, and `Text` has a setter but wraps. */
class Line implements Component {
  private cachedLines?: string[]
  private cachedWidth?: number
  private suffix = ''
  private text = ''

  /** `suffix` is kept whole: the text is truncated to leave room for it, so a
   *  long invocation never eats the expand hint at the end of the row. */
  setText(text: string, suffix = ''): void {
    if (text !== this.text || suffix !== this.suffix) {
      this.text = text
      this.suffix = suffix
      this.invalidate()
    }
  }

  invalidate(): void {
    this.cachedLines = undefined
    this.cachedWidth = undefined
  }

  render(width: number): string[] {
    if (this.cachedLines && this.cachedWidth === width) {
      return this.cachedLines
    }

    this.cachedWidth = width
    this.cachedLines = [this.compose(width)]

    return this.cachedLines
  }

  private compose(width: number): string {
    const room = this.suffix ? visibleWidth(this.suffix) + 1 : 0

    if (room > 0 && width > room + MIN_INVOCATION) {
      return `${fit(this.text, width - room)} ${this.suffix}`
    }

    return fit(this.text, width)
  }
}

export interface ToolCall {
  arguments?: Record<string, JsonValue>
  display?: null | string
  name: string
}

export interface ToolExecutionOptions {
  /** Picks the reassurance wording. Deterministic in tests. */
  choose?: (options: readonly string[]) => string
  /** Schedules the reassurance rows. Without one there are none — which is
   *  what every test that does not care about them gets. */
  clock?: CountdownClock
  /** How the expand key is spelled in the collapsed hint. WP3 owns the
   *  keybinding registry; this is only the wording. */
  expandHint?: string
  /** Injected so a test can assert on a duration. */
  now?: () => number
  paddingX?: number
  /** One-line collapsed rows with their output hidden, which is the default.
   *  False is `/quiet-tools off`: the first {@link PREVIEW_LINES} of the
   *  result, as this panel drew them before pi-fold's rule arrived. */
  quiet?: boolean
}

/** A short, single-line rendering of a tool call's arguments, for tools that
 *  name no display label of their own. */
export function previewArgs(args: Record<string, JsonValue>): string {
  const text = Object.entries(args)
    .map(([key, value]) => `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`)
    .join(' ')
    .replace(/\s+/g, ' ')

  return text.length > ARG_PREVIEW_CHARS ? text.slice(0, ARG_PREVIEW_CHARS - 1) + '…' : text
}

/** One argument as a line of text. */
function argText(value: JsonValue | undefined): string {
  if (value === undefined || value === null) {
    return ''
  }

  return inlineText(typeof value === 'string' ? value : (JSON.stringify(value) ?? ''))
}

/** Where a file tool was pointed. `file_path` is not ours — an MCP tool's, or
 *  a plugin's — and costs one `??` to honour. */
function pathArg(args: Record<string, JsonValue>): string {
  return argText(args.path) || argText(args.file_path)
}

/** `path:offset`, which is how this transcript already asks to be read. */
function readSummary(args: Record<string, JsonValue>): string {
  const path = pathArg(args)
  const offset = typeof args.offset === 'number' ? args.offset : undefined

  return path && offset !== undefined ? `${path}:${offset}` : path
}

/** The file and how many replacements were asked of it. */
function editSummary(args: Record<string, JsonValue>): string {
  const path = pathArg(args)
  const edits = Array.isArray(args.edits) ? args.edits.length : 0

  return edits > 0 ? `${path} (${edits} ${edits === 1 ? 'edit' : 'edits'})` : path
}

/** `pattern in path`, for the two tools that take both. */
function searchSummary(args: Record<string, JsonValue>): string {
  const pattern = argText(args.pattern)
  const where = pathArg(args) || '.'

  return pattern ? `${pattern} in ${where}` : where
}

/**
 * The first argument, for a tool with no format of its own: an MCP server's, a
 * plugin's, anything the gateway grew since this was written.
 *
 * The first argument that is *text*, because that is the one that says what the
 * call was about: a limit or a flag arriving ahead of it in the object is not
 * what the row is for. Shown on its own, since the header already names the
 * tool. With no text argument at all the first one keeps its key, because
 * `[{…}]` alone says nothing.
 */
function firstArg(args: Record<string, JsonValue>): string {
  const entries = Object.entries(args)
  const text = entries.find(([, value]) => typeof value === 'string' && value.trim())

  if (text) {
    return argText(text[1])
  }

  const entry = entries[0]

  if (!entry) {
    return ''
  }

  const rendered = argText(entry[1])

  return rendered ? `${entry[0]}=${rendered}` : entry[0]
}

/**
 * What a collapsed row says was invoked.
 *
 * One line per tool, the shortest thing that identifies the call: the command
 * for `bash`, the file for the file tools, the pattern and where for the search
 * tools, the URL, the query, the skill, the task. A tool nobody named here
 * shows its first argument.
 */
export function invocationSummary(name: string, args: Record<string, JsonValue> = {}): string {
  // A call with no arguments at all is every row a resumed session replays: the
  // wire carries the tool's name and nothing else, and a default filled in here
  // (`.` for a listing, say) would be an invocation that never happened.
  if (Object.keys(args).length === 0) {
    return ''
  }

  switch (name) {
    case 'bash':
      return argText(args.command)

    case 'read':
      return readSummary(args)

    case 'edit':
      return editSummary(args)

    case 'write':
      return pathArg(args)

    case 'ls':
      return pathArg(args) || '.'

    case 'find':
    case 'grep':
      return searchSummary(args)

    case 'web_fetch':
      return argText(args.url)

    case 'web_search':
      return argText(args.query)

    case 'use_skill':
      return argText(args.skill_id)

    default:
      // Every protein-design tool but `start` is about one task, and the task
      // is what the row is for.
      return name.startsWith('protein_design_') ? argText(args.task_id) || firstArg(args) : firstArg(args)
  }
}

/** `1.2s`, `340ms`. */
function formatDuration(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`
}

export class ToolExecution extends Container {
  private readonly args: Record<string, JsonValue>
  private readonly body = new Text('', 0, 0)
  private readonly box: Box
  private readonly choose: (options: readonly string[]) => string
  private readonly clock: CountdownClock | undefined
  private readonly display: null | string
  /** The one error line a quiet collapsed failure keeps. Truncated, where the
   *  body wraps, so a failure is one row and not a paragraph. */
  private readonly errorLine = new Line()
  private readonly expandHint: string
  private readonly header = new Line()
  private readonly name: string
  private readonly now: () => number
  private readonly startedAt: number
  private readonly theme: Theme
  private readonly tui: TUI

  private durationMs: null | number = null
  private expanded = false
  /** The palette the header and the body were painted in. Their strings are
   *  painted once and held by the child lines, so a palette that moves under a
   *  panel already on screen has to repaint them. */
  private paintedScheme: ColorScheme
  private progress = ''
  private quiet: boolean
  /** The one reassurance row, rewritten in place at most once. */
  private reassurance = ''
  private reassuranceStep = 0
  private result = ''
  private state: ToolState = 'pending'
  private timer: CountdownTimer | undefined
  private truncated = false

  constructor(theme: Theme, tui: TUI, call: ToolCall, opts: ToolExecutionOptions = {}) {
    super()

    this.theme = theme
    this.paintedScheme = theme.scheme
    this.tui = tui
    this.clock = opts.clock
    this.choose = opts.choose ?? (options => options[Math.floor(Math.random() * options.length)] ?? options[0]!)
    this.now = opts.now ?? (opts.clock ? () => opts.clock!.nowMs() : Date.now)
    this.startedAt = this.now()
    this.expandHint = opts.expandHint ?? 'ctrl+o'
    this.quiet = opts.quiet ?? true
    this.name = call.name
    this.args = call.arguments ?? {}
    this.display = call.display ?? null

    this.box = new Box(opts.paddingX ?? 1, 0)
    this.box.addChild(this.header)

    this.addChild(new Spacer(1))
    this.addChild(
      new MouseRegion(this.box, event => {
        if (!this.result || event.type !== 'click' || event.button !== 'left') {
          return undefined
        }

        this.setExpanded(!this.expanded)

        return { handled: true }
      })
    )

    this.redraw()
    this.scheduleReassurance()
  }

  get isExpanded(): boolean {
    return this.expanded
  }

  /** Still waiting for `tool.complete`. */
  get isPending(): boolean {
    return this.state === 'pending'
  }

  /** Whether a collapsed row is one line with its output hidden. */
  get isQuiet(): boolean {
    return this.quiet
  }

  /** Genuine progress. It never resets the call's age: a tool that reports
   *  every second is still a tool that has been running for a minute. */
  setProgress(preview: string): void {
    this.progress = preview.replace(/\s+/g, ' ').trim()
    this.redraw()
  }

  /** `durationMs` is how long the call took; `null` says it is not known,
   *  which is the case for a call replayed from a stored session, and leaves
   *  the timing off the header rather than reporting 0ms. */
  complete(result: { durationMs?: null | number; result_preview: string; truncated: boolean }): void {
    this.durationMs = result.durationMs === undefined ? this.now() - this.startedAt : result.durationMs
    this.dispose()
    this.progress = ''
    this.reassurance = ''
    this.result = result.result_preview
    this.truncated = result.truncated
    this.state = 'success'
    this.redraw()
  }

  /** The turn ended with no result for this call — cancelled, or the model
   *  call failed underneath it. */
  abandon(reason: string): void {
    if (this.state !== 'pending') {
      return
    }

    this.durationMs = this.now() - this.startedAt
    this.dispose()
    this.progress = ''
    this.reassurance = ''
    this.result = reason
    this.truncated = false
    this.state = 'error'
    this.redraw()
  }

  setExpanded(expanded: boolean): void {
    if (expanded === this.expanded) {
      return
    }

    this.expanded = expanded
    this.redraw()
    this.tui.requestRender()
  }

  /** `/quiet-tools`. Changes this row's drawing and nothing else: the result
   *  the model was given was settled before this panel existed. */
  setQuiet(quiet: boolean): void {
    if (quiet === this.quiet) {
      return
    }

    this.quiet = quiet
    this.redraw()
    this.tui.requestRender()
  }

  /** Release the reassurance timer. Idempotent, and called from every path
   *  that loses the component: completion, abandonment, a discarding retry and
   *  the end of the turn. */
  dispose(): void {
    this.timer?.dispose()
    this.timer = undefined
  }

  /**
   * Arm the next reassurance deadline, or show the row that is already due.
   *
   * Both deadlines are absolute, so a process that was suspended past them
   * comes back to the state it should be in rather than replaying a burst of
   * lines. The arm below only ever targets a deadline strictly in the future,
   * which is what keeps the timer from expiring inside its own constructor.
   */
  private scheduleReassurance(): void {
    this.timer?.dispose()
    this.timer = undefined

    if (!this.clock || this.state !== 'pending') {
      return
    }

    const elapsed = this.now() - this.startedAt

    if (elapsed >= REASSURE_AGAIN_MS) {
      this.showReassurance(2, elapsed)

      return
    }

    if (elapsed >= REASSURE_AT_MS) {
      this.showReassurance(1, elapsed)
    }

    const deadline = this.startedAt + (elapsed >= REASSURE_AT_MS ? REASSURE_AGAIN_MS : REASSURE_AT_MS)

    this.timer = new CountdownTimer(
      deadline,
      this.clock,
      // Scheduling only: the row asks for its own render when it changes.
      undefined,
      () => {},
      () => {
        this.timer = undefined
        this.scheduleReassurance()
      }
    )
  }

  private showReassurance(step: 1 | 2, elapsedMs: number): void {
    if (this.reassuranceStep >= step) {
      return
    }

    const phrases = waitPhrases(this.name, this.progress)
    const rest = step === 2 ? phrases.filter(phrase => !this.reassurance.startsWith(phrase)) : phrases

    this.reassuranceStep = step
    this.reassurance = waitLine(this.choose(rest.length > 0 ? rest : phrases), elapsedMs)
    this.redraw()
    this.tui.requestRender()
  }

  /** The header and the body hold painted strings, and the palette can change
   *  under a panel that is already on screen (the OSC 11 reply swaps it after
   *  boot). Repaint before the children drop their caches. */
  override invalidate(): void {
    this.redraw()
    super.invalidate()
  }

  /** Self-sufficient like every other component that holds painted lines: a
   *  palette that moved is noticed here rather than waited for. */
  override render(width: number): string[] {
    if (this.paintedScheme !== this.theme.scheme) {
      this.redraw()
      super.invalidate()
    }

    return super.render(width)
  }

  /** What the tool was asked to do, in one line. A tool that named its own
   *  label gets that; quiet rows get the per-tool summary, and the rest get
   *  the arguments as they came. */
  private get label(): string {
    const text = this.display ?? (this.quiet ? invocationSummary(this.name, this.args) : previewArgs(this.args))

    return text.replace(/\s+/g, ' ').trim()
  }

  /** The row is collapsed and hiding what the tool returned. */
  private get folded(): boolean {
    return this.quiet && !this.expanded && Boolean(this.result)
  }

  private redraw(): void {
    const t = this.theme

    this.paintedScheme = t.scheme

    this.box.setBgFn(text => t.bg(STATE_BG[this.state], text))

    const duration = this.durationMs === null ? '' : t.fg('muted', ` ${formatDuration(this.durationMs)}`)
    const label = this.label ? ' ' + t.fg('muted', this.label) : ''
    // The hint rides the invocation line while the output is hidden, so the
    // whole collapsed call is one row. `truncated` goes with it: it is the one
    // thing about a hidden result that the row would otherwise not say.
    const notes = this.folded ? [...(this.truncated ? ['truncated'] : []), `${this.expandHint} to expand`] : []

    this.header.setText(
      t.bold(t.fg('text', this.name)) + label + duration,
      notes.length > 0 ? t.fg('muted', `(${notes.join('; ')})`) : ''
    )

    this.mount(this.errorLine, this.drawErrorLine())
    this.mount(this.body, this.drawBody())
  }

  /** A quiet collapsed failure keeps its first line. Returns whether there is
   *  one to keep. */
  private drawErrorLine(): boolean {
    if (!this.folded || this.state !== 'error') {
      return false
    }

    const first = this.result.split('\n').find(line => line.trim()) ?? ''

    this.errorLine.setText(this.theme.fg('error', inlineText(first) || '[no output]'))

    return true
  }

  /** Progress, reassurance and — unless this row is folded — the result.
   *  Returns whether any of it is there. */
  private drawBody(): boolean {
    const t = this.theme
    const lines: string[] = []

    if (this.progress) {
      lines.push(t.fg('muted', this.progress))
    }

    // After the progress, never instead of it: what the tool actually reported
    // outranks the fact that we have been waiting for it.
    if (this.reassurance) {
      lines.push(t.fg('muted', this.reassurance))
    }

    if (this.result && !this.folded) {
      const all = this.result.split('\n')
      const shown = this.expanded ? all : all.slice(0, PREVIEW_LINES)
      const hidden = all.length - shown.length
      const tone = this.state === 'error' ? 'error' : 'muted'

      lines.push(...shown.map(line => t.fg(tone, line)))

      const notes: string[] = []

      if (hidden > 0) {
        notes.push(`${hidden} more ${hidden === 1 ? 'line' : 'lines'}, ${this.expandHint} to expand`)
      }

      if (this.truncated) {
        notes.push('truncated')
      }

      if (notes.length > 0) {
        lines.push(t.fg('muted', `… (${notes.join('; ')})`))
      }
    }

    this.body.setText(lines.join('\n'))

    return lines.length > 0
  }

  /** Keep `component` in the box exactly when `wanted`. */
  private mount(component: Component, wanted: boolean): void {
    const present = this.box.children.includes(component)

    if (wanted && !present) {
      this.box.addChild(component)
    } else if (!wanted && present) {
      this.box.removeChild(component)
    }
  }
}
