// The turn model: it owns the subscription for one session, sends and cancels
// turns, and translates the gateway's event stream into components inside the
// chat container it was given.
//
// Behaviour ported from `ui-tui/src/app/chatStream.ts` and `turnController.ts`:
// the 10 s server-ack watchdog, the "turn already in progress" guard, the
// discarding retry, and usage accumulation. The rendering is pi's, not the old
// app's: tools render inline where they happened, one streaming assistant
// component per model call, tool components looked up by `tool_call_id`.

import type { Component, Container, TUI } from '@earendil-works/pi-tui'

import type { CountdownClock } from './components/countdownTimer.js'
import type { Gateway } from './gateway.js'
import type { ProteinDesignProgressEvent, TurnEvent, UsageSnapshot } from './rpc/index.js'
import type { Theme } from './theme.js'

import { AssistantMessage } from './components/assistantMessage.js'
import { systemBlock, systemLine } from './components/systemLine.js'
import { ToolExecution } from './components/toolExecution.js'
import { UserMessage } from './components/userMessage.js'
import { TurnInProgressError } from './rpc/index.js'

/** Server-ack window. Measures `[turn.send → first event]` and nothing else:
 *  the gateway emits `message.start` before any model work, so this never
 *  measures first-token latency, which is routinely longer than this. */
export const DEFAULT_WATCHDOG_MS = 10_000

/**
 * What the footer needs.
 *
 * Conventions, because two of them are easy to mix up:
 *
 * - `input`, `output`, `cost` and `listCost` are the *session's*, as the server
 *   totals them: `message.complete` carries them in its `session_*` fields,
 *   from the same tracker `/status` reads, so the two surfaces cannot report
 *   different money for one session. They are taken, not added to — the server
 *   has already summed every call of every turn, this process's and the ones
 *   stored before it.
 * - `cacheHitPercent` is the share of the *last* call's prompt that came from
 *   cache, as the server reported it. It is kept as the percentage it is
 *   rather than turned into an absolute count: the wire states no absolute
 *   figure, and reconstructing one from a rounded percentage and a different
 *   call's prompt produced a number that was exact-looking and wrong.
 * - `contextUsed` / `contextMax` likewise describe the last call.
 */
export interface UsageTotals {
  /** Share of the last call's prompt served from cache, 0-100, or null when
   *  the provider stated none. */
  cacheHitPercent: null | number
  /** 0 when no table sizes the model; do not render a percentage of it. */
  contextMax: number
  /** What the window holds, in tokens. Null when nothing measures it: the
   *  conversation was compacted after the last call, so the only figure the
   *  wire ever stated describes a prompt that is gone. Show `?`, not the
   *  number, until a call that ran after the boundary reports one. */
  contextUsed: null | number
  /** USD the vendor says this session has cost, or null on a plan-billed
   *  provider that states no per-token figure. `/status` reports this. */
  cost: null | number
  /** USD the session's tokens are worth at the vendor's published price,
   *  whoever is billing. What the footer shows, as pi shows it: a plan reports
   *  no cost at all, and `$0.000` beside `(sub)` reads as free rather than as
   *  covered. Null when no price is published for the models used. */
  listCost: null | number
  /** The session's total prompt tokens, cached ones included. */
  input: number
  output: number
  /** Part of `output` is this UI's own estimate of what is streaming right
   *  now, not a vendor count. Cleared by the next authoritative figure. */
  outputEstimated: boolean
}

/**
 * The controller refused the send before anything left this process.
 *
 * Nothing reached the gateway, so the prompt is exactly as unsent as it was a
 * moment ago and belongs back at the head of the queue. Distinct from
 * `TurnInProgressError`, which is the gateway refusing a send it did receive.
 */
export class TurnNotStartedError extends Error {}

export type TurnStatus = 'error' | 'idle' | 'running'

export interface TurnControllerOptions {
  /** The transcript. The controller only ever appends to it, and removes what
   *  a discarding retry made void. */
  chat: Container
  /** Passed to each tool panel for its long-run reassurance row. */
  clock?: CountdownClock
  gateway: Gateway
  /** Injected so tests can assert on tool durations. */
  now?: () => number
  theme: Theme
  tui: TUI
  watchdogMs?: number
}

/** Characters per token, for the estimate shown between usage events. The
 *  crude ratio every vendor's own guidance quotes; it is labelled as an
 *  estimate wherever it is shown, and replaced the moment the turn reports. */
const CHARS_PER_TOKEN = 4

/** How many turn ids to remember after they end. A late frame arrives within
 *  one turn of the one it belongs to; this is generous and bounded. */
const REMEMBERED_TURNS = 32

/** Remember `id`, dropping the oldest once the set is full. Sets iterate in
 *  insertion order, so the first key is the oldest. */
function remember(ids: Set<string>, id: string): void {
  ids.delete(id)
  ids.add(id)

  while (ids.size > REMEMBERED_TURNS) {
    const oldest = ids.values().next().value as string

    ids.delete(oldest)
  }
}

function emptyUsage(): UsageTotals {
  return {
    cacheHitPercent: null,
    listCost: null,
    contextMax: 0,
    contextUsed: 0,
    cost: null,
    input: 0,
    output: 0,
    outputEstimated: false
  }
}

/** Local `HH:MM`, or `MM-DD HH:MM` when it was not today. A missed reminder's
 *  whole point is how long ago it was, and a bare `09:00` after a weekend away
 *  reads like this morning. */
function formatScheduledAt(iso: string): string {
  const at = new Date(iso)

  if (Number.isNaN(at.getTime())) {
    return iso
  }

  const now = new Date()
  const clock = `${String(at.getHours()).padStart(2, '0')}:${String(at.getMinutes()).padStart(2, '0')}`
  const sameDay =
    at.getFullYear() === now.getFullYear() && at.getMonth() === now.getMonth() && at.getDate() === now.getDate()

  if (sameDay) {
    return clock
  }

  return `${String(at.getMonth() + 1).padStart(2, '0')}-${String(at.getDate()).padStart(2, '0')} ${clock}`
}

export class TurnController {
  /** A turn ended. The app drains its queued input here. */
  onIdle?: () => void
  /** A protein-design task reported progress. Task-global background news, not
   *  anything about the turn in flight — the monitor owns it. */
  onProteinProgress?: (payload: ProteinDesignProgressEvent['payload']) => void
  /** The turn for this prompt stored nothing on the gateway, so there is no
   *  exchange for it to undo. */
  onUnsaved?: (content: string) => void
  onStatus?: (status: TurnStatus) => void
  onUsage?: (usage: UsageTotals) => void

  private readonly chat: Container
  private readonly clock: CountdownClock | undefined
  private readonly gateway: Gateway
  private readonly now: () => number
  private readonly theme: Theme
  private readonly tools = new Map<string, ToolExecution>()
  private readonly tui: TUI
  private readonly watchdogMs: number

  /** Components appended since the current model call began, so a discarding
   *  retry drops that call's work and keeps everything before it. */
  private callComponents: Component[] = []
  private assistant: AssistantMessage | null = null
  /** The echo for the turn in flight, so a turn that produced nothing can say
   *  so on the component the transcript shows. */
  private echo: null | UserMessage = null
  /** Whether this turn produced any assistant work at all — text, reasoning or
   *  a tool call. See {@link ownsTerminalEvent} for the other half of the
   *  identity problem. */
  private produced = false
  /** Which session the figures in `settled` were counted for. */
  private accountedSession: null | string = null
  /** Characters of text and reasoning streamed since the last authoritative
   *  figure, for the estimate that fills the gap between usage events. */
  private liveChars = 0
  /** Characters this session has added to the window since the last call
   *  reported its size: prompts sent and replies streamed, including those of
   *  a turn that failed. pi adds the same estimate to the last figure it
   *  trusts, so the context bar moves while a conversation does. */
  private trailingChars = 0
  /** What the stored history is estimated to hold, from the server, for a
   *  session resumed with a transcript no call of this process has reported
   *  on. pi estimates the same thing from the messages it just loaded. */
  private contextBaseline = 0
  /** Characters streamed by the model call in progress. A retry that discards
   *  the call takes them back out of the context estimate: text nobody kept is
   *  not in the conversation the next call will send. */
  private callChars = 0
  /** What this turn has added to `trailingChars`: its prompt and everything
   *  streamed since. A turn the gateway stored nothing for takes all of it
   *  back, not just the prompt — a cancelled reply is no more in the next
   *  call's conversation than the question that asked for it. */
  private turnChars = 0
  /** A compaction has happened since the last figure was reported. */
  private contextUnknown = false
  /** Which model's capacity the footer is showing. Bumped by every selection
   *  the gateway confirms; a turn records the value it started under, and a
   *  completion from an older one is an answer about a model this session has
   *  left. Its occupancy still counts -- the conversation did not change --
   *  but its window size describes somebody else's model. */
  private windowGeneration = 0
  private turnWindowGeneration = 0
  private liveOutput = 0
  /** Turns this controller has finished with. A `message.start` naming one of
   *  them is a late frame for a turn that is over, not a turn to resume. */
  private readonly retired = new Set<string>()
  /** Turns whose usage has already been counted. The same completion arriving
   *  twice must not be added twice. */
  private readonly settledTurns = new Set<string>()
  private sendInFlight = false
  /** Bumped by every attach and detach. An event from a subscription this
   *  controller has already left, or a `turn.send` that resolves after the
   *  session changed underneath it, carries a stale generation and is
   *  dropped rather than applied to whatever is on screen now. */
  private sessionGeneration = 0
  private sessionKey: null | string = null
  private settled = emptyUsage()
  private thinkingCollapsed = true
  private toolsExpanded = false
  /** Collapsed tool rows are one line with their output hidden, which is where
   *  `/quiet-tools` starts. */
  private toolsQuiet = true
  private turnId: null | string = null
  /** Bumped whenever a turn is retired (completed, failed, force-reset,
   *  detached). A `turn.send` continuation whose generation has moved on
   *  belongs to a turn that is already over and must install nothing. */
  private turnGeneration = 0
  private unsubscribe: (() => Promise<void>) | null = null
  private watchdog: null | ReturnType<typeof setTimeout> = null

  constructor(opts: TurnControllerOptions) {
    this.chat = opts.chat
    this.clock = opts.clock
    this.gateway = opts.gateway
    this.theme = opts.theme
    this.tui = opts.tui
    this.now = opts.now ?? Date.now
    this.watchdogMs = opts.watchdogMs ?? DEFAULT_WATCHDOG_MS
  }

  /** A turn is running, or we are still waiting for the server to accept one. */
  get active(): boolean {
    return this.sendInFlight || this.turnId !== null
  }

  /** What the window holds: the last call's own figure plus an estimate of
   *  everything said since. A turn that fails still put its prompt and
   *  whatever it streamed in front of the model. */
  private get contextUsed(): null | number {
    if (this.contextUnknown) {
      return null
    }

    // A call's own figure supersedes the opening estimate the moment one
    // lands; until then the resumed transcript is all the window holds.
    // With neither -- a session's first turn, and it failed -- what was said
    // is still all there is to count, which is what pi counts when no
    // assistant usage exists yet.
    const base = this.settled.contextUsed || this.contextBaseline

    return base + Math.ceil(this.trailingChars / CHARS_PER_TOKEN)
  }

  /**
   * Say how big the window is, when something other than a call says so.
   *
   * The measured figure belongs to the model that answered. After a switch the
   * session runs on another one, whose window may be a different size, and the
   * next completion is the only thing that would have replaced it: until then
   * the footer divided this session's occupancy by the previous model's
   * capacity. The gateway's answer about the model now selected supersedes it.
   */
  setContextWindow(tokens: number): void {
    // Bumped even when the size is unchanged: two models can share a window
    // size, and what this marks is that the selection moved, not that the
    // number did. Without it a switch between two 200k models left an
    // in-flight turn's completion still counted as current.
    this.windowGeneration += 1

    if (tokens === this.settled.contextMax) {
      return
    }

    this.settled.contextMax = tokens
    this.onUsage?.(this.usage)
  }

  /**
   * Say how many tokens the stored conversation holds, as the server sizes it.
   *
   * Two moments need this. A resumed session's window is full before this
   * process sends anything, and nothing on the wire says so until the first
   * call reports. And `/undo` takes an exchange back out, after which the last
   * call's figure describes a conversation that is longer than the one the
   * session now has — so this supersedes it, and the estimate counts up from
   * here again until the next call states its own prompt.
   */
  setContextBaseline(tokens: null | number): void {
    const before = this.contextUsed

    // Null is the server saying it cannot size this conversation: a
    // compaction marker its backend replays stands in front of the stored
    // history, so the messages before it are not what the next call sends and
    // nothing here knows what the marker is worth. Unknown until a call
    // reports, exactly as it is after a compaction this process watched.
    this.contextUnknown = tokens === null
    this.contextBaseline = tokens ?? 0
    this.settled.contextUsed = 0
    this.trailingChars = 0
    this.callChars = 0
    this.turnChars = 0

    if (this.contextUsed !== before) {
      this.onUsage?.(this.usage)
    }
  }

  get usage(): UsageTotals {
    // Between `turn.usage` events the only thing known about the reply is how
    // much of it has been rendered, so say so rather than hold the counter
    // still while a long answer streams. Flagged, so the footer can mark it.
    const estimate = Math.ceil(this.liveChars / CHARS_PER_TOKEN)

    return {
      ...this.settled,
      contextUsed: this.contextUsed,
      output: this.settled.output + this.liveOutput + estimate,
      outputEstimated: estimate > 0
    }
  }

  /** Subscribe to a session's turn events. Safe to call once per session. */
  async attach(sessionKey: string): Promise<void> {
    await this.detach()

    const generation = (this.sessionGeneration += 1)

    this.sessionKey = sessionKey
    this.resetAccounting(sessionKey)

    const subscription = await this.gateway.turnSubscribe(sessionKey, event => {
      // Events keep arriving until the server acknowledges the unsubscribe.
      // They describe a session this controller has left; the transcript on
      // screen is someone else's.
      if (generation === this.sessionGeneration) {
        this.handleEvent(event)
      }
    })

    if (generation !== this.sessionGeneration) {
      // Detached (or re-attached elsewhere) while the subscribe was in flight.
      await subscription.unsubscribe().catch(() => {
        // Already gone is gone enough.
      })

      return
    }

    this.unsubscribe = subscription.unsubscribe
  }

  async detach(): Promise<void> {
    this.sessionGeneration += 1
    this.turnGeneration += 1
    this.retired.clear()
    this.settledTurns.clear()
    this.clearWatchdog()
    this.clearLiveUsage()
    this.sendInFlight = false
    this.turnId = null

    // The panels stay on screen until the transcript is replaced, but their
    // timers belong to a turn on a session this controller is leaving.
    for (const tool of this.tools.values()) {
      tool.dispose()
    }

    this.tools.clear()

    const unsubscribe = this.unsubscribe

    this.unsubscribe = null
    this.sessionKey = null

    if (unsubscribe) {
      await unsubscribe()
    }
  }

  /**
   * Echo the message and send it. Rejects when a turn is already running — the
   * caller is expected to check {@link active} and queue instead.
   */
  async send(content: string): Promise<void> {
    if (!this.sessionKey) {
      throw new TurnNotStartedError('no session — nothing was sent')
    }

    if (this.active) {
      throw new TurnNotStartedError('turn already in progress — wait for it to finish or cancel it')
    }

    const echo = new UserMessage(this.theme, content)

    this.echo = echo
    this.produced = false
    // Which model this turn is bound to, for as long as it runs. A selection
    // made while it is in flight moves the session on; this turn's completion
    // still arrives, describing the model it was sent to.
    this.turnWindowGeneration = this.windowGeneration
    // In the window from here, whatever becomes of the turn.
    this.trailingChars += content.length
    this.turnChars = content.length
    // Said now rather than at the first token: the prompt is in front of the
    // model from the moment it is sent, and a footer that waits for the reply
    // shows the window as it was before the question.
    this.onUsage?.(this.usage)
    this.append(echo, false)
    this.setStatus('running')

    // Armed before the await, so it also covers a `turn.send` that never
    // returns, and so the accept/`message.start` race always finds it armed.
    this.sendInFlight = true
    this.armWatchdog()
    this.tui.requestRender()

    // The gateway can answer the send after the turn it opened has already
    // started *and* finished — a valid server writes the response and both
    // terminal events into one chunk, and the read loop drains it before this
    // continuation resumes. Retiring the turn moves the generation on, and a
    // continuation that finds it moved has nothing left to install.
    const generation = this.turnGeneration
    let accepted: boolean

    try {
      const result = await this.gateway.turnSend(this.sessionKey, content)

      if (generation !== this.turnGeneration) {
        return
      }

      accepted = result.accepted

      // The server's turn_id from `message.start` is authoritative, but hold
      // this one so `active` stays true between the accept and that event.
      if (accepted) {
        this.turnId = result.turn_id
      }
    } catch (err) {
      if (generation !== this.turnGeneration) {
        // The failure belongs to a turn or a session that is already gone.
        // The caller still learns the send failed; the UI is not rewound.
        throw err
      }

      this.clearWatchdog()

      if (err instanceof TurnInProgressError) {
        // The gateway is still unwinding the previous turn (it reports a
        // cancel before the cancelled task has finished). Nothing happened
        // here: take the echo back and leave the state idle without telling
        // the app a turn ended, so the caller can re-queue and try again.
        this.chat.removeChild(echo)
        this.untake()
        this.sendInFlight = false
        this.setStatus('idle')
        this.tui.requestRender()

        throw err
      }

      // The send itself failed, so the gateway stored nothing for it.
      this.forget(echo)
      this.append(systemBlock(this.theme, `error: ${err instanceof Error ? err.message : String(err)}`, 'error'), false)
      this.endTurn('error')

      throw err
    }

    this.sendInFlight = false

    if (!accepted) {
      this.forget(echo)
      this.append(systemLine(this.theme, 'the server did not accept the turn', 'warn'), false)
      this.endTurn('idle')
    }
  }

  /** Ask the server to stop. The UI resets on the `error` event that follows
   *  with reason `cancelled_by_client`, not here. */
  async cancel(): Promise<void> {
    if (!this.sessionKey || !this.active) {
      return
    }

    this.append(systemLine(this.theme, 'cancelling…'), false)
    this.tui.requestRender()

    try {
      await this.gateway.turnCancel(this.sessionKey)
    } catch {
      // The server could not be reached or did not know the turn: recover the
      // prompt locally rather than leave the UI wedged.
      this.forceReset('cancel failed — input restored')
    }
  }

  /** Drop the turn locally, without waiting for the server. Backs the watchdog
   *  and the Ctrl+C escape hatch. */
  forceReset(note?: string): void {
    if (note) {
      this.append(systemLine(this.theme, note, 'warn'), false)
    }

    this.endTurn('idle')
  }

  handleEvent(event: TurnEvent): void {
    // Before anything about the turn, watchdog included: this event is a
    // broadcast about a detached worker that subscribed sessions all receive.
    // Treating it as the server acknowledging *our* `turn.send` would disarm
    // the watchdog for a turn nothing has answered yet.
    if (event.type === 'protein_design.progress') {
      this.onProteinProgress?.(event.payload)
      this.tui.requestRender()

      return
    }

    if (this.isStaleFrame(event)) {
      return
    }

    // Any event proves the subscription is live.
    this.clearWatchdog()

    switch (event.type) {
      case 'message.start':
        this.turnId = event.payload.turn_id
        this.liveChars = 0
        this.liveOutput = 0
        this.beginCall()
        this.setStatus('running')
        break

      case 'episode.start':
        this.beginCall()
        break

      case 'token.delta':
        this.produced = true
        this.liveChars += event.payload.text.length
        this.trailingChars += event.payload.text.length
        this.callChars += event.payload.text.length
        this.turnChars += event.payload.text.length
        this.openAssistant().appendText(event.payload.text)
        this.onUsage?.(this.usage)
        break

      case 'thinking.delta':
        this.produced = true
        this.liveChars += event.payload.text.length
        this.trailingChars += event.payload.text.length
        this.callChars += event.payload.text.length
        this.turnChars += event.payload.text.length
        this.openAssistant().appendThinking(event.payload.text)
        this.onUsage?.(this.usage)
        break

      case 'tool.start': {
        this.produced = true

        const { arguments: args, display, name, tool_call_id } = event.payload

        this.assistant?.setHasToolCalls(true)
        this.closeAssistant()

        const tool = new ToolExecution(
          this.theme,
          this.tui,
          { arguments: args, display, name },
          { now: this.now, quiet: this.toolsQuiet, ...(this.clock ? { clock: this.clock } : {}) }
        )

        tool.setExpanded(this.toolsExpanded)
        this.tools.set(tool_call_id, tool)
        this.append(tool, true)
        break
      }

      case 'tool.progress':
        this.tools.get(event.payload.tool_call_id)?.setProgress(event.payload.preview)
        break

      case 'tool.complete': {
        const { result_preview, tool_call_id, truncated } = event.payload

        this.tools.get(tool_call_id)?.complete({ result_preview, truncated })
        break
      }

      case 'turn.notice':
        // Not the reply and not a failure: something about the turn the user
        // would otherwise never learn, said once, above the output.
        this.append(systemLine(this.theme, event.payload.text, 'warn'), false)
        break

      case 'turn.retry':
        this.onRetry(event.payload)
        break

      case 'turn.usage':
        // `completion_tokens` is this turn's output across every model call so
        // far, and `reasoning_tokens` is the share of it the vendor attributed
        // to reasoning (`agent/loop/main.py:_reasoning_tokens_of`) — a subset,
        // not a second figure to add. It supersedes the estimate.
        this.liveChars = 0
        this.liveOutput = event.payload.completion_tokens
        this.onUsage?.(this.usage)
        break

      case 'message.complete':
        if (event.payload.turn_id !== null) {
          remember(this.settledTurns, event.payload.turn_id)
        }

        this.recordUsage(event.payload.usage)
        this.endTurn('idle')
        break

      case 'error':
        this.onError(event.payload)
        break

      case 'cron.delivered': {
        const { fired_at, name, text } = event.payload

        this.append(systemBlock(this.theme, `⏰ ${fired_at ? `${name} @ ${fired_at}` : name}\n${text}`), false)
        break
      }

      case 'cron.missed': {
        const { count, items } = event.payload
        const lines = items.map(
          item => `  ${item.name} — scheduled ${formatScheduledAt(item.scheduled_at)}: ${item.message}`
        )

        this.append(
          systemBlock(this.theme, `⏰ missed ${count} ${count === 1 ? 'reminder' : 'reminders'}\n${lines.join('\n')}`),
          false
        )
        break
      }

      // A new variant in the schema lands here and fails the type-check,
      // which is the point.
      default: {
        const exhaustive: never = event

        void exhaustive
      }
    }

    this.tui.requestRender()
  }

  /**
   * Show or hide reasoning across the whole transcript.
   *
   * Always walks the children, even when the stored default already says
   * this: a click on one thinking label moves that component alone, so
   * `/details thinking collapsed` has to reach it whatever the default is.
   */
  setThinkingCollapsed(collapsed: boolean): void {
    this.thinkingCollapsed = collapsed

    for (const child of this.chat.children) {
      if (child instanceof AssistantMessage) {
        child.setThinkingCollapsed(collapsed)
      }
    }

    this.tui.requestRender()
  }

  /** Expand or collapse every tool panel, individually-clicked ones included. */
  setToolsExpanded(expanded: boolean): void {
    this.toolsExpanded = expanded

    for (const child of this.chat.children) {
      if (child instanceof ToolExecution) {
        child.setExpanded(expanded)
      }
    }

    this.tui.requestRender()
  }

  /**
   * `/quiet-tools`. Whether a collapsed tool row is one line with its output
   * hidden, or the first lines of every result.
   *
   * Walks the children like {@link setToolsExpanded} does, and for the same
   * reason: the setting is the transcript's, not only the next call's.
   */
  setToolsQuiet(quiet: boolean): void {
    this.toolsQuiet = quiet

    for (const child of this.chat.children) {
      if (child instanceof ToolExecution) {
        child.setQuiet(quiet)
      }
    }

    this.tui.requestRender()
  }

  /** Whether collapsed tool rows are quiet right now. A session replaying a
   *  stored transcript draws its rows the same way. */
  get quietTools(): boolean {
    return this.toolsQuiet
  }

  /** Ctrl+T. Returns whether reasoning is now visible. */
  toggleThinking(): boolean {
    this.setThinkingCollapsed(!this.thinkingCollapsed)

    return !this.thinkingCollapsed
  }

  /** Ctrl+O. Returns whether tool panels are now expanded. */
  toggleTools(): boolean {
    this.setToolsExpanded(!this.toolsExpanded)

    return this.toolsExpanded
  }

  // ── internals ──────────────────────────────────────────────────────

  private append(component: Component, partOfCall: boolean): void {
    this.chat.addChild(component)

    if (partOfCall) {
      this.callComponents.push(component)
    }
  }

  /** Start a model call: everything after this point can be discarded by a
   *  retry, and the next text delta opens a new assistant message. */
  private beginCall(): void {
    this.closeAssistant()
    this.callComponents = []
    this.callChars = 0
  }

  private closeAssistant(): void {
    this.assistant?.finish()
    this.assistant = null
  }

  private openAssistant(): AssistantMessage {
    if (!this.assistant) {
      this.assistant = new AssistantMessage(this.theme, { thinkingCollapsed: this.thinkingCollapsed })
      this.append(this.assistant, true)
    }

    return this.assistant
  }

  private onRetry(payload: Extract<TurnEvent, { type: 'turn.retry' }>['payload']): void {
    const { attempt, discard, reason, total } = payload

    if (discard) {
      this.discardCall()
    }

    this.append(systemLine(this.theme, `retrying ${attempt}/${total}${reason ? `: ${reason}` : ''}`, 'warn'), false)
  }

  /** The re-run replaces this call's work, so drop what it streamed rather
   *  than let the re-run double it. */
  private discardCall(): void {
    // Void, not merely hidden: the retried call sends the same conversation
    // this one was sent, without a word of what it streamed.
    this.trailingChars = Math.max(0, this.trailingChars - this.callChars)
    this.turnChars = Math.max(0, this.turnChars - this.callChars)
    this.callChars = 0

    for (const component of this.callComponents) {
      this.chat.removeChild(component)
    }

    for (const [id, tool] of this.tools) {
      if (this.callComponents.includes(tool)) {
        // The component is leaving the transcript, so its reassurance timer
        // has to leave with it.
        tool.dispose()
        this.tools.delete(id)
      }
    }

    this.callComponents = []
    this.assistant = null
  }

  private onError(payload: Extract<TurnEvent, { type: 'error' }>['payload']): void {
    const cancelled = payload.reason === 'cancelled_by_client'

    // Two ways a turn is known to have stored nothing, and the echo has to say
    // so or `/undo` removes the wrong exchange:
    //
    // - It was cancelled. The loop saves in `_after_turn`, which runs only
    //   after `_run_agent_loop` returns; `asyncio.CancelledError` propagates
    //   instead (`agent/loop/main.py`), so however much of the reply was
    //   streamed, none of it was written.
    // - It produced nothing at all. The gateway's no-scheduler path emits
    //   exactly `message.start` and this error, and persists no message
    //   (`tui_rpc/methods/turn.py`).
    //
    // Anything else is assumed saved: a stream that breaks after some text
    // comes back as an interrupted reply the loop returns and stores.
    if (cancelled || !this.produced) {
      this.forget(this.echo)
    }

    this.abandonPendingTools(cancelled ? 'cancelled' : 'no result — the turn failed')

    if (cancelled) {
      this.append(systemLine(this.theme, 'turn cancelled'), false)
      this.endTurn('idle')

      return
    }

    if (this.assistant) {
      this.assistant.setNote(`error: ${payload.message}`, 'error')
    } else {
      this.append(systemBlock(this.theme, `error: ${payload.message}`, 'error'), false)
    }

    this.endTurn('error')
  }

  /** Say, here and to whoever is listening, that this prompt was never saved. */
  private forget(echo: null | UserMessage): void {
    if (!echo?.saved) {
      return
    }

    echo.markUnsaved()
    this.untake()
    this.onUnsaved?.(echo.text)
  }

  /** Take a whole turn back out of the context estimate. The gateway stored
   *  nothing for it -- it was cancelled, refused, or failed before producing
   *  anything -- so neither the prompt nor whatever was streamed in answer to
   *  it is in the conversation the next call will send. */
  private untake(): void {
    this.trailingChars = Math.max(0, this.trailingChars - this.turnChars)
    this.turnChars = 0
    this.onUsage?.(this.usage)
  }

  private abandonPendingTools(reason: string): void {
    for (const tool of this.tools.values()) {
      tool.abandon(reason)
      // `abandon` releases the timer only for a call that was still pending;
      // a completed one released it already. Both are safe to ask twice.
      tool.dispose()
    }
  }

  private recordUsage(usage: UsageSnapshot): void {
    const hit = usage.cache_hit_percent ?? null

    // The session's own totals, as the server sums them — assigned, not added.
    // The token and context fields of this event describe one model call and
    // the cost fields describe one turn; adding a turn's cost per completion
    // rebuilt a session total in this process, which then drifted from the one
    // `/status` reports (a resumed session's stored turns were in that one and
    // not in this). `session_*` is that same total, so the footer and `/status`
    // are one figure. Absent only from a completion no turn runner filled in,
    // and then what was counted stands.
    this.settled.input = usage.session_input_tokens ?? this.settled.input
    this.settled.output = usage.session_output_tokens ?? this.settled.output
    this.settled.cost = usage.session_cost_usd ?? this.settled.cost
    this.settled.listCost = usage.session_list_cost_usd ?? this.settled.listCost
    this.settled.cacheHitPercent = hit
    // The call's own figure supersedes everything estimated since -- unless a
    // compaction ran after the call, which leaves the figure describing a
    // prompt the session no longer has and nothing describing the one it does.
    this.contextUnknown = usage.context_compacted === true
    this.settled.contextUsed = usage.context_used ?? this.settled.contextUsed
    this.trailingChars = 0
    this.callChars = 0
    this.turnChars = 0
    // Only from a call this session's current model answered. A turn that was
    // still running when the model changed reports the capacity of the model
    // it ran on, and taking it put the footer back on the old denominator
    // beside the new model's name.
    if (this.turnWindowGeneration === this.windowGeneration) {
      this.settled.contextMax = usage.context_max ?? this.settled.contextMax
    }

    this.liveChars = 0
    this.liveOutput = 0
    this.onUsage?.(this.usage)
  }

  /**
   * Whether this frame is about a turn that is over, and so changes nothing.
   *
   * Three rules, one per shape the problem takes:
   *
   * - A `message.start` naming a retired turn is a late frame, not a turn to
   *   resume. Taking it reactivated a turn the watchdog or Ctrl+C had already
   *   dropped, with no watchdog armed, and every later send was refused as
   *   already in progress.
   * - A completion for a turn whose usage is already counted is a duplicate.
   *   Counting it again doubled the session's totals from one report.
   * - A completion for some *other* turn while one is running belongs to the
   *   turn before it and must not end this one. With nothing running it is a
   *   late report for a turn this UI dropped: its usage still counts once.
   *
   * An unlabelled completion is attributed to the turn on screen, and dropped
   * when there is none — there is nothing to attribute it to, and the wire
   * carries nothing that would settle it.
   */
  private isStaleFrame(event: TurnEvent): boolean {
    if (event.type === 'message.start') {
      return this.retired.has(event.payload.turn_id)
    }

    if (event.type !== 'message.complete') {
      return false
    }

    const { turn_id } = event.payload

    if (turn_id === null) {
      return !this.active
    }

    return this.settledTurns.has(turn_id) || (this.active && this.turnId !== turn_id)
  }

  /** Token counts and cost describe one session. Switching to another one
   *  starts its accounting from zero; re-attaching the same session (a
   *  reconnect) keeps what has been counted. */
  private resetAccounting(sessionKey: string): void {
    if (this.accountedSession === sessionKey) {
      return
    }

    this.accountedSession = sessionKey
    this.settled = emptyUsage()
    this.contextBaseline = 0
    this.contextUnknown = false
    this.trailingChars = 0
    this.turnChars = 0
    this.liveOutput = 0
    this.onUsage?.(this.usage)
  }

  private endTurn(status: TurnStatus): void {
    // A turn the watchdog already dropped can still be followed by a late
    // `message.complete`. Its usage still counts, but the queue must not be
    // drained twice.
    const wasActive = this.active

    this.turnGeneration += 1

    // The id is finished with. A `message.start` naming it afterwards is a
    // late frame for a turn that is over, not one to pick up again.
    if (this.turnId !== null) {
      remember(this.retired, this.turnId)
    }

    this.echo = null
    this.clearWatchdog()
    this.closeAssistant()
    this.abandonPendingTools('no result — the turn ended')
    this.sendInFlight = false
    this.turnId = null
    this.callComponents = []
    this.tools.clear()
    this.clearLiveUsage()
    this.setStatus(status)
    this.tui.requestRender()

    if (wasActive) {
      this.onIdle?.()
    }
  }

  /**
   * Drop what was only ever an estimate, and say so.
   *
   * Clearing the counters is not enough: the app keeps the last snapshot it
   * was handed, so a turn cancelled mid-reply left `↓~6` in the footer of an
   * idle UI until the next turn overwrote it. Every path that ends a turn
   * comes through here.
   */
  private clearLiveUsage(): void {
    const had = this.liveChars > 0 || this.liveOutput > 0

    this.liveChars = 0
    this.liveOutput = 0

    if (had) {
      this.onUsage?.(this.usage)
    }
  }

  private setStatus(status: TurnStatus): void {
    this.onStatus?.(status)
  }

  private armWatchdog(): void {
    this.clearWatchdog()

    this.watchdog = setTimeout(() => {
      this.watchdog = null

      if (!this.active) {
        return
      }

      this.forceReset('turn produced no response — input restored (press Enter to retry)')
      this.tui.requestRender()
    }, this.watchdogMs)

    this.watchdog.unref?.()
  }

  private clearWatchdog(): void {
    if (this.watchdog !== null) {
      clearTimeout(this.watchdog)
      this.watchdog = null
    }
  }
}
