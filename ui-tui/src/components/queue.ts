// Messages typed while a turn is running, and the list of them drawn above the
// editor (pi's pendingMessages look).
//
// `display.busy_input_mode` decides what a submit during a turn means:
//   - `queue`   (the TUI default): hold it and send it when the turn ends.
//   - `interrupt`: cancel the running turn first, then send it when that turn's
//     terminal event arrives. The gateway refuses a second `turn.send` while a
//     turn is unwinding, so even `interrupt` goes through the queue rather than
//     racing the cancel.
// A full-screen TUI defaults to `queue` because you are normally writing the
// next prompt while the agent is still streaming, and an unintended interrupt
// loses that work.

import type { Component } from '@earendil-works/pi-tui'

import { Container, Spacer } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

import { ThemedLine } from './themedText.js'

export type BusyInputMode = 'interrupt' | 'queue'

/** What asking for the head to go out now actually did. `blocked` and
 *  `editing` are the important ones: the queue is untouched either way, so the
 *  prompt is still there. */
export type QueueSendResult = 'blocked' | 'editing' | 'empty' | 'sent'

const BUSY_MODES = new Set<BusyInputMode>(['interrupt', 'queue'])
const DEFAULT_BUSY_MODE: BusyInputMode = 'queue'

/** Two Enters on an empty editor count as one gesture only within this window. */
const DOUBLE_ENTER_MS = 2000

export function normalizeBusyInputMode(raw: unknown): BusyInputMode {
  if (typeof raw !== 'string') {
    return DEFAULT_BUSY_MODE
  }

  const mode = raw.trim().toLowerCase() as BusyInputMode

  return BUSY_MODES.has(mode) ? mode : DEFAULT_BUSY_MODE
}

export interface MessageQueueActions {
  /** Cancel the turn in flight. */
  interrupt(): void
  /** Start a turn with this text. */
  send(text: string): void
}

export interface MessageQueueOptions {
  actions: MessageQueueActions
  /** Key hint shown under the pending list. */
  submitHint?: string
  /**
   * The editor slot is taken — a picker, a broker prompt, a sign-in handoff or
   * a session switch — so nothing may leave the queue at all.
   *
   * This is checked before a head is taken, not after, so a message stays where
   * it is rather than being sent into whatever the user is in the middle of
   * answering. A submit while blocked queues even in `interrupt` mode: nothing
   * behind a prompt is worth cancelling a turn for.
   */
  isDispatchBlocked?: () => boolean
  /** Nothing may be sent right now: a turn is running, or one is being
   *  cancelled and the gateway has not confirmed it yet. */
  isTurnActive: () => boolean
  mode?: BusyInputMode
  now?: () => number
  theme: Theme
}

/** The pending-message list plus the rules for what a submit does. */
export class MessageQueue {
  /** The component to mount between the transcript and the editor. */
  readonly view = new Container()

  private readonly actions: MessageQueueActions
  private readonly isDispatchBlocked: () => boolean
  private readonly isTurnActive: () => boolean
  private readonly now: () => number
  private readonly submitHint: string
  private readonly theme: Theme

  private editIndex: null | number = null
  private items: string[] = []
  private lastEmptySubmitAt = 0
  private mode: BusyInputMode

  constructor(options: MessageQueueOptions) {
    this.actions = options.actions
    this.isDispatchBlocked = options.isDispatchBlocked ?? (() => false)
    this.isTurnActive = options.isTurnActive
    this.mode = options.mode ?? DEFAULT_BUSY_MODE
    this.now = options.now ?? Date.now
    this.submitHint = options.submitHint ?? 'ctrl+k'
    this.theme = options.theme
    this.render()
  }

  get length(): number {
    return this.items.length
  }

  /** What Enter does while a turn is running, as the queue is actually
   *  behaving. `busy` is not a config key the gateway serves, so this is the
   *  only place the live answer exists. */
  get busyMode(): BusyInputMode {
    return this.mode
  }

  list(): string[] {
    return [...this.items]
  }

  /** Which queued message is open in the editor, or null. */
  get editing(): null | number {
    return this.editIndex
  }

  /** Mark a queued message as the one being edited. Out-of-range clears it. */
  setEditing(index: null | number): void {
    this.editIndex = index !== null && index >= 0 && index < this.items.length ? index : null
    this.render()
  }

  /** Overwrite a queued message in place; it keeps its position in the queue. */
  replace(index: number, text: string): boolean {
    const content = text.trim()

    if (index < 0 || index >= this.items.length || !content) {
      return false
    }

    this.items[index] = content
    this.render()

    return true
  }

  /** Drop one queued message. */
  remove(index: number): boolean {
    if (index < 0 || index >= this.items.length) {
      return false
    }

    this.items.splice(index, 1)

    if (this.editIndex !== null && this.editIndex >= this.items.length) {
      this.editIndex = null
    }

    this.render()

    return true
  }

  setMode(mode: BusyInputMode): void {
    this.mode = mode
  }

  /**
   * What the editor's `onSubmit` calls. Returns what was done, so the caller can
   * report it; `interrupted` also means the text went to the queue.
   */
  submit(text: string): 'ignored' | 'interrupted' | 'queued' | 'sent' {
    const content = text.trim()

    if (!content) {
      return this.emptySubmit()
    }

    this.lastEmptySubmitAt = 0

    if (this.isDispatchBlocked()) {
      // Hold it in order. Interrupting for it would cancel a turn the user is
      // still being asked about.
      this.items.push(content)
      this.render()

      return 'queued'
    }

    if (!this.isTurnActive()) {
      if (this.items.length === 0) {
        this.actions.send(content)

        return 'sent'
      }

      // Idle, but with messages ahead of this one: they go first.
      this.items.push(content)
      this.submitHead()

      return 'queued'
    }

    this.items.push(content)
    this.render()

    if (this.mode === 'interrupt') {
      this.actions.interrupt()

      return 'interrupted'
    }

    return 'queued'
  }

  /**
   * Send the head now, whatever the busy-input mode says. Bound to Ctrl+K.
   *
   * It passes the same barrier automatic draining does. Shifting the head
   * first and finding out afterwards that nothing could be sent is how a
   * queued prompt used to disappear: `TurnController.send` refuses locally
   * while a turn is still active, and that refusal never put it back.
   */
  submitHead(): QueueSendResult {
    if (this.isDispatchBlocked() || this.isTurnActive()) {
      return 'blocked'
    }

    // The head is open in the editor. Sending it would send the text the user
    // is part-way through replacing, and would strand the draft that text
    // displaced — the edit would be over with nothing to return to. It is
    // reserved until apply, drop or cancel lets go of it, and those release
    // the queue explicitly.
    if (this.editIndex === 0) {
      return 'editing'
    }

    const head = this.items.shift()

    if (head === undefined) {
      return 'empty'
    }

    // An edit further down the queue moves up with its message.
    if (this.editIndex !== null) {
      this.editIndex -= 1
    }

    this.render()
    this.actions.send(head)

    return 'sent'
  }

  /** Put a message back at the head: the gateway refused it while it was
   *  still unwinding the previous turn, and it goes out first next time. */
  unshift(text: string): void {
    this.items.unshift(text)

    if (this.editIndex !== null) {
      this.editIndex += 1
    }

    this.render()
  }

  /** Called when a turn ends: start the next queued message, if any. */
  drain(): boolean {
    this.lastEmptySubmitAt = 0

    return this.submitHead() === 'sent'
  }

  clear(): void {
    this.items = []
    this.editIndex = null
    this.lastEmptySubmitAt = 0
    this.render()
  }

  /** Enter on an empty editor: twice in a row interrupts a running turn. */
  private emptySubmit(): 'ignored' | 'interrupted' {
    if (this.isDispatchBlocked() || !this.isTurnActive()) {
      this.lastEmptySubmitAt = 0

      return 'ignored'
    }

    const at = this.now()

    if (this.lastEmptySubmitAt && at - this.lastEmptySubmitAt <= DOUBLE_ENTER_MS) {
      this.lastEmptySubmitAt = 0
      this.actions.interrupt()

      return 'interrupted'
    }

    this.lastEmptySubmitAt = at

    return 'ignored'
  }

  private render(): void {
    this.view.clear()

    if (this.items.length === 0) {
      return
    }

    this.view.addChild(new Spacer(1))

    this.items.forEach((item, index) => {
      this.view.addChild(this.line(`${index === this.editIndex ? 'Editing:' : 'Queued:'} ${item}`))
    })

    this.view.addChild(
      this.line(
        this.editIndex === null
          ? `↳ ${this.submitHint} to send the first now`
          : '↳ enter to replace it, alt+x to drop it, esc to leave it as it was'
      )
    )
  }

  /** Painted at render time, not here: the OSC 11 background reply can swap
   *  the palette after the first prompt is already queued, and a string that
   *  was coloured when it was built cannot be recoloured. */
  private line(text: string): Component {
    return new ThemedLine(painted => this.theme.fg('muted', painted), text.replace(/\s+/g, ' '), 1)
  }
}
