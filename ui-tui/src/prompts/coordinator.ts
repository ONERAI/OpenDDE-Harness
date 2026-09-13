// The broker prompts: who is pending, when each one expires, and which one is
// on screen.
//
// The three brokers do not share a contract. Approval carries an absolute
// `expires_at` and emits `approval.closed` for every outcome. Confirm carries no
// timestamps at all, so its deadline is measured from the moment the
// notification arrived, and the backend's own 35 s ceiling stays authoritative.
// Clarify carries neither a deadline nor a close event, so it has no countdown
// at all rather than an invented one.
//
// One rule covers all three: the first of answer, expiry and close wins, and it
// wins synchronously. Everything after it is a late callback that may report,
// but may not resurrect a prompt or answer for the user.

import type { KeybindingsManager, TUI } from '@earendil-works/pi-tui'

import type { CountdownClock } from '../components/countdownTimer.js'
import type { SelectorHost, SelectorLease } from '../components/selector.js'
import type { Gateway, GatewayNotification } from '../gateway.js'
import type { Theme } from '../theme.js'
import type { ApprovalRequest, ClarifyRequest, ConfirmRequest } from './decode.js'

import { CountdownTimer } from '../components/countdownTimer.js'
import { ApprovalPrompt } from './approvalPrompt.js'
import { ClarifyPrompt } from './clarifyPrompt.js'
import { ConfirmPrompt } from './confirmPrompt.js'
import { decodeApprovalClosed, decodeApprovalRequest, decodeClarifyRequest, decodeConfirmRequest } from './decode.js'

/** The confirm broker's ceiling is 35 s; the visible window is the 30 s the
 *  approval broker uses, so a choice made in time still has room to travel. */
export const CONFIRM_VISIBLE_MS = 30_000

/** Commands longer than this are shown in full in the transcript instead of
 *  inside the prompt, where they would push the choices off the screen. */
const COMMAND_PREVIEW_CHARS = 200

type PromptKind = 'approval' | 'clarify' | 'confirm'

type PromptView = ApprovalPrompt | ClarifyPrompt | ConfirmPrompt

interface PendingPrompt {
  approval?: ApprovalRequest
  clarify?: ClarifyRequest
  /** Which local command was running when a confirm arrived, if any. */
  commandToken?: symbol
  confirm?: ConfirmRequest
  /** Conversation the prompt belongs to; null for connection-scoped confirms. */
  conversationId: null | string
  /** Connection generation: a record from a dropped socket can never match. */
  epoch: number
  id: string
  kind: PromptKind
  lease?: SelectorLease
  settled: boolean
  timer?: CountdownTimer
  view?: PromptView
}

export interface PromptCoordinatorOptions {
  clock: CountdownClock
  gateway: Gateway
  host: SelectorHost
  keybindings: KeybindingsManager
  /** Told when a prompt starts and stops waiting for an answer, so the window
   *  title can carry an attention cue while the user is away from the tab. */
  onAttention?: (waiting: boolean) => void
  /** Told whenever the pending set changes, so the app re-evaluates its gates. */
  onChange?: () => void
  /** The live conversation. Approvals and questions for any other one are
   *  answered by their own id and never shown here. */
  sessionId: () => null | string
  theme: Theme
  transcript: {
    print(text: string, tone?: 'error' | 'muted' | 'ok' | 'warn'): void
    printBlock(text: string, title?: string): void
  }
  tui: TUI
}

export class PromptCoordinator {
  private attention = false
  private commandToken: symbol | undefined
  private connectionEpoch = 0
  private readonly options: PromptCoordinatorOptions
  private readonly queue: PendingPrompt[] = []
  private readonly seen = new Set<string>()

  constructor(options: PromptCoordinatorOptions) {
    this.options = options
  }

  /** Something is pending, so nothing may be sent to the model yet. */
  get blocked(): boolean {
    return this.queue.length > 0
  }

  /** The prompt on screen, for tests and for the app's key router. */
  get active(): PromptView | undefined {
    return this.queue[0]?.view
  }

  /** Mark the start of an interactive command, so a `confirm.request` that
   *  arrives while it runs can be attributed to it. */
  beginCommand(): symbol {
    const token = Symbol('command')

    this.commandToken = token

    return token
  }

  /** The command finished. A confirm still pending under its token is stale:
   *  the backend already applied its own default when the call returned, so it
   *  is dropped locally rather than answered. */
  endCommand(token: symbol): void {
    if (this.commandToken === token) {
      this.commandToken = undefined
    }

    for (const record of [...this.queue]) {
      if (record.kind === 'confirm' && record.commandToken === token) {
        this.drop(record)
      }
    }
  }

  handleNotification(method: GatewayNotification['method'], params: unknown): void {
    switch (method) {
      case 'approval.closed':
        this.onApprovalClosed(params)
        break

      case 'approval.request':
        this.onApprovalRequest(params)
        break

      case 'clarify.request':
        this.onClarifyRequest(params)
        break

      case 'confirm.request':
        this.onConfirmRequest(params)
        break
    }
  }

  /** A different session is live now: nothing pending belongs to it. */
  onSessionChange(): void {
    for (const record of [...this.queue]) {
      this.cancelOutstanding(record)
    }
  }

  /**
   * The turn is over — cancelled, failed, or finished with a question still
   * outstanding. Its approvals and questions cannot be answered usefully any
   * more, and leaving one on screen keeps the queue blocked until someone
   * presses Escape. Confirms belong to commands rather than to turns, so they
   * are left alone.
   */
  onTurnError(sessionId: null | string): void {
    for (const record of [...this.queue]) {
      if (record.kind === 'confirm') {
        continue
      }

      if (!record.conversationId || record.conversationId === sessionId) {
        this.cancelOutstanding(record)
      }
    }
  }

  /** The socket went away. The brokers have their own disconnect fail-safes, so
   *  nothing is sent; the local records just go. */
  onDisconnect(): void {
    this.connectionEpoch += 1
    this.seen.clear()

    for (const record of [...this.queue]) {
      this.drop(record)
    }
  }

  dispose(): void {
    for (const record of [...this.queue]) {
      this.drop(record)
    }
  }

  // ── arrivals ───────────────────────────────────────────────────────

  private onApprovalRequest(params: unknown): void {
    const request = decodeApprovalRequest(params)

    if (!request) {
      this.options.transcript.print('ignored an approval request the gateway sent without an id', 'warn')

      return
    }

    if (this.isDuplicate('approval', request.approvalId)) {
      return
    }

    const live = this.options.sessionId()

    if (live && request.conversationId !== live) {
      // Another conversation's question. Answering it here would approve a
      // command this screen never showed, so it is denied by its own id.
      void this.respondApproval(request, 'deny')
      this.options.transcript.print('denied an approval request from another session', 'warn')

      return
    }

    this.enqueue({
      approval: request,
      conversationId: request.conversationId,
      epoch: this.connectionEpoch,
      id: request.approvalId,
      kind: 'approval',
      settled: false
    })
  }

  private onApprovalClosed(params: unknown): void {
    const closed = decodeApprovalClosed(params)

    if (!closed) {
      return
    }

    const record = this.queue.find(
      item =>
        item.kind === 'approval' &&
        item.id === closed.approvalId &&
        item.approval?.conversationId === closed.conversationId
    )

    if (record) {
      // The broker already decided. Never respond to it again.
      this.drop(record)
    }
  }

  private onClarifyRequest(params: unknown): void {
    const request = decodeClarifyRequest(params)

    if (!request) {
      this.options.transcript.print('ignored a clarify request the gateway sent without an id', 'warn')

      return
    }

    if (this.isDuplicate('clarify', request.requestId)) {
      return
    }

    const live = this.options.sessionId()

    if (live && request.conversationId && request.conversationId !== live) {
      void this.respondClarify(request.requestId, { cancelled: true })
      this.options.transcript.print('cancelled a question from another session', 'warn')

      return
    }

    // One question per conversation: a new one means the broker already
    // defaulted the old one, so the old record goes without a reply.
    for (const record of [...this.queue]) {
      if (record.kind === 'clarify' && record.conversationId === request.conversationId) {
        this.drop(record)
      }
    }

    this.enqueue({
      clarify: request,
      conversationId: request.conversationId || null,
      epoch: this.connectionEpoch,
      id: request.requestId,
      kind: 'clarify',
      settled: false
    })
  }

  private onConfirmRequest(params: unknown): void {
    const request = decodeConfirmRequest(params)

    if (!request) {
      this.options.transcript.print('ignored a malformed confirm request', 'warn')

      return
    }

    if (this.isDuplicate('confirm', request.requestId)) {
      return
    }

    this.enqueue({
      confirm: request,
      // The payload carries no session, and inventing one would let a session
      // switch cancel a confirm that has nothing to do with it.
      conversationId: null,
      epoch: this.connectionEpoch,
      id: request.requestId,
      kind: 'confirm',
      settled: false,
      ...(this.commandToken ? { commandToken: this.commandToken } : {})
    })
  }

  private isDuplicate(kind: PromptKind, id: string): boolean {
    const key = `${this.connectionEpoch}:${kind}:${id}`

    if (this.seen.has(key)) {
      return true
    }

    this.seen.add(key)

    return false
  }

  // ── queue ──────────────────────────────────────────────────────────

  private enqueue(record: PendingPrompt): void {
    this.queue.push(record)
    this.startTimer(record)
    this.announceAttention()
    this.options.onChange?.()
    this.pump()
  }

  /** Say so only when the answer changes: the title is rewritten each time. */
  private announceAttention(): void {
    const waiting = this.queue.length > 0

    if (waiting !== this.attention) {
      this.attention = waiting
      this.options.onAttention?.(waiting)
    }
  }

  /** Deadlines run from arrival, not from display: a prompt that waited behind
   *  another one has less time left, not a fresh window. */
  private startTimer(record: PendingPrompt): void {
    const deadline = this.deadlineFor(record)

    if (deadline === null) {
      return
    }

    record.timer = new CountdownTimer(
      deadline,
      this.options.clock,
      this.options.tui,
      seconds => {
        const view = record.view

        if (view instanceof ApprovalPrompt || view instanceof ConfirmPrompt) {
          view.setSeconds(seconds)
        }
      },
      () => this.expire(record)
    )
  }

  private deadlineFor(record: PendingPrompt): null | number {
    if (record.kind === 'approval') {
      // Unix seconds on the wire. A zero or missing value is a request we
      // cannot time, and an approval we cannot time is one we will not grant.
      return (record.approval?.expiresAt ?? 0) * 1000
    }

    if (record.kind === 'confirm') {
      return this.options.clock.nowMs() + CONFIRM_VISIBLE_MS
    }

    // Clarify: the broker's timeout is not transmitted, so there is nothing
    // honest to count down from.
    return null
  }

  private pump(): void {
    const head = this.queue[0]

    if (!head || head.view || head.settled) {
      return
    }

    if (head.timer?.isExpired) {
      // It ran out while it waited behind another prompt. Settle it rather than
      // showing a window that is already closed.
      this.expire(head)

      return
    }

    this.options.host.show(lease => {
      head.lease = lease

      const view = this.buildView(head)

      head.view = view
      head.timer?.checkNow()

      return {
        component: view,
        dispose: () => {
          head.view = undefined
        },
        focus: view
      }
    })
  }

  private buildView(record: PendingPrompt): PromptView {
    const { keybindings, theme } = this.options

    if (record.kind === 'approval' && record.approval) {
      const request = record.approval
      const oneLine = request.command.split('\n')[0] ?? ''
      const truncated = request.command.length > COMMAND_PREVIEW_CHARS || request.command.includes('\n')

      if (truncated) {
        // Only claim the full text is above when it actually was appended.
        this.options.transcript.printBlock(request.command, 'Command awaiting approval')
      }

      return new ApprovalPrompt({
        command: truncated ? `${oneLine.slice(0, COMMAND_PREVIEW_CHARS)}…` : request.command,
        description: request.description,
        keybindings,
        onChoose: choice => this.answerApproval(record, choice),
        theme,
        truncated
      })
    }

    if (record.kind === 'clarify' && record.clarify) {
      return new ClarifyPrompt({
        choices: record.clarify.choices,
        keybindings,
        onAnswer: answer => this.answerClarify(record, { answer }),
        onCancel: () => this.answerClarify(record, { cancelled: true }),
        question: record.clarify.question,
        theme
      })
    }

    const confirm = record.confirm!

    return new ConfirmPrompt({
      defaultAnswer: confirm.defaultAnswer,
      keybindings,
      mode: 'rpc',
      onAnswer: value => this.answerConfirm(record, value),
      prompt: confirm.prompt,
      theme,
      title: 'Confirm'
    })
  }

  // ── settlement ─────────────────────────────────────────────────────

  private answerApproval(record: PendingPrompt, choice: 'allow' | 'deny'): void {
    if (!record.approval) {
      return
    }

    // The second between ticks is exactly where a late "allow" would slip
    // through, so the deadline is rechecked before the choice is taken.
    record.timer?.checkNow()

    if (record.settled) {
      return
    }

    this.close(record, this.respondApproval(record.approval, choice))
  }

  private answerClarify(record: PendingPrompt, response: { answer: string } | { cancelled: true }): void {
    if (record.settled || !record.clarify) {
      return
    }

    this.close(record, this.respondClarify(record.clarify.requestId, response))
  }

  private answerConfirm(record: PendingPrompt, value: boolean): void {
    record.timer?.checkNow()

    if (record.settled || !record.confirm) {
      return
    }

    this.close(record, this.options.gateway.confirmRespond(record.confirm.requestId, value))
  }

  /** The deadline passed. Approval fails closed; confirm sends the request's
   *  own default, which is what the broker would apply anyway. */
  private expire(record: PendingPrompt): void {
    if (record.settled) {
      return
    }

    if (record.kind === 'approval' && record.approval) {
      this.options.transcript.print('approval request expired — denied', 'warn')
      this.close(record, this.respondApproval(record.approval, 'deny'))

      return
    }

    if (record.kind === 'confirm' && record.confirm) {
      this.close(record, this.options.gateway.confirmRespond(record.confirm.requestId, record.confirm.defaultAnswer))
    }
  }

  /**
   * Session switch and turn end: say no, by each record's own id.
   *
   * Quietly: a cancelled turn's broker has usually defaulted its own waiters
   * before this runs, so `ok:false` here is the expected answer rather than
   * something worth telling the user about.
   */
  private cancelOutstanding(record: PendingPrompt): void {
    if (record.settled) {
      return
    }

    if (record.kind === 'approval' && record.approval) {
      this.close(record, this.respondApproval(record.approval, 'deny'), true)

      return
    }

    if (record.kind === 'clarify' && record.clarify) {
      this.close(record, this.respondClarify(record.clarify.requestId, { cancelled: true }), true)

      return
    }

    if (record.kind === 'confirm' && record.confirm) {
      this.close(record, this.options.gateway.confirmRespond(record.confirm.requestId, false), true)
    }
  }

  /** Close a record without answering: the broker already decided, or there is
   *  no socket left to answer on. */
  private drop(record: PendingPrompt): void {
    if (record.settled) {
      return
    }

    record.settled = true
    record.timer?.dispose()
    this.finish(record)
  }

  /** Settle synchronously, then let the reply land. Nothing the RPC returns can
   *  reopen the prompt; a refusal or an error is reported and no more. */
  private close(record: PendingPrompt, sent: Promise<{ ok: boolean }>, quiet = false): void {
    record.settled = true
    record.timer?.dispose()
    record.view?.setBusy(true)

    void sent
      .then(result => {
        if (!result.ok && !quiet) {
          // Stale, already answered, or not accepted — never "approved".
          this.options.transcript.print('the gateway had already closed that request', 'warn')
        }
      })
      .catch((err: unknown) => {
        if (!quiet) {
          this.options.transcript.print(`could not send that answer: ${message(err)}`, 'error')
        }
      })
      .finally(() => this.finish(record))
  }

  private finish(record: PendingPrompt): void {
    const index = this.queue.indexOf(record)

    if (index >= 0) {
      this.queue.splice(index, 1)
    }

    record.timer?.dispose()

    if (record.lease?.isCurrent()) {
      record.lease.done('done')
    }

    record.view = undefined
    record.lease = undefined
    this.announceAttention()
    this.options.onChange?.()
    this.pump()
  }

  private respondApproval(request: ApprovalRequest, choice: 'allow' | 'deny'): Promise<{ ok: boolean }> {
    return this.options.gateway.approvalRespond(request.approvalId, request.conversationId, choice)
  }

  private respondClarify(
    requestId: string,
    response: { answer: string } | { cancelled: true }
  ): Promise<{ ok: boolean }> {
    return this.options.gateway.clarifyRespond(requestId, response)
  }
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}
