// Which session is live, and every path that replaces it: opening one at
// boot, `/new`, `/resume`, `/branch`, and deleting the one you are in.
//
// What is *known* about a session lives next door in `sessionFacts.ts`; this
// file is about identity and the switch itself — the reservation that keeps a
// prompt from starting on a session that is about to go away.

import type { Container, TUI } from '@earendil-works/pi-tui'

import type { HarnessEditor } from '../components/editor.js'
import type { MessageQueue } from '../components/queue.js'
import type { SystemTone } from '../components/systemLine.js'
import type { Gateway } from '../gateway.js'
import type { Phase2 } from '../phase2.js'
import type { Phase3 } from '../phase3.js'
import type { JsonValue, SessionCreateResult, SessionResumeResult } from '../rpc/index.js'
import type { Theme } from '../theme.js'
import type { TurnController } from '../turn.js'
import type { SessionFacts } from './sessionFacts.js'
import type { TranscriptView } from './transcript.js'
import type { TurnRunner } from './turnRunner.js'

import { pickPlaceholder } from '../content/placeholders.js'
import { renderTranscript } from '../resume.js'

/** Whether the gateway's config says to reopen the most recent session.
 *  `config.get` serves flat dotted keys, but has served a nested `display`
 *  block in the past, so read both shapes. */
function autoResumeRecent(config: Record<string, JsonValue>): boolean {
  if (config['display.tui_auto_resume_recent'] === true) {
    return true
  }

  const display = config.display

  return typeof display === 'object' && display !== null && !Array.isArray(display)
    ? (display as Record<string, JsonValue>).tui_auto_resume_recent === true
    : false
}

export interface SessionHubOptions {
  /** The transcript's container, for replaying a resumed session into. */
  chat: Container
  /** Built after this hub is, and so reached at call time — as phase 2, phase
   *  3 and the turn runner are. Nothing here touches them while constructing. */
  editor: () => HarnessEditor
  env: NodeJS.ProcessEnv
  facts: SessionFacts
  gateway: Gateway
  /** What a session switch has to wait for, beyond this hub's own reservation. */
  isTurnActive: () => boolean
  /** Cleared when a session is replaced: the counters belong to the old one. */
  onSessionReplaced: () => void
  phase2: () => Phase2
  phase3: () => Phase3
  print: (text: string, tone?: SystemTone) => void
  queue: MessageQueue
  theme: Theme
  transcript: TranscriptView
  tui: TUI
  turn: TurnController
  turns: () => TurnRunner
}

export class SessionHub {
  private epoch = 0
  private key: null | string = null
  /** A session transition is reserved. Set before the opening RPC and held
   *  until the new session is attached, so nothing starts a turn on the
   *  session that is about to be replaced. */
  private switching = false

  constructor(private readonly opts: SessionHubOptions) {}

  get id(): null | string {
    return this.key
  }

  /** A transition is reserved. The queue dispatches nothing while it is. */
  get inTransition(): boolean {
    return this.switching
  }

  /** A new generation each time a session is adopted. Anything that was in
   *  flight for the previous one checks this and stands down. */
  get generation(): number {
    return this.epoch
  }

  async openSession(config: Record<string, JsonValue>): Promise<SessionCreateResult | SessionResumeResult> {
    const requested = (this.opts.env.OPENDDE_HARNESS_TUI_RESUME ?? '').trim()

    if (requested) {
      return this.opts.gateway.sessionResume(requested)
    }

    if (autoResumeRecent(config)) {
      const recent = await this.opts.gateway.sessionMostRecent()

      if (recent.session_id) {
        return this.opts.gateway.sessionResume(recent.session_id)
      }
    }

    return this.opts.gateway.sessionCreate()
  }

  /**
   * One session transition, reserved before the first RPC of it.
   *
   * Everything that replaces the live session goes through here: `/new`,
   * `/resume`, `/branch` and deleting the session you are in. Checking that a
   * turn is idle and then awaiting the opening call is not enough — in that
   * window a queued prompt used to start on the session about to be thrown
   * away, and its reply landed in a transcript nobody could see. The
   * reservation closes the window: while it is held the queue dispatches
   * nothing and a second transition is refused.
   *
   * Queued prompts are kept rather than dropped. They are the user's, not the
   * old session's, and they go out on the new session once it is attached.
   * (A switch asked for while messages are already queued is refused outright
   * — see {@link switchBlockedReason} — so this is about prompts typed during
   * the switch itself.)
   */
  async transition<T>(run: () => Promise<T>, refused: T): Promise<T> {
    if (this.switching) {
      this.opts.print('a session switch is already running', 'warn')
      this.opts.tui.requestRender()

      return refused
    }

    this.switching = true

    try {
      return await run()
    } finally {
      this.switching = false
      this.opts.tui.requestRender()
      // Nothing else drains after a switch, and anything typed during it is
      // still waiting.
      this.opts.turns().scheduleDrain(0)
    }
  }

  /** A fork the gateway minted for us: same transcript, new key, and the
   *  parent's model and provider, which the branch response does not repeat. */
  async adoptFork(id: string, title?: string): Promise<void> {
    await this.transition(async () => {
      await this.adoptSession({ session_id: id, info: undefined })

      if (title) {
        this.opts.facts.setTitle(title)
      }
    }, undefined)
  }

  /** Make `opened` the live session: subscribe to it, show its facts, replay
   *  its messages. A local fork adoption supplies only the id, so this accepts
   *  a partial resume result. The previous session, if any, is closed first. */
  async adoptSession(opened: Pick<SessionCreateResult, 'session_id'> & Partial<SessionResumeResult>): Promise<void> {
    this.opts.phase2().setTransition(true)

    try {
      await this.leaveSession()

      // A new generation: pending prompts and open pickers belonged to the
      // session that just went away.
      this.epoch += 1
      this.opts.phase2().onSessionChange()

      this.key = opened.session_id
      // A fresh suggestion per session, so an empty prompt is never the same
      // sentence twice in a row.
      this.opts.editor().setPlaceholder(pickPlaceholder())
      this.opts.facts.applySessionInfo(opened.session_id, opened.info)

      if (opened.messages?.length) {
        renderTranscript(this.opts.chat, this.opts.theme, this.opts.tui, opened.messages, {
          quiet: this.opts.turn.quietTools
        })
      }

      await this.opts.turn.attach(opened.session_id)
      // After attaching, which starts this session's accounting from zero.
      // Null travels: it is the gateway saying this conversation cannot be
      // sized, which is not the same as a session holding nothing.
      const opening = this.opts.facts.info.contextTokens

      this.opts.turn.setContextBaseline(opening === undefined ? 0 : opening)
    } finally {
      this.opts.phase2().setTransition(false)
    }

    this.opts.tui.requestRender()
  }

  private async leaveSession(): Promise<void> {
    const previous = this.key

    await this.opts.turn.detach()
    this.key = null

    // The queue is deliberately not cleared. A prompt typed while the switch
    // was in flight belongs to the person who typed it, not to the session
    // that happened to be open, and it goes out on the new one. A switch
    // requested with messages already waiting is refused instead.
    if (previous) {
      await this.opts.gateway.sessionClose(previous).catch(() => {
        // A session the gateway already dropped is closed enough.
      })
    }
  }

  /** Why a session switch cannot happen right now, for the picker to show. */
  switchBlockedReason(): null | string {
    if (this.switching) {
      return 'a session switch is already running'
    }

    if (this.opts.isTurnActive()) {
      return 'a turn is running'
    }

    if (this.opts.turns().cancelling) {
      return 'a cancel is still unwinding'
    }

    if (this.opts.queue.length > 0) {
      return `${this.opts.queue.length} message${this.opts.queue.length === 1 ? '' : 's'} are still queued`
    }

    return null
  }

  /**
   * Adopt `id`, unless something newer got there first.
   *
   * `session.resume` of an id the gateway cannot load answers with a *new*
   * empty session rather than an error, so the returned id is compared with the
   * requested one and the difference is said out loud.
   */
  async resumeSession(id: string, stillCurrent?: () => boolean): Promise<void> {
    await this.transition(() => this.openResumed(id, stillCurrent), undefined)
  }

  private async openResumed(id: string, stillCurrent?: () => boolean): Promise<void> {
    const epoch = this.epoch
    const opened = await this.opts.gateway.sessionResume(id)

    if ((stillCurrent && !stillCurrent()) || epoch !== this.epoch) {
      // A newer session was adopted while this call was in flight. Leave it
      // alone and let go of what we just opened.
      if (opened.session_id !== this.key) {
        await this.opts.gateway.sessionClose(opened.session_id).catch(() => {})
      }

      return
    }

    await this.switchSession(opened)

    if (opened.session_id === id) {
      this.opts.print(`resumed ${opened.session_id}`)
    } else {
      this.opts.print(`session unavailable; opened a new session ${opened.session_id}`, 'warn')
    }
  }

  /** `/new`: a session of its own, optionally titled. */
  async createSession(title?: string): Promise<void> {
    await this.transition(async () => {
      const opened = await this.opts.gateway.sessionCreate()

      await this.switchSession(opened)

      if (title) {
        await this.opts.gateway.sessionTitle(opened.session_id, title).catch(() => {})
        this.opts.facts.setTitle(title)
      }

      this.opts.print(`new session ${opened.session_id}`)
    }, undefined)
  }

  /**
   * Delete a session. The active one is left first — detached and closed —
   * because closing it afterwards would write the record back that the delete
   * just removed. A failed delete puts the old session back on screen.
   */
  async removeSession(id: string): Promise<boolean> {
    if (id !== this.key) {
      const result = await this.opts.gateway.sessionDelete(id)

      // `deleted: null` means nothing was removed.
      return result.deleted === id
    }

    // Deleting the session you are in replaces it, so it is a transition.
    return this.transition(() => this.removeCurrentSession(id), false)
  }

  private async removeCurrentSession(id: string): Promise<boolean> {
    await this.leaveSession()

    try {
      const result = await this.opts.gateway.sessionDelete(id)

      if (result.deleted !== id) {
        await this.switchSession(await this.opts.gateway.sessionResume(id))

        return false
      }
    } catch (err) {
      try {
        await this.switchSession(await this.opts.gateway.sessionResume(id))
      } catch {
        // The old session is gone too; the error below is what matters.
      }

      throw err
    }

    await this.switchSession(await this.opts.gateway.sessionCreate())

    return true
  }

  /** A fresh transcript for a session that replaces this one. */
  private async switchSession(opened: SessionCreateResult | SessionResumeResult): Promise<void> {
    this.opts.phase3().queueEditing.reset()
    this.opts.transcript.clear()
    this.opts.onSessionReplaced()
    await this.adoptSession(opened)
  }
}
