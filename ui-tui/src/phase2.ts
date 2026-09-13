// Phase 2 in one place: who may own the editor slot, and what the app has to
// stop doing while something else does.
//
// The app keeps its transcript, its turn controller, its queue and its editor.
// This module adds the slot around that editor, the pickers that can
// stand in it, the coordinator that puts broker prompts there ahead of any
// picker, the fan-out of pushed sign-in steps to whichever stage is showing one,
// and the single answer to "is it safe to send something right now".

import type { Component, Container, KeybindingsManager, TUI } from '@earendil-works/pi-tui'

import type { CountdownClock } from './components/countdownTimer.js'
import type { Gateway, GatewayNotification, LoginStepPush } from './gateway.js'
import type { SelectorDeps, SelectorSession, TranscriptSink } from './selectors/deps.js'
import type { ModelListActions, ModelScope } from './selectors/modelList.js'
import type { EnabledIds } from './selectors/scopedModels.js'
import type { SessionSelectorActions } from './selectors/sessionSelector.js'
import type { Theme } from './theme.js'

import { systemClock } from './components/countdownTimer.js'
import { SelectorHost } from './components/selector.js'
import { openBrowser } from './lib/openBrowser.js'
import { PromptCoordinator } from './prompts/coordinator.js'
import { createAuthSelector } from './selectors/authSelector.js'
import { createModelListSelector } from './selectors/modelList.js'
import { createScopedModelsSelector } from './selectors/scopedModels.js'
import { createSessionSelector } from './selectors/sessionSelector.js'
import { createThinkingSelector } from './selectors/thinkingSelector.js'

export interface Phase2Options {
  /** Injected in tests so countdowns are deterministic. */
  clock?: CountdownClock
  editor: Component
  /** The slot: holds the editor, or whatever replaces it. */
  editorContainer: Container
  gateway: Gateway
  keybindings: KeybindingsManager
  /** Applied model facts, for the footer and the session panel. */
  onModelApplied: ModelListActions['applied']
  /** Called whenever the answer to "may anything be sent" might have changed. */
  onGateChange?: () => void
  /** A broker prompt is waiting for an answer, or has stopped waiting. */
  onPromptAttention?: (waiting: boolean) => void
  /** Applied reasoning effort, for the footer. */
  onThinkingApplied: (effort: string | undefined, model: string) => void
  sessionActions: SessionSelectorActions
  /** The live session, read when it is asked for. */
  session: () => SelectorSession
  theme: Theme
  transcript: TranscriptSink
  tui: TUI
}

export class Phase2 {
  readonly prompts: PromptCoordinator
  readonly selectors: SelectorHost

  private commandToken: symbol | undefined
  private readonly deps: SelectorDeps
  /** Whoever is showing a sign-in right now. Empty the rest of the time. */
  private readonly loginListeners = new Set<(step: LoginStepPush) => void>()
  private readonly options: Phase2Options
  /** pi's session-only model scope: chosen in `/scoped-models` this session and
   *  not yet saved. Undefined follows the saved scope. */
  private scopedModels: EnabledIds | undefined
  private transition = false

  constructor(options: Phase2Options) {
    this.options = options
    this.selectors = new SelectorHost({
      editor: options.editor,
      editorContainer: options.editorContainer,
      onChange: () => options.onGateChange?.(),
      session: () => {
        const live = options.session()

        return { epoch: live.epoch, id: live.id }
      },
      tui: options.tui
    })

    this.prompts = new PromptCoordinator({
      clock: options.clock ?? systemClock(),
      gateway: options.gateway,
      host: this.selectors,
      keybindings: options.keybindings,
      onAttention: waiting => options.onPromptAttention?.(waiting),
      onChange: () => options.onGateChange?.(),
      sessionId: () => options.session().id,
      theme: options.theme,
      transcript: options.transcript,
      tui: options.tui
    })

    this.deps = {
      gateway: options.gateway,
      keybindings: options.keybindings,
      onLoginStep: handler => {
        this.loginListeners.add(handler)

        return () => this.loginListeners.delete(handler)
      },
      openBrowser,
      session: options.session,
      theme: options.theme,
      transcript: options.transcript,
      tui: options.tui
    }
  }

  /** Something other than the editor is in the slot, so the app's global key
   *  listener must stand aside. */
  get slotActive(): boolean {
    return this.selectors.active
  }

  /** Nothing may be sent to the model: a picker, a prompt, a sign-in or a
   *  session switch is in the way. A sign-in is one of the picker's stages, so
   *  `selectors.active` covers it. */
  get blocked(): boolean {
    return this.selectors.active || this.prompts.blocked || this.transition
  }

  /** An interactive command is running; a second one would race it and would
   *  make a `confirm.request` impossible to attribute. */
  get commandBusy(): boolean {
    return this.commandToken !== undefined
  }

  /** A session is being adopted. Holds the gate across the switch. */
  setTransition(active: boolean): void {
    this.transition = active
    this.options.onGateChange?.()
  }

  beginCommand(): symbol {
    const token = this.prompts.beginCommand()

    this.commandToken = token

    return token
  }

  endCommand(token: symbol): void {
    if (this.commandToken === token) {
      this.commandToken = undefined
    }

    this.prompts.endCommand(token)
  }

  handleNotification(method: GatewayNotification['method'], params: unknown): void {
    if (method === 'login.step') {
      for (const listener of [...this.loginListeners]) {
        listener(params as LoginStepPush)
      }

      return
    }

    this.prompts.handleNotification(method, params)
  }

  /** A new session is live: pending prompts belong to the old one, and any
   *  picker was opened against it. */
  onSessionChange(): void {
    this.prompts.onSessionChange()
    this.selectors.reset('session-switch')
  }

  onTurnError(sessionId: null | string): void {
    this.prompts.onTurnError(sessionId)
    this.selectors.reset('turn-error')
  }

  /**
   * The turn ended without an error — normally, or because it was cancelled.
   *
   * A cancelled turn's question is as unanswerable as a failed one's, and the
   * broker has already defaulted it, so the prompt is retired and the queue
   * unblocked. A picker the user opened during the turn is left alone: they
   * opened it, and the turn finishing is no reason to take it away.
   */
  onTurnEnded(sessionId: null | string): void {
    this.prompts.onTurnError(sessionId)
  }

  onDisconnect(): void {
    this.prompts.onDisconnect()
    this.selectors.reset('disconnect')
  }

  dispose(): void {
    this.prompts.dispose()
    this.selectors.reset('shutdown')
  }

  // ── pickers ────────────────────────────────────────────────────────

  /** pi's `/model`: every model the connected providers serve, in one list;
   *  the session's scope when one was chosen this session, else the saved one. */
  openModelPicker(scope: ModelScope, query?: string): void {
    this.selectors.show(lease =>
      createModelListSelector({
        actions: { applied: this.options.onModelApplied },
        deps: this.deps,
        lease,
        scope,
        sessionScope: this.scopedModels,
        ...(query?.trim() ? { initialQuery: query.trim() } : {})
      })
    )
  }

  /** pi's `/scoped-models`: which models `/model` shows under "scoped". */
  openScopedModelsPicker(): void {
    this.selectors.show(lease =>
      createScopedModelsSelector({
        deps: this.deps,
        lease,
        onChange: enabled => {
          this.scopedModels = enabled
        },
        sessionScope: this.scopedModels
      })
    )
  }

  /** pi's `/login`: which way in, then which provider, then the sign-in. */
  openLoginPicker(provider?: string): void {
    this.selectors.show(lease =>
      createAuthSelector({
        deps: this.deps,
        entry: 'login',
        lease,
        ...(provider?.trim() ? { entryProvider: provider.trim() } : {})
      })
    )
  }

  /** pi's `/logout`: the credentials this machine holds. */
  openLogoutPicker(): void {
    this.selectors.show(lease => createAuthSelector({ deps: this.deps, entry: 'logout', lease }))
  }

  openThinkingPicker(): void {
    this.selectors.show(lease =>
      createThinkingSelector({ deps: this.deps, lease, onApplied: this.options.onThinkingApplied })
    )
  }

  openSessionPicker(): void {
    this.selectors.show(lease =>
      createSessionSelector({ actions: this.options.sessionActions, deps: this.deps, lease })
    )
  }
}
