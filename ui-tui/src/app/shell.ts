// The terminal around the app: who reads its keys, what happens when it is
// resized, and the order things are let go of on the way out.

import type { TUI } from '@earendil-works/pi-tui'

import type { HarnessEditor } from '../components/editor.js'
import type { Gateway } from '../gateway.js'
import type { InputHistory } from '../lib/history.js'
import type { GlobalKeyHandler } from '../lib/keybindings.js'
import type { Phase2 } from '../phase2.js'
import type { Phase3 } from '../phase3.js'
import type { TurnController } from '../turn.js'
import type { SessionFacts } from './sessionFacts.js'
import type { TranscriptView } from './transcript.js'
import type { TurnRunner } from './turnRunner.js'

import { BRAND } from '../content/brand.js'

/** SIGWINCH bursts while a window is dragged; report the size once it settles. */
const RESIZE_DEBOUNCE_MS = 100

export function listenSigwinch(handler: () => void): () => void {
  process.on('SIGWINCH', handler)

  return () => void process.off('SIGWINCH', handler)
}

export interface AppShellOptions {
  editor: () => HarnessEditor
  gateway: Gateway
  /** Persisted prompt history, when this process keeps any. */
  history?: InputHistory
  keys: () => GlobalKeyHandler
  /** Subscribe to terminal resizes. Defaults to SIGWINCH; tests pass a stub. */
  listenResize: (handler: () => void) => () => void
  onExit: (code: number) => void
  /** Built after this; reached at call time. */
  phase2: () => Phase2
  phase3: () => Phase3
  facts: () => SessionFacts
  transcript: TranscriptView
  tui: TUI
  turn: TurnController
  turns: () => TurnRunner
}

export class AppShell {
  private readonly unlistenResize: () => void
  /** Undone by `quit`; held so a renderer switch can re-register it. */
  private unlistenInput: (() => void) | undefined
  private resizeTimer: null | ReturnType<typeof setTimeout> = null
  private leaving = false

  constructor(private readonly opts: AppShellOptions) {
    this.unlistenResize = opts.listenResize(() => this.onResize())
  }

  /** The UI is on its way out; nothing should act on a late event. */
  get quitting(): boolean {
    return this.leaving
  }

  /** The global key listener belongs to whichever renderer is on screen, so a
   *  fullscreen switch registers it again on the new one. */
  installInputListener(): void {
    this.unlistenInput?.()
    this.unlistenInput = this.opts.tui.addInputListener(data => this.opts.keys().handleGlobalKey(data))
  }

  /** $VISUAL owns the terminal for the length of the edit; nothing polls the
   *  task root while it does. */
  async openExternalEditor(): Promise<void> {
    this.opts.phase3().setHandoff(true)

    try {
      await this.opts.editor().openExternal()
    } finally {
      this.opts.phase3().setHandoff(false)
    }
  }

  onResize(): void {
    if (this.resizeTimer) {
      clearTimeout(this.resizeTimer)
    }

    this.resizeTimer = setTimeout(() => {
      this.resizeTimer = null

      const { columns, rows } = this.opts.tui.terminal

      void this.opts.gateway.terminalResize(columns, rows).catch(() => {
        // The size is a courtesy for CLI output; a refusal changes nothing here.
      })
    }, RESIZE_DEBOUNCE_MS)
    this.resizeTimer.unref?.()
  }

  quit(code: number): void {
    if (this.leaving) {
      return
    }

    this.leaving = true
    this.opts.phase2().dispose()
    this.opts.phase3().dispose()
    this.unlistenResize()
    this.unlistenInput?.()
    this.opts.facts().dispose()
    this.opts.turns().dispose()
    // A prompt that lost the file lock is buffered until the next append, and
    // on the way out there is no next one.
    this.opts.history?.flush()

    void this.opts.turn.detach().catch(() => {})

    // Back to the main screen before the last render, so what was on the
    // alternate screen is in the shell's scrollback rather than gone with it.
    this.opts.phase3().prepareExit()
    this.opts.transcript.print(BRAND.goodbye)
    this.opts.tui.renderNow()
    this.opts.tui.stop()
    this.opts.onExit(code)
  }
}
