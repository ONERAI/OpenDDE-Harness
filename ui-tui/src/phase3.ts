// Phase 3 in one place: the background work the app watches but does not run,
// and the screen it draws all of it on.
//
// Three things live here. The protein-design task monitor and its bar, which
// read the local task root and nothing else. The renderer owner, which lets the
// same component tree move between the main screen and the alternate screen at
// runtime. And the small bits of glue that only exist because of those two —
// editing a queued message, pasting over ssh, and masking the alternate
// screen's own keys while an inline selector owns the keyboard.

import type { Component, KeybindingsConfig, KeybindingsManager, TUI } from '@earendil-works/pi-tui'

import { Container, TuiAltScreen } from '@earendil-works/pi-tui'

import type { RendererAccess, TaskAccess } from './commands/index.js'
import type { CountdownClock } from './components/countdownTimer.js'
import type { MessageQueue } from './components/queue.js'
import type { SystemTone } from './components/systemLine.js'
import type { EditableInput } from './queueEditing.js'
import type { RendererMode, RendererOwner } from './renderer.js'
import type { ProteinProgressPayload } from './tasks/monitor.js'
import type { Theme } from './theme.js'

import { createChatViewport } from './chatViewport.js'
import { systemClock } from './components/countdownTimer.js'
import { ProteinDesignTaskBar } from './components/proteinDesignTaskBar.js'
import { readClipboardText } from './lib/clipboard.js'
import { readOsc52Clipboard } from './lib/osc52Read.js'
import { QueueEditing } from './queueEditing.js'
import { ProteinDesignTasks } from './tasks/monitor.js'

/** The alternate screen's own keys, silenced while a selector or a broker
 *  prompt owns input. The renderer registers its viewport listener before the
 *  app's, so an unmasked PageUp would scroll the transcript out from under a
 *  picker rather than move its selection. */
const ALT_SCREEN_BINDINGS = [
  'tui.altScreen.bottom',
  'tui.altScreen.halfPageDown',
  'tui.altScreen.halfPageUp',
  'tui.altScreen.lineDown',
  'tui.altScreen.lineUp',
  'tui.altScreen.nextPrompt',
  'tui.altScreen.pageDown',
  'tui.altScreen.pageUp',
  'tui.altScreen.previousPrompt',
  'tui.altScreen.search',
  'tui.altScreen.searchClose',
  'tui.altScreen.searchNext',
  'tui.altScreen.searchPrevious',
  'tui.altScreen.top'
] as const

export interface Phase3Options {
  /** The transcript. Joined with the session panel into one scrollable half. */
  chat: Component
  clock?: CountdownClock
  editor: EditableInput
  /** Send whatever the queue can now send. Called when a queue edit lets go of
   *  the message it had reserved. */
  drainQueue?: () => void
  /** The editor slot. Docked, so it stays on screen in fullscreen. */
  editorContainer: Component
  env?: NodeJS.ProcessEnv
  footer: Component
  keybindings: KeybindingsManager
  /** The session panel. Scrolls with the transcript, like pi's header. */
  panel: Component
  queue: MessageQueue
  /** The queue's own view, between the bar and the editor. */
  queueView: Component
  /** Re-register the app's global input listener; the old registration
   *  belonged to the renderer that just stopped. */
  rebindInput?: () => void
  /** Absent in tests and in any build that owns a plain TUI. */
  renderer?: RendererOwner
  /** Something changed that the screen should show. */
  requestRender: () => void
  /** The busy indicator's row, between the queue and the editor. */
  status: Component
  /** The session generation, for the epoch guard on awaited task output. */
  sessionEpoch: () => number
  theme: Theme
  /** One line into the transcript. */
  print: (text: string, tone?: SystemTone) => void
  tui: TUI
}

/**
 * The task monitor as `/tasks` and `/task` see it.
 *
 * Built from a thunk because the command context exists before phase 3 does:
 * the editor's completion popup needs the context at construction, and the
 * monitor needs the editor.
 */
export function taskAccess(phase3: () => Phase3, sessionEpoch: () => number): TaskAccess {
  return {
    focus: taskId => phase3().tasks.focus(taskId),
    list: () => phase3().tasks.list(),
    readLogTail: (taskId, maxLines) => phase3().tasks.readLogTail(taskId, maxLines),
    refresh: () => phase3().tasks.refresh(),
    resolve: query => phase3().tasks.resolve(query),
    rootFailure: () => phase3().tasks.rootFailure,
    rootPath: () => phase3().tasks.rootPath,
    sessionEpoch,
    setVisible: visible => phase3().tasks.setVisible(visible),
    visible: () => phase3().tasks.visible
  }
}

export class Phase3 {
  /** The scrolling half of the layout: session panel over the transcript. */
  readonly document: Container
  readonly queueEditing: QueueEditing
  readonly taskBar: ProteinDesignTaskBar
  readonly tasks: ProteinDesignTasks

  private readonly options: Phase3Options
  private readonly unsubscribe: () => void

  private handoff = false
  private masked = false
  private savedBindings: KeybindingsConfig | undefined

  constructor(options: Phase3Options) {
    this.options = options

    this.tasks = new ProteinDesignTasks({
      clock: options.clock ?? systemClock(),
      onTerminal: (text, ok) => options.print(text, ok ? 'ok' : 'warn'),
      ...(options.env ? { env: options.env } : {})
    })

    this.taskBar = new ProteinDesignTaskBar(options.theme, this.tasks)
    this.unsubscribe = this.tasks.subscribe(() => {
      this.taskBar.invalidate()
      options.requestRender()
    })

    this.queueEditing = new QueueEditing({
      editor: options.editor,
      onChange: () => options.requestRender(),
      // The queue may have been holding for this edit: a turn that ended while
      // the head was reserved found nothing it was allowed to send.
      ...(options.drainQueue ? { onReleased: options.drainQueue } : {}),
      queue: options.queue
    })

    // One container for the two components that scroll together. In regular
    // mode it renders exactly as the two of them did one under the other.
    this.document = new Container()
    this.document.addChild(options.panel)
    this.document.addChild(options.chat)
  }

  /** Top to bottom, the app's children. The same list in both modes. */
  get children(): Component[] {
    return [
      this.document,
      this.taskBar,
      this.options.queueView,
      this.options.status,
      this.options.editorContainer,
      this.options.footer
    ]
  }

  get renderer(): RendererAccess | undefined {
    const owner = this.options.renderer

    if (!owner) {
      return undefined
    }

    return { mode: () => owner.mode, switchTo: mode => this.switchTo(owner, mode) }
  }

  /** Mount the tree, and build the fullscreen layout over the same objects. */
  mount(): void {
    const owner = this.options.renderer

    for (const child of this.children) {
      this.options.tui.addChild(child)
    }

    if (!owner) {
      return
    }

    const viewport = createChatViewport({
      document: this.document,
      editor: this.options.editorContainer,
      footer: this.options.footer,
      queue: this.options.queueView,
      status: this.options.status,
      taskBar: this.taskBar,
      theme: this.options.theme
    })

    owner.setLayoutRoot(viewport.root)
    owner.onSwitched = () => this.options.rebindInput?.()
  }

  /** A `protein_design.progress` event, straight from the turn controller. */
  onProteinProgress(payload: ProteinProgressPayload): void {
    this.tasks.applyProgress(payload)
  }

  /** Something other than the editor owns the keyboard, or no longer does. */
  setSlotActive(active: boolean): void {
    if (!this.options.renderer || active === this.masked) {
      return
    }

    this.masked = active

    if (active) {
      this.savedBindings = this.options.keybindings.getUserBindings()
      this.options.keybindings.setUserBindings({
        ...this.savedBindings,
        ...Object.fromEntries(ALT_SCREEN_BINDINGS.map(name => [name, []]))
      })

      return
    }

    // The user's own bindings go back exactly as they were; the mask never
    // becomes their configuration.
    this.options.keybindings.setUserBindings(this.savedBindings ?? {})
    this.savedBindings = undefined
  }

  /** A child process is taking the terminal, or has given it back. Nothing
   *  with a recurring timer runs meanwhile, and the screen cannot change
   *  underneath the child. */
  setHandoff(active: boolean): void {
    this.handoff = active
    this.tasks.setPaused(active)
  }

  /** Ctrl+C with a fullscreen selection copies it, and nothing else. */
  copySelection(): boolean {
    const current = this.options.renderer?.renderer

    if (!(current instanceof TuiAltScreen) || !current.hasActiveSelection()) {
      return false
    }

    void current.copyActiveSelectionToClipboard()

    return true
  }

  /**
   * Paste into the editor.
   *
   * A native clipboard tool first, because when there is one it is the real
   * clipboard. OSC 52 second, which is the only path that works over ssh —
   * and which most terminals refuse, so a silent one is not an error worth
   * shouting about.
   */
  async paste(): Promise<void> {
    const native = await readClipboardText()
    const text = native ?? (await readOsc52Clipboard(this.options.tui, this.options.env ?? process.env))

    if (!text) {
      this.options.print('nothing to paste — your terminal did not answer the clipboard query', 'warn')
      this.options.requestRender()

      return
    }

    this.options.editor.insertTextAtCursor(text)
    this.options.requestRender()
  }

  /** Leaving. The app renders once and stops straight after this, and that
   *  render is what puts the conversation in the shell's own buffer rather
   *  than losing it with the alternate one. */
  prepareExit(): void {
    this.options.renderer?.exitToMainScreen()
  }

  dispose(): void {
    this.unsubscribe()
    this.tasks.dispose()
    this.setSlotActive(false)
  }

  private switchTo(owner: RendererOwner, mode: RendererMode): null | string {
    if (this.masked) {
      return 'finish what is on screen first'
    }

    if (this.handoff) {
      return 'a child process has the terminal'
    }

    const refused = owner.switchTo(mode)

    if (!refused) {
      this.options.tui.requestRender(true)
    }

    return refused
  }
}
