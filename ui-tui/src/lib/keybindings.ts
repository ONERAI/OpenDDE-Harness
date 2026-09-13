// Key bindings: pi-tui's editor/selection set plus an `app.*` namespace for the
// things this app does, in pi's shape (`packages/coding-agent/src/core/keybindings.ts`).
//
// `handleGlobalKey` is the listener the app wires into `tui.addInputListener`.
// Input listeners run before the focused component, so anything this consumes
// never reaches the editor — which is why several bindings consume only under a
// condition (Ctrl+D only on an empty editor, Ctrl+K only with something queued)
// and otherwise fall through to the editor's readline behaviour.

import type {
  Keybinding,
  KeybindingDefinitions,
  KeybindingsConfig,
  TuiInputListenerResult
} from '@earendil-works/pi-tui'

import { isKeyRelease, KeybindingsManager, setKeybindings, TUI_KEYBINDINGS } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

export interface AppKeybindings {
  'app.editor.external': true
  'app.exit': true
  'app.interrupt': true
  'app.models.clearAll': true
  'app.models.enableAll': true
  'app.models.reorderDown': true
  'app.models.reorderUp': true
  'app.models.save': true
  'app.models.toggleProvider': true
  'app.paste': true
  'app.queue.drop': true
  'app.queue.edit.newer': true
  'app.queue.edit.older': true
  'app.queue.submit': true
  'app.redraw': true
  'app.reset': true
  'app.session.delete': true
  'app.thinking.toggle': true
  'app.tools.expand': true
  'app.yolo.toggle': true
}

export type AppKeybinding = keyof AppKeybindings

declare module '@earendil-works/pi-tui' {
  // Widen pi-tui's global registry so `matches(data, 'app.…')` type-checks.
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  interface Keybindings extends AppKeybindings {}
}

export const KEYBINDINGS = {
  ...TUI_KEYBINDINGS,
  // Escape is pi's cancel key. Ctrl+C is the ladder below, not an exit key.
  'app.interrupt': { defaultKeys: 'escape', description: 'Cancel the running turn' },
  'app.reset': { defaultKeys: 'ctrl+c', description: 'Cancel, clear the input, or quit' },
  'app.exit': { defaultKeys: 'ctrl+d', description: 'Quit when the editor is empty' },
  'app.redraw': { defaultKeys: 'ctrl+l', description: 'Repaint the screen' },
  'app.thinking.toggle': { defaultKeys: 'ctrl+t', description: 'Show or hide thinking' },
  'app.tools.expand': { defaultKeys: 'ctrl+o', description: 'Expand or collapse tool output' },
  'app.yolo.toggle': { defaultKeys: 'shift+tab', description: 'Toggle yolo (skip approvals)' },
  'app.queue.submit': { defaultKeys: 'ctrl+k', description: 'Send the first queued message now' },
  // Alt+Up/Down: pi's editor owns the bare arrows for the cursor and the
  // prompt history, so queue editing takes the modified pair.
  'app.queue.edit.older': { defaultKeys: 'alt+up', description: 'Edit an earlier queued message' },
  'app.queue.edit.newer': { defaultKeys: 'alt+down', description: 'Edit a later queued message' },
  'app.queue.drop': { defaultKeys: 'alt+x', description: 'Drop the queued message being edited' },
  // Terminal-native paste still works; this is the path for a clipboard that
  // lives in the terminal emulator rather than on this host (ssh).
  'app.paste': { defaultKeys: 'ctrl+v', description: 'Paste from the clipboard' },
  // Ctrl+G is the primary; VS Code and its forks bind it to "Find Next" before
  // the TUI sees it, and Alt+G arrives intact on every platform.
  'app.editor.external': { defaultKeys: ['ctrl+g', 'alt+g'], description: 'Edit the prompt in $VISUAL/$EDITOR' },
  // Selector-stage actions. Only the stage that lists one in its hints reads it;
  // outside that stage the same bytes are search text or an editor binding, so
  // these deliberately share keys with the editor set.
  // pi's own keys for its model selectors, with pi's own names.
  'app.models.save': { defaultKeys: 'ctrl+s', description: 'Save model selection' },
  'app.models.enableAll': { defaultKeys: 'ctrl+a', description: 'Enable all models' },
  'app.models.clearAll': { defaultKeys: 'ctrl+x', description: 'Clear all models' },
  'app.models.toggleProvider': { defaultKeys: 'ctrl+p', description: 'Toggle all models for provider' },
  'app.models.reorderUp': { defaultKeys: 'alt+up', description: 'Move model up in order' },
  'app.models.reorderDown': { defaultKeys: 'alt+down', description: 'Move model down in order' },
  'app.session.delete': { defaultKeys: 'ctrl+d', description: 'Delete the selected session' }
} as const satisfies KeybindingDefinitions

/**
 * pi-tui's bindings plus ours, with optional user overrides.
 *
 * The manager is also installed as pi-tui's global: stock widgets (`Input`,
 * `SelectList`, `SettingsList`) read `getKeybindings()` rather than an injected
 * manager, so a private manager would leave the selectors' displayed hints
 * disagreeing with what the widgets actually do.
 */
export function createKeybindings(userBindings: KeybindingsConfig = {}): KeybindingsManager {
  const manager = new KeybindingsManager(KEYBINDINGS, userBindings)

  setKeybindings(manager)

  return manager
}

/** One key part as it is shown: macOS calls the alt key option, so say option
 *  there, and a hint that names its keys as words capitalises them. */
function keyPart(part: string, platform: NodeJS.Platform, capitalize: boolean): string {
  const shown = platform === 'darwin' && part.toLowerCase() === 'alt' ? 'option' : part
  return capitalize ? shown.charAt(0).toUpperCase() + shown.slice(1) : shown
}

function formatKeys(
  keybindings: KeybindingsManager,
  binding: Keybinding,
  platform: NodeJS.Platform,
  capitalize: boolean
): string {
  return keybindings
    .getKeys(binding)
    .map(key =>
      key
        .split('+')
        .map(part => keyPart(part, platform, capitalize))
        .join('+')
    )
    .join('/')
}

/** The keys bound to an action, as `ctrl+g/alt+g`. pi's `keyText`: what a
 *  `<keys> <what it does>` hint reads, where the keys are chrome beside the
 *  words and stay lowercase. */
export function keyText(
  keybindings: KeybindingsManager,
  binding: Keybinding,
  platform: NodeJS.Platform = process.platform
): string {
  return formatKeys(keybindings, binding, platform, false)
}

/** The same keys as `Ctrl+G/Alt+G`. pi's `keyDisplayText`: what a hint *line*
 *  reads, where each key opens its own clause ("Ctrl+S to save") and is read as
 *  a word rather than as chrome. pi picks between the two per line, so this
 *  file offers both and each line says which it is. */
export function keyDisplayText(
  keybindings: KeybindingsManager,
  binding: Keybinding,
  platform: NodeJS.Platform = process.platform
): string {
  return formatKeys(keybindings, binding, platform, true)
}

/** `<keys> <what it does>`, dim then muted, as pi renders selector hints.
 *
 *  Both greys: a hint is chrome, and chrome carries no hue. */
export function keyHint(
  keybindings: KeybindingsManager,
  theme: Theme,
  binding: Keybinding,
  description: string
): string {
  return `${theme.fg('dim', keyText(keybindings, binding))} ${theme.fg('muted', description)}`
}

/** The same shape for keys no binding owns, such as the arrow pair. */
export function rawKeyHint(theme: Theme, keys: string, description: string): string {
  return `${theme.fg('dim', keys)} ${theme.fg('muted', description)}`
}

/** Join hints with the spacing pi uses between them. */
export function joinHints(hints: string[]): string {
  return hints.filter(Boolean).join('   ')
}

export type CtrlCAction = 'arm-quit' | 'cancel-turn' | 'clear-input' | 'force-reset' | 'quit'

/** How long an armed quit stays armed. Claude Code's window, near enough: long
 *  enough for a deliberate second press, short enough that a Ctrl+C minutes
 *  later is a fresh intention rather than the other half of an old one. */
export const QUIT_ARM_MS = 2000

/** What the status row says while the quit is armed. */
export const QUIT_HINT = 'press ctrl+c again to exit'

export interface CtrlCState {
  /** A previous Ctrl+C already asked the server to cancel this turn. This is
   *  about the turn, and has nothing to do with `quitArmed`: a cancel that has
   *  not settled and an offer to exit are different states, and sharing one
   *  flag between them is how a cancel came to quit the app. */
  escapeArmed: boolean
  /** The editor holds text, or a message is queued. */
  hasPendingInput: boolean
  /** A previous Ctrl+C offered to exit, recently enough to still count. */
  quitArmed: boolean
  /** A turn is in flight. */
  turnActive: boolean
}

/**
 * Which rung of the Ctrl+C ladder a keypress lands on.
 *
 * Ctrl+C is not an exit key: it quits only from an idle UI with an empty
 * editor. Reading it as "press once to exit" is what made the old dogfood e2e
 * suite flaky (#228), so the decision lives here where it is testable without a
 * pty or a live turn.
 *
 * - `cancel-turn`: first Ctrl+C during a turn. Asks the server to cancel; the
 *   resulting `error(reason=cancelled_by_client)` event resets the UI.
 * - `force-reset`: second Ctrl+C in the same turn, i.e. the cancel produced no
 *   terminal event (events lost, server wedged). Resets locally so the user is
 *   never stuck waiting for a response that cannot arrive.
 * - `clear-input`: idle, but there is text or a queued line to drop.
 * - `arm-quit`: idle and empty, and nothing has offered to exit yet. Says so
 *   and waits.
 * - `quit`: idle, empty, and the offer is still standing.
 *
 * Quit is two presses because one press used to be enough, which is not what
 * Ctrl+C means to anyone. It was reachable two ways: an idle press with an
 * empty editor went straight through, and so did the ordinary cancel — press
 * one cancelled the turn, the turn ended, and press two arrived to find no
 * turn running and an empty editor. Both now land on `arm-quit` first, and
 * `quit` is unreachable without the hint having been shown.
 *
 * The old app had another rung, `interrupt-legacy`, for `session.interrupt`.
 * That method does not exist on the gateway; `turn.cancel` is the only path.
 */
export function decideCtrlC(state: CtrlCState): CtrlCAction {
  if (state.turnActive) {
    return state.escapeArmed ? 'force-reset' : 'cancel-turn'
  }

  if (state.hasPendingInput) {
    return 'clear-input'
  }

  return state.quitArmed ? 'quit' : 'arm-quit'
}

/** The editor surface the key handler needs; the real `Editor` satisfies it. */
export interface KeyEditor {
  getText(): string
  isShowingAutocomplete(): boolean
}

export interface GlobalKeyActions {
  /** Ask the server to cancel the turn in flight. */
  cancelTurn(): void
  /** Leave a queue edit without changing the queued message. Returns whether
   *  one was open — Escape only stops meaning "cancel the turn" when it was. */
  cancelQueueEdit(): boolean
  /** Copy the fullscreen selection, if there is one. Returns whether it did:
   *  a selection outranks the Ctrl+C ladder, and nothing else does. */
  copySelection(): boolean
  /** Pull a queued message into the editor. Returns whether one was there. */
  editQueued(direction: -1 | 1): boolean
  /** Drop the queued message being edited. Returns whether one was open. */
  dropQueued(): boolean
  /** Read the clipboard into the editor. */
  paste(): void
  /** Drop the editor draft *and* the queued messages. Ctrl+C's idle rung
   *  reaches both, so a queue the user asked to clear actually clears. */
  clearPendingInput(): void
  /** Drop the turn locally after a cancel produced no terminal event. */
  forceReset(): void
  /** Show or hide the standing offer to exit. Shown when a quit is armed and
   *  hidden the moment it is not, so the row never promises a second press
   *  that would no longer be taken. */
  showQuitHint(show: boolean): void
  openExternalEditor(): void
  quit(): void
  redraw(): void
  /** Send the head of the queue now. Only called when something is queued. */
  submitQueueHead(): void
  toggleThinking(): void
  toggleTools(): void
  toggleYolo(): void
}

export interface GlobalKeyOptions {
  actions: GlobalKeyActions
  editor: KeyEditor
  /** Something is waiting in the message queue. */
  hasQueued: () => boolean
  /** A selector or a broker prompt owns the editor slot. While one does, the
   *  focused component is the only thing that reads input; this listener steps
   *  aside so Esc/Ctrl+C/Ctrl+D/Tab/digits cannot also act on the app behind
   *  it. Ctrl+L stays, because a repaint is never ambiguous. */
  isSlotActive?: () => boolean
  keybindings: KeybindingsManager
  isTurnActive: () => boolean
  /** Reads the clock for the quit-arm window. Tests drive it by hand. */
  now?: () => number
}

export interface GlobalKeyHandler {
  /** Forget that a cancel is already in flight. Call when a turn ends. */
  disarm(): void
  /**
   * Take the standing offer to exit back.
   *
   * Called by whatever supersedes it: a turn starting, a command starting.
   * Separate from {@link disarm}, which is about a cancel in flight — sharing
   * one lifecycle hook between the two is how a cancel came to quit the app,
   * and sharing one flag would be the same mistake again. The window alone is
   * not enough: a turn that starts and finishes between two presses leaves the
   * offer standing and the second press takes it.
   */
  disarmQuit(): void
  /** The `tui.addInputListener` callback. */
  handleGlobalKey(data: string): TuiInputListenerResult
}

const CONSUME: TuiInputListenerResult = { consume: true }

/** The app-level key handler, as an object so the escape-armed flag has a home. */
export function createGlobalKeyHandler(opts: GlobalKeyOptions): GlobalKeyHandler {
  const { actions, editor, hasQueued, isSlotActive, isTurnActive, keybindings } = opts
  const now = opts.now ?? (() => Date.now())
  let escapeArmed = false
  /** When the standing offer to exit was made, or null if there is none. A
   *  timestamp rather than a flag, so the window closes on its own without a
   *  timer to own and dispose. */
  let quitArmedAt: null | number = null

  const disarmQuit = () => {
    if (quitArmedAt === null) {
      return
    }

    quitArmedAt = null
    actions.showQuitHint(false)
  }

  /** Whether the offer still stands. A turn starting takes it back — the first
   *  press then means cancel, and an offer made before it is stale. */
  const quitArmed = (): boolean => {
    if (quitArmedAt === null) {
      return false
    }

    if (isTurnActive() || now() - quitArmedAt >= QUIT_ARM_MS) {
      disarmQuit()

      return false
    }

    return true
  }

  const disarm = () => {
    escapeArmed = false
  }

  const handleCtrlC = (): TuiInputListenerResult => {
    // A fullscreen selection is what Ctrl+C means everywhere else, so it wins
    // before the cancel/clear/quit ladder is consulted at all. It also spends
    // the press: this one copied, so it is not the second half of an offer to
    // exit, and leaving the offer up would let the next press quit on the
    // strength of a press that did something else entirely.
    if (actions.copySelection()) {
      disarmQuit()

      return CONSUME
    }

    const action = decideCtrlC({
      escapeArmed,
      hasPendingInput: editor.getText().length > 0 || hasQueued(),
      quitArmed: quitArmed(),
      turnActive: isTurnActive()
    })

    switch (action) {
      case 'arm-quit':
        quitArmedAt = now()
        actions.showQuitHint(true)
        break

      case 'cancel-turn':
        escapeArmed = true
        actions.cancelTurn()
        break

      case 'clear-input':
        disarmQuit()
        actions.clearPendingInput()
        break

      case 'force-reset':
        escapeArmed = false
        actions.forceReset()
        break

      case 'quit':
        disarmQuit()
        actions.quit()
        break
    }

    return CONSUME
  }

  const handleGlobalKey = (data: string): TuiInputListenerResult => {
    // A terminal on the kitty keyboard protocol reports the release of every
    // key as well as its press. The TUI drops releases before the focused
    // component sees them, but this listener runs ahead of that filter, so a
    // release here would act a second time: ctrl+o expanding on the press and
    // collapsing on the release reads as a key that did nothing.
    if (isKeyRelease(data)) {
      return undefined
    }

    // Anything that is not Ctrl+C takes the offer back. The second press has
    // to be the next thing the user does, not the next Ctrl+C whenever it
    // happens to come.
    if (!keybindings.matches(data, 'app.reset')) {
      disarmQuit()
    }

    if (isSlotActive?.()) {
      if (keybindings.matches(data, 'app.redraw')) {
        actions.redraw()

        return CONSUME
      }

      // Everything else belongs to the selector or prompt that has focus.
      return undefined
    }

    if (keybindings.matches(data, 'app.editor.external')) {
      actions.openExternalEditor()

      return CONSUME
    }

    if (keybindings.matches(data, 'app.interrupt')) {
      // While the completion popup is open, escape closes it — that is the
      // editor's job, so leave the key alone.
      if (editor.isShowingAutocomplete()) {
        return undefined
      }

      // A queue edit is something the user opened deliberately, and the hint
      // under the queue promises Escape leaves it. Cancelling the turn is the
      // next press.
      if (actions.cancelQueueEdit()) {
        return CONSUME
      }

      if (!isTurnActive()) {
        return undefined
      }

      escapeArmed = true
      actions.cancelTurn()

      return CONSUME
    }

    if (keybindings.matches(data, 'app.reset')) {
      // Ctrl+C is also pi's select-cancel key, so with the completion popup
      // open it closes the popup rather than clearing the line. A running turn
      // outranks that: cancelling it is what the user means.
      if (editor.isShowingAutocomplete() && !isTurnActive()) {
        return undefined
      }

      return handleCtrlC()
    }

    // Ctrl+D quits an empty editor and deletes forward otherwise, like pi.
    if (keybindings.matches(data, 'app.exit')) {
      if (editor.getText().length > 0) {
        return undefined
      }

      actions.quit()

      return CONSUME
    }

    if (keybindings.matches(data, 'app.redraw')) {
      actions.redraw()

      return CONSUME
    }

    if (keybindings.matches(data, 'app.thinking.toggle')) {
      actions.toggleThinking()

      return CONSUME
    }

    if (keybindings.matches(data, 'app.tools.expand')) {
      actions.toggleTools()

      return CONSUME
    }

    if (keybindings.matches(data, 'app.yolo.toggle')) {
      // Shift+Tab is also the completion popup's back-cycle key.
      if (editor.isShowingAutocomplete()) {
        return undefined
      }

      actions.toggleYolo()

      return CONSUME
    }

    // Ctrl+K is readline's kill-to-end-of-line. Claim it only when there is a
    // queued message to send, so an empty queue leaves the editor binding alone.
    if (keybindings.matches(data, 'app.queue.submit') && hasQueued()) {
      actions.submitQueueHead()

      return CONSUME
    }

    // The queue-editing trio claims its keys only when there is something to
    // edit, so an empty queue leaves them to the editor.
    if (keybindings.matches(data, 'app.queue.edit.older')) {
      return actions.editQueued(-1) ? CONSUME : undefined
    }

    if (keybindings.matches(data, 'app.queue.edit.newer')) {
      return actions.editQueued(1) ? CONSUME : undefined
    }

    if (keybindings.matches(data, 'app.queue.drop')) {
      return actions.dropQueued() ? CONSUME : undefined
    }

    if (keybindings.matches(data, 'app.paste')) {
      actions.paste()

      return CONSUME
    }

    return undefined
  }

  return { disarm, disarmQuit, handleGlobalKey }
}
