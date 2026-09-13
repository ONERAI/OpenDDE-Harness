import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { GlobalKeyActions, KeyEditor } from '../lib/keybindings.js'

import {
  createGlobalKeyHandler,
  createKeybindings,
  decideCtrlC,
  keyDisplayText,
  keyText,
  QUIT_ARM_MS
} from '../lib/keybindings.js'

const KEYS = {
  altG: '\x1bg',
  ctrlC: '\x03',
  ctrlD: '\x04',
  ctrlG: '\x07',
  ctrlK: '\x0b',
  ctrlL: '\x0c',
  ctrlO: '\x0f',
  ctrlT: '\x14',
  escape: '\x1b',
  letterA: 'a',
  shiftTab: '\x1b[Z'
}

function actions(
  clearPendingInput: () => void = () => {}
): GlobalKeyActions & Record<string, ReturnType<typeof vi.fn>> {
  return {
    cancelQueueEdit: vi.fn(() => false),
    cancelTurn: vi.fn(),
    clearPendingInput: vi.fn(clearPendingInput),
    copySelection: vi.fn(() => false),
    dropQueued: vi.fn(() => false),
    editQueued: vi.fn(() => false),
    forceReset: vi.fn(),
    openExternalEditor: vi.fn(),
    showQuitHint: vi.fn(),
    paste: vi.fn(),
    quit: vi.fn(),
    redraw: vi.fn(),
    submitQueueHead: vi.fn(),
    toggleThinking: vi.fn(),
    toggleTools: vi.fn(),
    toggleYolo: vi.fn()
  }
}

class FakeEditor implements KeyEditor {
  autocomplete = false

  constructor(private text = '') {}

  getText(): string {
    return this.text
  }

  isShowingAutocomplete(): boolean {
    return this.autocomplete
  }

  setText(text: string): void {
    this.text = text
  }
}

describe('decideCtrlC', () => {
  const idle = { escapeArmed: false, hasPendingInput: false, quitArmed: false, turnActive: false }

  it('cancels the first time during a turn and force-resets the second', () => {
    const busy = { ...idle, turnActive: true }

    expect(decideCtrlC(busy)).toBe('cancel-turn')
    expect(decideCtrlC({ ...busy, escapeArmed: true })).toBe('force-reset')
  })

  it('clears pending input before it offers to quit', () => {
    expect(decideCtrlC({ ...idle, hasPendingInput: true })).toBe('clear-input')
    expect(decideCtrlC({ ...idle, hasPendingInput: true, quitArmed: true })).toBe('clear-input')
  })

  it('offers to quit before it quits', () => {
    expect(decideCtrlC(idle)).toBe('arm-quit')
    expect(decideCtrlC({ ...idle, quitArmed: true })).toBe('quit')
  })

  it('keeps the cancel arm and the quit arm apart', () => {
    // A cancel that has not settled says nothing about exiting. Sharing one
    // flag between the two is how a cancel came to quit the app.
    expect(decideCtrlC({ ...idle, escapeArmed: true })).toBe('arm-quit')
    expect(decideCtrlC({ ...idle, quitArmed: true, turnActive: true })).toBe('cancel-turn')
  })
})

describe('keybinding matches', () => {
  const keybindings = createKeybindings()

  it.each([
    ['app.reset', KEYS.ctrlC],
    ['app.exit', KEYS.ctrlD],
    ['app.redraw', KEYS.ctrlL],
    ['app.thinking.toggle', KEYS.ctrlT],
    ['app.tools.expand', KEYS.ctrlO],
    ['app.yolo.toggle', KEYS.shiftTab],
    ['app.queue.submit', KEYS.ctrlK],
    ['app.interrupt', KEYS.escape],
    ['app.editor.external', KEYS.ctrlG],
    ['app.editor.external', KEYS.altG]
  ] as const)('%s matches its key', (binding, data) => {
    expect(keybindings.matches(data, binding)).toBe(true)
  })

  it('does not match an unrelated key', () => {
    expect(keybindings.matches(KEYS.letterA, 'app.reset')).toBe(false)
  })

  it('honours a user override', () => {
    const custom = createKeybindings({ 'app.redraw': 'ctrl+t' })

    expect(custom.matches(KEYS.ctrlT, 'app.redraw')).toBe(true)
    expect(custom.matches(KEYS.ctrlL, 'app.redraw')).toBe(false)
  })

  it('renders hints from the bound keys, lowercase beside a word and capitalised in a line of its own', () => {
    expect(keyText(keybindings, 'app.editor.external', 'linux')).toBe('ctrl+g/alt+g')
    expect(keyText(keybindings, 'app.editor.external', 'darwin')).toBe('ctrl+g/option+g')
    // pi's `keyDisplayText`: every part of every key, and option on macOS too.
    expect(keyDisplayText(keybindings, 'app.editor.external', 'linux')).toBe('Ctrl+G/Alt+G')
    expect(keyDisplayText(keybindings, 'app.editor.external', 'darwin')).toBe('Ctrl+G/Option+G')
    expect(keyDisplayText(keybindings, 'tui.select.confirm', 'linux')).toBe('Enter')
  })
})

describe('handleGlobalKey', () => {
  let act: ReturnType<typeof actions>
  let editor: FakeEditor
  let turnActive: boolean
  let queued: number
  let clock: number

  function handler() {
    return createGlobalKeyHandler({
      actions: act,
      editor,
      hasQueued: () => queued > 0,
      isTurnActive: () => turnActive,
      keybindings: createKeybindings(),
      now: () => clock
    })
  }

  beforeEach(() => {
    editor = new FakeEditor()
    turnActive = false
    queued = 0
    clock = 1_000_000
    // The app's `clearPendingInput` drops both; the fake does the same so the
    // ladder can be walked past this rung.
    act = actions(() => {
      editor.setText('')
      queued = 0
    })
  })

  it('walks the Ctrl+C ladder across presses', () => {
    turnActive = true

    const keys = handler()

    expect(keys.handleGlobalKey(KEYS.ctrlC)).toEqual({ consume: true })
    expect(act.cancelTurn).toHaveBeenCalledTimes(1)

    keys.handleGlobalKey(KEYS.ctrlC)
    expect(act.forceReset).toHaveBeenCalledTimes(1)

    turnActive = false
    keys.handleGlobalKey(KEYS.ctrlC)
    expect(act.quit).not.toHaveBeenCalled()
    expect(act.showQuitHint).toHaveBeenLastCalledWith(true)

    keys.handleGlobalKey(KEYS.ctrlC)
    expect(act.quit).toHaveBeenCalledTimes(1)
  })

  it('does not quit on one press at an idle empty prompt', () => {
    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).not.toHaveBeenCalled()
    expect(act.showQuitHint).toHaveBeenCalledWith(true)

    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).toHaveBeenCalledTimes(1)
  })

  it('does not quit on the press that follows a cancel', () => {
    // The owner's bug, exactly: press one cancels, the turn ends before press
    // two arrives, and press two used to find an idle empty prompt and exit.
    turnActive = true

    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlC)
    expect(act.cancelTurn).toHaveBeenCalledTimes(1)

    turnActive = false
    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).not.toHaveBeenCalled()
    expect(act.showQuitHint).toHaveBeenLastCalledWith(true)
  })

  it('lets the offer lapse', () => {
    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlC)
    clock += QUIT_ARM_MS

    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).not.toHaveBeenCalled()
    // Taken back before it was offered again, so the row never shows a stale one.
    expect(vi.mocked(act.showQuitHint).mock.calls.map(call => call[0])).toEqual([true, false, true])
  })

  it('quits on a second press just inside the window', () => {
    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlC)
    clock += QUIT_ARM_MS - 1
    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).toHaveBeenCalledTimes(1)
  })

  it('takes the offer back when another key is pressed', () => {
    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlC)
    keys.handleGlobalKey(KEYS.letterA)

    expect(act.showQuitHint).toHaveBeenLastCalledWith(false)

    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).not.toHaveBeenCalled()
  })

  it('takes the offer back when a turn starts', () => {
    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlC)
    expect(act.showQuitHint).toHaveBeenLastCalledWith(true)

    turnActive = true
    keys.handleGlobalKey(KEYS.ctrlC)

    // The press cancels the turn, and the offer made before it is gone.
    expect(act.cancelTurn).toHaveBeenCalledTimes(1)
    expect(act.quit).not.toHaveBeenCalled()
    expect(act.showQuitHint).toHaveBeenLastCalledWith(false)

    turnActive = false
    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).not.toHaveBeenCalled()
  })

  it('exposes retracting the offer separately from forgetting a cancel', () => {
    // Two lifecycle hooks because they are two states. The composed-app test
    // is the one that proves a real turn calls this one; this proves the hook
    // does what the app needs when it does.
    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlC)
    expect(act.showQuitHint).toHaveBeenLastCalledWith(true)

    keys.disarm()
    expect(act.showQuitHint).toHaveBeenLastCalledWith(true)

    keys.disarmQuit()
    expect(act.showQuitHint).toHaveBeenLastCalledWith(false)

    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).not.toHaveBeenCalled()
    expect(act.showQuitHint).toHaveBeenLastCalledWith(true)
  })

  it('takes the offer back when the press copied a selection instead', () => {
    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlC)
    expect(act.showQuitHint).toHaveBeenLastCalledWith(true)

    // This press copied. It did something, so it is not the second half of an
    // offer to exit.
    vi.mocked(act.copySelection).mockReturnValueOnce(true)
    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).not.toHaveBeenCalled()
    expect(act.showQuitHint).toHaveBeenLastCalledWith(false)

    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).not.toHaveBeenCalled()
  })

  it('offers again after the input it cleared', () => {
    editor.setText('half a prompt')

    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlC)
    expect(editor.getText()).toBe('')
    expect(act.quit).not.toHaveBeenCalled()

    keys.handleGlobalKey(KEYS.ctrlC)
    expect(act.quit).not.toHaveBeenCalled()
    expect(act.showQuitHint).toHaveBeenLastCalledWith(true)
  })

  it('still quits on one Ctrl+D, which is the explicit exit key', () => {
    handler().handleGlobalKey(KEYS.ctrlD)

    expect(act.quit).toHaveBeenCalledTimes(1)
  })

  it('clears the editor rather than quitting when it holds text', () => {
    editor.setText('half a prompt')

    handler().handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).not.toHaveBeenCalled()
    expect(editor.getText()).toBe('')
  })

  it('lets Ctrl+C close the completion popup before it touches the line', () => {
    editor.setText('/th')
    editor.autocomplete = true

    const keys = handler()

    expect(keys.handleGlobalKey(KEYS.ctrlC)).toBeUndefined()
    expect(editor.getText()).toBe('/th')

    editor.autocomplete = false
    keys.handleGlobalKey(KEYS.ctrlC)
    expect(editor.getText()).toBe('')
  })

  it('still cancels a running turn on Ctrl+C with the popup open', () => {
    turnActive = true
    editor.autocomplete = true

    handler().handleGlobalKey(KEYS.ctrlC)

    expect(act.cancelTurn).toHaveBeenCalledTimes(1)
  })

  it('counts a queued message as pending input', () => {
    queued = 1

    handler().handleGlobalKey(KEYS.ctrlC)

    expect(act.quit).not.toHaveBeenCalled()
  })

  it('clears the queue on an empty editor, so the next presses can quit', () => {
    queued = 1

    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.clearPendingInput).toHaveBeenCalledTimes(1)
    expect(queued).toBe(0)
    expect(act.quit).not.toHaveBeenCalled()

    // Clearing the queue only empties the prompt. Exiting from there still
    // takes the offer and the press that accepts it.
    keys.handleGlobalKey(KEYS.ctrlC)
    expect(act.quit).not.toHaveBeenCalled()
    expect(act.showQuitHint).toHaveBeenLastCalledWith(true)

    keys.handleGlobalKey(KEYS.ctrlC)
    expect(act.quit).toHaveBeenCalledTimes(1)
  })

  it('clears the draft and the queue together', () => {
    queued = 2
    editor.setText('half a prompt')

    handler().handleGlobalKey(KEYS.ctrlC)

    expect(editor.getText()).toBe('')
    expect(queued).toBe(0)
  })

  it('re-arms after a turn ends', () => {
    turnActive = true

    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlC)
    keys.disarm()
    keys.handleGlobalKey(KEYS.ctrlC)

    expect(act.cancelTurn).toHaveBeenCalledTimes(2)
    expect(act.forceReset).not.toHaveBeenCalled()
  })

  it('cancels a turn on escape but leaves the key alone otherwise', () => {
    const keys = handler()

    expect(keys.handleGlobalKey(KEYS.escape)).toBeUndefined()

    turnActive = true
    expect(keys.handleGlobalKey(KEYS.escape)).toEqual({ consume: true })
    expect(act.cancelTurn).toHaveBeenCalledTimes(1)

    editor.autocomplete = true
    expect(keys.handleGlobalKey(KEYS.escape)).toBeUndefined()
  })

  it('quits on Ctrl+D only while the editor is empty', () => {
    const keys = handler()

    editor.setText('x')
    expect(keys.handleGlobalKey(KEYS.ctrlD)).toBeUndefined()
    expect(act.quit).not.toHaveBeenCalled()

    editor.setText('')
    expect(keys.handleGlobalKey(KEYS.ctrlD)).toEqual({ consume: true })
    expect(act.quit).toHaveBeenCalledTimes(1)
  })

  it('claims Ctrl+K only when something is queued', () => {
    const keys = handler()

    expect(keys.handleGlobalKey(KEYS.ctrlK)).toBeUndefined()

    queued = 2
    expect(keys.handleGlobalKey(KEYS.ctrlK)).toEqual({ consume: true })
    expect(act.submitQueueHead).toHaveBeenCalledTimes(1)
  })

  it('leaves Shift+Tab to the completion popup while it is open', () => {
    const keys = handler()

    editor.autocomplete = true
    expect(keys.handleGlobalKey(KEYS.shiftTab)).toBeUndefined()

    editor.autocomplete = false
    expect(keys.handleGlobalKey(KEYS.shiftTab)).toEqual({ consume: true })
    expect(act.toggleYolo).toHaveBeenCalledTimes(1)
  })

  it('acts on a key press once when the terminal also reports its release', () => {
    // A terminal on the kitty keyboard protocol (pi-tui asks for flag 2,
    // report event types) sends ctrl+o as a press and then a release. The TUI
    // drops releases before the focused component, but the global listener
    // runs ahead of that filter, so it has to drop them itself: expanding on
    // the press and collapsing on the release reads as a key that did nothing.
    const keys = handler()

    keys.handleGlobalKey('\x1b[111;5u')
    expect(keys.handleGlobalKey('\x1b[111;5:3u')).toBeUndefined()
    keys.handleGlobalKey('\x1b[116;5u')
    keys.handleGlobalKey('\x1b[116;5:3u')

    expect(act.toggleTools).toHaveBeenCalledTimes(1)
    expect(act.toggleThinking).toHaveBeenCalledTimes(1)
  })

  it('routes the remaining app keys to their actions', () => {
    const keys = handler()

    keys.handleGlobalKey(KEYS.ctrlT)
    keys.handleGlobalKey(KEYS.ctrlO)
    keys.handleGlobalKey(KEYS.ctrlL)
    keys.handleGlobalKey(KEYS.ctrlG)

    expect(act.toggleThinking).toHaveBeenCalledTimes(1)
    expect(act.toggleTools).toHaveBeenCalledTimes(1)
    expect(act.redraw).toHaveBeenCalledTimes(1)
    expect(act.openExternalEditor).toHaveBeenCalledTimes(1)
  })

  it('passes ordinary characters through', () => {
    expect(handler().handleGlobalKey(KEYS.letterA)).toBeUndefined()
  })
})
