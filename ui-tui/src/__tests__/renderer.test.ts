// The renderer owner: one terminal, two screens, and the same components.

import { Container, Text, TuiAltScreen, TuiMainScreen } from '@earendil-works/pi-tui'
import { describe, expect, it } from 'vitest'

import { createChatViewport } from '../chatViewport.js'
import { createTuiReference, initialRendererMode, mouseEnabled, RendererOwner, scrollLines } from '../renderer.js'
import { Theme } from '../theme.js'
import { FakeTerminal } from './fakes.js'

const theme = new Theme('dark', 0)

/** A terminal that can refuse to start once, standing in for a renderer that
 *  cannot take the screen. */
class FlakyTerminal extends FakeTerminal {
  failNextStart = false

  override start(onInput: (data: string) => void, onResize: () => void): void {
    if (this.failNextStart) {
      this.failNextStart = false

      throw new Error('the terminal refused to start')
    }

    super.start(onInput, onResize)
  }
}

function build(mode: 'fullscreen' | 'regular' = 'regular') {
  const terminal = new FakeTerminal(80, 24)
  const owner = new RendererOwner({ env: {}, mode, terminal, theme })

  return { owner, terminal }
}

/** The alternate screen is entered and left with these. */
const ENTER_ALT = '\x1b[?1049h'
const LEAVE_ALT = '\x1b[?1049l'

const count = (haystack: string, needle: string) => haystack.split(needle).length - 1

describe('choosing a renderer', () => {
  it('starts in fullscreen unless the environment asks for the main screen', () => {
    expect(initialRendererMode({})).toBe('fullscreen')
    expect(initialRendererMode({ OPENDDE_HARNESS_TUI_FULLSCREEN: '1' })).toBe('fullscreen')
    expect(initialRendererMode({ OPENDDE_HARNESS_TUI_FULLSCREEN: 'yes' })).toBe('fullscreen')
    expect(initialRendererMode({ OPENDDE_HARNESS_TUI_FULLSCREEN: '' })).toBe('fullscreen')

    for (const off of ['0', 'false', 'FALSE', 'no', 'off', ' Off ']) {
      expect(initialRendererMode({ OPENDDE_HARNESS_TUI_FULLSCREEN: off })).toBe('regular')
    }
  })

  it('reads the mouse and wheel settings, clamping the wheel', () => {
    expect(mouseEnabled({})).toBe(true)
    expect(mouseEnabled({ OPENDDE_HARNESS_TUI_DISABLE_MOUSE: '1' })).toBe(false)

    expect(scrollLines({})).toBe(3)
    expect(scrollLines({ OPENDDE_HARNESS_TUI_SCROLL_SPEED: '7.4' })).toBe(7)
    expect(scrollLines({ OPENDDE_HARNESS_TUI_SCROLL_SPEED: '900' })).toBe(20)
    expect(scrollLines({ OPENDDE_HARNESS_TUI_SCROLL_SPEED: '-2' })).toBe(3)
    expect(scrollLines({ OPENDDE_HARNESS_TUI_SCROLL_SPEED: 'lots' })).toBe(3)
    // The old name still works when the current one is absent.
    expect(scrollLines({ CLAUDE_CODE_SCROLL_SPEED: '5' })).toBe(5)
  })
})

describe('switching renderers', () => {
  it('keeps every component object across the switch and back', () => {
    const { owner, terminal } = build()
    const first = new Text('one', 0, 0)
    const second = new Text('two', 0, 0)

    owner.reference.addChild(first)
    owner.reference.addChild(second)
    owner.reference.start()

    expect(owner.switchTo('fullscreen')).toBeNull()
    expect(owner.mode).toBe('fullscreen')
    expect(owner.renderer).toBeInstanceOf(TuiAltScreen)
    expect(owner.reference.children).toEqual([first, second])
    expect(terminal.output()).toContain(ENTER_ALT)

    expect(owner.switchTo('regular')).toBeNull()
    expect(owner.reference.children).toEqual([first, second])
    expect(owner.renderer).toBeInstanceOf(TuiMainScreen)
    expect(terminal.output()).toContain(LEAVE_ALT)

    owner.dispose()
  })

  it('restores the focused component', () => {
    const { owner } = build()
    const focusable = Object.assign(new Text('editor', 0, 0), { focused: false })

    owner.reference.addChild(focusable)
    owner.reference.setFocus(focusable)
    owner.reference.start()

    owner.switchTo('fullscreen')
    expect(owner.renderer.getFocusedComponent()).toBe(focusable)

    owner.dispose()
  })

  it('does nothing when it is already in that mode', () => {
    const { owner, terminal } = build()

    owner.reference.start()

    const before = terminal.writes.length

    expect(owner.switchTo('regular')).toBeNull()
    expect(terminal.writes.length).toBe(before)

    owner.dispose()
  })

  it('refuses while an overlay is open', () => {
    const { owner } = build()

    owner.reference.start()
    owner.reference.showOverlay(new Text('overlay', 0, 0))

    expect(owner.switchTo('fullscreen')).toContain('overlay')
    expect(owner.mode).toBe('regular')

    owner.dispose()
  })

  it('hands the fullscreen layout the same objects the dock uses', () => {
    const { owner } = build()
    const document = new Text('doc', 0, 0)
    const editor = new Text('editor', 0, 0)
    const viewport = createChatViewport({
      document,
      editor,
      footer: new Text('footer', 0, 0),
      queue: new Text('queue', 0, 0),
      status: new Text('', 0, 0),
      taskBar: new Text('bar', 0, 0),
      theme
    })

    owner.setLayoutRoot(viewport.root)
    owner.reference.addChild(document)
    owner.reference.addChild(editor)
    owner.reference.start()
    owner.switchTo('fullscreen')

    // The scroll view survives the switch, so scroll position and follow-end
    // survive an off/on cycle rather than being rebuilt at the bottom.
    expect(owner.layoutRoot).toBe(viewport.root)
    owner.switchTo('regular')
    expect(owner.layoutRoot).toBe(viewport.root)

    owner.dispose()
  })

  it('rolls back to the renderer it had when the new one cannot start', () => {
    const terminal = new FlakyTerminal()
    const owner = new RendererOwner({ env: {}, mode: 'regular', terminal, theme })
    const child = new Text('kept', 0, 0)

    owner.reference.addChild(child)
    owner.reference.setFocus(child)
    owner.reference.start()

    const before = owner.renderer

    terminal.failNextStart = true

    const refused = owner.switchTo('fullscreen')

    expect(refused).toContain('terminal refused to start')
    // Terminal ownership may never be ambiguous: the old renderer is back,
    // with its tree and its focus.
    expect(owner.renderer).toBe(before)
    expect(owner.mode).toBe('regular')
    expect(owner.reference.children).toEqual([child])
    expect(owner.renderer.getFocusedComponent()).toBe(child)

    owner.dispose()
  })

  it('leaves the alternate screen again when the new renderer cannot start', () => {
    const terminal = new FlakyTerminal()
    const owner = new RendererOwner({ env: {}, mode: 'regular', terminal, theme })

    owner.reference.addChild(new Text('kept', 0, 0))
    owner.reference.start()

    terminal.failNextStart = true
    expect(owner.switchTo('fullscreen')).toContain('terminal refused to start')

    // Pi enters the alternate screen, turns mouse reporting on and hides the
    // cursor *before* it calls `Terminal.start`, so the physical screen is in
    // fullscreen by the time the start throws. Rolling the object tree back
    // and leaving it there is how the UI ends up drawing regular-mode output
    // into the alternate buffer.
    const written = terminal.output()

    expect(count(written, ENTER_ALT)).toBe(1)
    expect(count(written, LEAVE_ALT)).toBe(1)
    expect(owner.mode).toBe('regular')

    // And the screen is usable afterwards: the next switch works.
    expect(owner.switchTo('fullscreen')).toBeNull()
    expect(count(terminal.output(), ENTER_ALT)).toBe(2)

    owner.switchTo('regular')
    expect(count(terminal.output(), LEAVE_ALT)).toBe(2)

    owner.dispose()
  })

  it('leaves the alternate screen again when the return to regular cannot start', () => {
    const terminal = new FlakyTerminal()
    const owner = new RendererOwner({ env: {}, mode: 'fullscreen', terminal, theme })

    owner.reference.addChild(new Text('kept', 0, 0))
    owner.reference.start()
    expect(count(terminal.output(), ENTER_ALT)).toBe(1)

    terminal.failNextStart = true
    expect(owner.switchTo('regular')).toContain('terminal refused to start')

    // The fullscreen renderer was stopped on the way out, so the alternate
    // screen has been left whichever direction the failure happened in.
    expect(owner.mode).toBe('fullscreen')
    expect(count(terminal.output(), LEAVE_ALT)).toBe(1)

    owner.dispose()
  })
})

describe('leaving for good', () => {
  /** A layout with a transcript taller than the terminal, so the difference
   *  between replaying a frame and replaying the conversation is visible. */
  function conversation(rows: number) {
    const terminal = new FakeTerminal(40, 10)
    const chat = new Container()

    for (let index = 0; index < rows; index += 1) {
      chat.addChild(new Text(`line ${index}`, 0, 0))
    }

    const document = new Container()

    document.addChild(chat)

    const owner = new RendererOwner({ env: {}, mode: 'fullscreen', terminal, theme })
    const viewport = createChatViewport({
      document,
      editor: new Text('editor', 0, 0),
      footer: new Text('footer', 0, 0),
      queue: new Text('', 0, 0),
      status: new Text('', 0, 0),
      taskBar: new Text('', 0, 0),
      theme
    })

    owner.setLayoutRoot(viewport.root)
    owner.reference.addChild(document)
    owner.reference.start()
    owner.reference.renderNow()
    terminal.writes.length = 0

    return { owner, terminal }
  }

  it('writes the whole conversation into the main buffer, not just the last frame', () => {
    const { owner, terminal } = conversation(30)

    owner.dispose()

    const written = terminal.output()

    // Stopping the alternate screen with `preserveScreen: false` replays one
    // viewport — ten rows here. The conversation is thirty.
    expect(written).toContain(LEAVE_ALT)
    expect(written).toContain('line 0')
    expect(written).toContain('line 15')
    expect(written).toContain('line 29')
  })

  it('does it once, and leaves the mode at regular', () => {
    const { owner, terminal } = conversation(4)

    owner.dispose()

    expect(owner.mode).toBe('regular')
    expect(count(terminal.output(), LEAVE_ALT)).toBe(1)

    // A second pass — the app quits and then the signal cleanup runs — must
    // not replay the conversation again.
    const after = terminal.output().length

    owner.dispose()
    expect(terminal.output().slice(after)).not.toContain('line 0')
  })

  it('closes the search overlay rather than being refused by it', () => {
    const { owner, terminal } = conversation(4)

    owner.reference.showOverlay(new Text('search', 0, 0))
    owner.dispose()

    expect(owner.mode).toBe('regular')
    expect(terminal.output()).toContain('line 0')
  })

  it('is a no-op on the main screen', () => {
    const { owner } = build()

    owner.reference.addChild(new Text('kept', 0, 0))
    owner.reference.start()

    expect(owner.exitToMainScreen()).toBe(false)
    owner.dispose()
  })
})

describe('the stable reference', () => {
  it('sends a method captured before the switch to the renderer after it', () => {
    const { owner } = build()
    // Pulled out once and held, the way a component does at construction.
    const addChild = owner.reference.addChild
    const late = new Text('added after the switch', 0, 0)

    owner.reference.start()
    owner.switchTo('fullscreen')
    addChild(late)

    expect(owner.renderer.children).toContain(late)

    owner.dispose()
  })

  it('forwards property reads and writes to whatever is current', () => {
    let current = new TuiMainScreen(new FakeTerminal(80, 24))
    const reference = createTuiReference(() => current)

    expect(reference.mode).toBe('regular')

    current = new TuiAltScreen(new FakeTerminal(80, 24)) as unknown as TuiMainScreen
    expect(reference.mode).toBe('fullscreen')
  })
})
