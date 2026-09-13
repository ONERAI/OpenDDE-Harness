// Editing a message that is already queued, and reading the clipboard out of
// the terminal itself.

import type { spawn as SpawnFn } from 'node:child_process'

import { stripTerminalSequences, TuiMainScreen } from '@earendil-works/pi-tui'
import { EventEmitter } from 'node:events'
import { describe, expect, it } from 'vitest'

import { HarnessEditor } from '../components/editor.js'
import { MessageQueue } from '../components/queue.js'
import { readClipboardText } from '../lib/clipboard.js'
import { osc52ReadQuery, parseOsc52Reply, readOsc52Clipboard, stripOsc52Reply } from '../lib/osc52Read.js'
import { QueueEditing } from '../queueEditing.js'
import { Theme } from '../theme.js'
import { FakeTerminal } from './fakes.js'

const theme = new Theme('dark', 0)

/** Bracketed paste, as the terminal sends it. */
const paste = (text: string) => `\x1b[200~${text}\x1b[201~`

/** The real editor, not a string box: the paste table is the whole point of
 *  the draft-restoration cases below and a fake does not have one. */
function build(items: string[] = []) {
  const sent: string[] = []
  const released: number[] = []
  const turn = { active: true }
  const queue = new MessageQueue({
    actions: { interrupt: () => {}, send: text => void sent.push(text) },
    isTurnActive: () => turn.active,
    theme
  })

  for (const item of items) {
    queue.submit(item)
  }

  const tui = new TuiMainScreen(new FakeTerminal(100, 30))
  const editor = new HarnessEditor(tui, theme.editorTheme())
  const editing = new QueueEditing({
    editor,
    // What the app wires this to: the queue may have been holding for the
    // message the edit had reserved.
    onReleased: () => {
      released.push(queue.length)
      queue.drain()
    },
    queue
  })

  return { editing, editor, queue, released, sent, turn }
}

const view = (queue: MessageQueue) => queue.view.render(80).map(line => stripTerminalSequences(line).trim())

describe('editing a queued message', () => {
  it('does nothing when the queue is empty', () => {
    const { editing, editor } = build()

    editor.setText('a draft')
    expect(editing.cycle(-1)).toBe(false)
    expect(editor.getText()).toBe('a draft')
  })

  it('pulls the newest queued message into the editor and keeps the draft', () => {
    const { editing, editor, queue } = build(['first', 'second'])

    editor.setText('half-typed')
    expect(editing.cycle(-1)).toBe(true)

    expect(editor.getText()).toBe('second')
    expect(queue.editing).toBe(1)
    expect(view(queue).some(line => line.startsWith('Editing: second'))).toBe(true)

    expect(editing.cancel()).toBe(true)
    expect(editor.getText()).toBe('half-typed')
    expect(queue.editing).toBeNull()
    expect(queue.list()).toEqual(['first', 'second'])
  })

  it('wraps through the queue in both directions', () => {
    const { editing, editor } = build(['a', 'b', 'c'])

    editing.cycle(-1)
    expect(editor.getText()).toBe('c')
    editing.cycle(-1)
    expect(editor.getText()).toBe('b')
    editing.cycle(1)
    expect(editor.getText()).toBe('c')
    editing.cycle(1)
    expect(editor.getText()).toBe('a')
  })

  it('replaces the message in place rather than queueing a second copy', () => {
    const { editing, editor, queue } = build(['first', 'second'])

    editor.setText('draft')
    editing.cycle(-1)

    expect(editing.apply('second, rewritten')).toBe(true)
    expect(queue.list()).toEqual(['first', 'second, rewritten'])
    expect(editor.getText()).toBe('draft')
  })

  it('drops the message when the edit is emptied, and when it is dropped outright', () => {
    const { editing, queue } = build(['first', 'second'])

    editing.cycle(-1)
    expect(editing.apply('   ')).toBe(true)
    expect(queue.list()).toEqual(['first'])

    editing.cycle(-1)
    expect(editing.drop()).toBe(true)
    expect(queue.list()).toEqual([])
  })

  it('leaves an ordinary submit alone when no edit is open', () => {
    const { editing } = build(['first'])

    expect(editing.apply('a new message')).toBe(false)
    expect(editing.cancel()).toBe(false)
    expect(editing.drop()).toBe(false)
  })

  it('follows the message when the queue shifts under it', () => {
    const { editing, queue } = build(['first', 'second'])

    editing.cycle(-1)
    expect(queue.editing).toBe(1)

    queue.unshift('jumped the queue')
    expect(queue.editing).toBe(2)
    expect(queue.list()[2]).toBe('second')
  })

  it('reserves the message being edited from dispatch', () => {
    const { editing, editor, queue, sent, turn } = build(['first', 'second'])

    editor.setText('half-typed')
    editing.cycle(1)
    expect(queue.editing).toBe(0)

    editor.setText('first, rewritten')
    turn.active = false

    // The turn ended mid-edit. Sending the head now would send the text the
    // user is replacing and leave the rewrite in an editor with nothing to
    // apply it to.
    expect(queue.submitHead()).toBe('editing')
    expect(queue.drain()).toBe(false)
    expect(sent).toEqual([])
    expect(queue.list()).toEqual(['first', 'second'])
    expect(editor.getText()).toBe('first, rewritten')
    expect(editing.active).toBe(true)
  })

  it('releases the message and drains when the edit is applied', () => {
    const { editing, editor, queue, released, sent, turn } = build(['first', 'second'])

    editor.setText('half-typed')
    editing.cycle(1)
    turn.active = false

    expect(editing.apply('first, rewritten')).toBe(true)

    expect(queue.list()).toEqual(['second'])
    expect(sent).toEqual(['first, rewritten'])
    // The displaced draft comes back even though the message left the queue
    // in the same breath.
    expect(editor.getText()).toBe('half-typed')
    expect(released).toEqual([2])
  })

  it('releases and drains when the edit is cancelled', () => {
    const { editing, editor, queue, sent, turn } = build(['first'])

    editor.setText('half-typed')
    editing.cycle(1)
    turn.active = false

    expect(queue.drain()).toBe(false)
    expect(editing.cancel()).toBe(true)

    expect(sent).toEqual(['first'])
    expect(editor.getText()).toBe('half-typed')
  })

  it('releases and drains when the edit is dropped', () => {
    const { editing, editor, queue, sent, turn } = build(['first', 'second'])

    editor.setText('half-typed')
    editing.cycle(1)
    turn.active = false

    expect(editing.drop()).toBe(true)

    expect(queue.list()).toEqual([])
    expect(sent).toEqual(['second'])
    expect(editor.getText()).toBe('half-typed')
  })

  it('gives the draft back even if the queue lets go of the edit on its own', () => {
    const { editing, editor, queue } = build(['first'])

    editor.setText('half-typed')
    editing.cycle(1)

    // Nothing should do this any more; the draft is still the user's text.
    queue.setEditing(null)

    expect(editing.cycle(1)).toBe(true)
    expect(editing.cancel()).toBe(true)
    expect(editor.getText()).toBe('half-typed')
  })

  it('follows the message when an earlier one goes out under it', () => {
    const { editing, queue, turn } = build(['first', 'second'])

    editing.cycle(-1)
    expect(queue.editing).toBe(1)

    turn.active = false
    queue.submitHead()

    expect(queue.editing).toBe(0)
    expect(queue.list()).toEqual(['second'])
  })
})

describe('a draft holding a large paste', () => {
  const BIG = Array.from({ length: 20 }, (_, index) => `PASTED ${index}`).join('\n')

  it('comes back with its marker and its contents', () => {
    const { editing, editor, queue } = build(['queued'])

    editor.handleInput(paste(BIG))

    // The editor collapses a paste this size to a marker; what it would send
    // is the twenty lines behind it.
    expect(editor.getText()).toMatch(/^\[paste #\d+ \+20 lines\]$/)
    expect(editor.getExpandedText()).toBe(BIG)

    editing.cycle(-1)
    expect(editor.getText()).toBe('queued')

    expect(editing.cancel()).toBe(true)
    expect(queue.list()).toEqual(['queued'])

    // Both halves: the marker is on screen again and the payload is behind it,
    // so submitting the restored draft sends what was pasted.
    expect(editor.getText()).toMatch(/^\[paste #\d+ \+20 lines\]$/)
    expect(editor.getExpandedText()).toBe(BIG)
  })

  it('keeps every character when the draft mixed prose with the paste', () => {
    const { editing, editor } = build(['queued'])

    editor.setText('look at this: ')
    editor.handleInput(paste(BIG))

    const expanded = editor.getExpandedText()

    expect(expanded).toContain('look at this: ')
    expect(expanded).toContain('PASTED 19')

    editing.cycle(-1)
    editing.cancel()

    // The marker does not survive a mixed draft — there is no way to put one
    // back around part of a line — but nothing is lost and it stays editable.
    expect(editor.getExpandedText()).toBe(expanded)
  })

  it('is restored when the edit is applied, not only when it is cancelled', () => {
    const { editing, editor, queue } = build(['queued'])

    editor.handleInput(paste(BIG))
    editing.cycle(-1)

    expect(editing.apply('queued, rewritten')).toBe(true)
    expect(queue.list()).toEqual(['queued, rewritten'])
    expect(editor.getExpandedText()).toBe(BIG)
  })

  it('leaves an ordinary draft exactly as it was', () => {
    const { editing, editor } = build(['queued'])

    editor.setText('plain draft')
    editing.cycle(-1)
    editing.cancel()

    expect(editor.getText()).toBe('plain draft')
    expect(editor.getExpandedText()).toBe('plain draft')
  })
})

describe('reading the clipboard over OSC 52', () => {
  it('decodes a reply and leaves the rest of the burst alone', () => {
    const encoded = Buffer.from('pasted text', 'utf8').toString('base64')

    expect(parseOsc52Reply(`\x1b]52;c;${encoded}\x07`)).toBe('pasted text')
    expect(parseOsc52Reply(`\x1b]52;c;${encoded}\x1b\\`)).toBe('pasted text')
    expect(stripOsc52Reply(`\x1b]52;c;${encoded}\x07rest`)).toBe('rest')
    expect(parseOsc52Reply('ordinary keystrokes')).toBeNull()
    expect(parseOsc52Reply('\x1b]52;c;?\x07')).toBeNull()
  })

  it('wraps the query for tmux and screen', () => {
    expect(osc52ReadQuery({})).toBe('\x1b]52;c;?\x07')
    expect(osc52ReadQuery({ TMUX: '/tmp/tmux' })).toContain('\x1bPtmux;')
    expect(osc52ReadQuery({ STY: '1.pts-0' })).toContain('\x1bP\x1b]52')
  })

  it('asks the terminal and resolves with what it answers', async () => {
    const terminal = new FakeTerminal(80, 24)
    const tui = new TuiMainScreen(terminal)

    tui.start()

    const reading = readOsc52Clipboard(tui, {}, 1000)

    expect(terminal.output()).toContain('\x1b]52;c;?')
    terminal.onInput?.(`\x1b]52;c;${Buffer.from('from the terminal').toString('base64')}\x07`)

    expect(await reading).toBe('from the terminal')

    tui.stop()
  })

  it('gives up quietly on a terminal that never answers', async () => {
    const tui = new TuiMainScreen(new FakeTerminal(80, 24))

    tui.start()
    expect(await readOsc52Clipboard(tui, {}, 5)).toBeNull()
    tui.stop()
  })
})

/** A `spawn` whose children start and then say nothing at all: no data, no
 *  `close`, no `error`. This is what a wedged `xclip` looks like. */
function hangingSpawn() {
  const killed: string[] = []
  const started: string[] = []

  const spawn = ((cmd: string, args: readonly string[] = []) => {
    started.push([cmd, ...args].join(' '))

    const stdout = Object.assign(new EventEmitter(), { setEncoding: () => {} })

    return Object.assign(new EventEmitter(), {
      kill: (signal?: string) => {
        killed.push(`${cmd} ${signal ?? ''}`.trim())

        return true
      },
      stdout
    })
  }) as unknown as typeof SpawnFn

  return { killed, spawn, started }
}

describe('reading the clipboard from a native helper', () => {
  it('gives up on a helper that never answers, and kills it', async () => {
    const { killed, spawn, started } = hangingSpawn()
    const began = Date.now()

    // Without a deadline this never settles, and the OSC 52 fallback behind it
    // is never even asked — paste simply does nothing, for ever.
    const text = await readClipboardText('linux', spawn, {}, 20)

    expect(text).toBeNull()
    expect(started.length).toBeGreaterThan(0)
    expect(killed).toHaveLength(started.length)
    expect(Date.now() - began).toBeLessThan(2000)
  })

  it('takes what a helper that does answer says', async () => {
    const spawn = ((cmd: string) => {
      const stdout = Object.assign(new EventEmitter(), { setEncoding: () => {} })
      const child = Object.assign(new EventEmitter(), { kill: () => true, stdout })

      queueMicrotask(() => {
        stdout.emit('data', `from ${cmd}`)
        child.emit('close', 0)
      })

      return child
    }) as unknown as typeof SpawnFn

    expect(await readClipboardText('darwin', spawn, {}, 500)).toBe('from pbpaste')
  })
})
