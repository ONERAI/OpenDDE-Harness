// The terminal's OSC 11 background reply lands up to 200 ms after boot and can
// swap the whole palette under a screen that is already populated (entry.ts).
// Everything on it has to repaint: `tui.invalidate()` plus a render must emit
// the new palette's ANSI and none of the old one's.

import type { Component, TUI } from '@earendil-works/pi-tui'

import { TuiMainScreen } from '@earendil-works/pi-tui'
import { describe, expect, it } from 'vitest'

import type { ProteinDesignTasks } from '../tasks/monitor.js'

import { AssistantMessage } from '../components/assistantMessage.js'
import { Footer } from '../components/footer.js'
import { HelpPanel } from '../components/helpPanel.js'
import { ProteinDesignTaskBar } from '../components/proteinDesignTaskBar.js'
import { MessageQueue } from '../components/queue.js'
import { SessionPanel } from '../components/sessionPanel.js'
import { systemBlock, systemLine } from '../components/systemLine.js'
import { ToolExecution } from '../components/toolExecution.js'
import { UserMessage } from '../components/userMessage.js'
import { createKeybindings } from '../lib/keybindings.js'
import { Theme } from '../theme.js'
import { FakeTerminal } from './fakes.js'

// Truecolor, so each palette has an unmistakable signature on the wire.
// Body text is unpainted now — it is the terminal's own — so the signature of
// a scheme is its greys, its semantic colors and its grounds.
const DARK = {
  dim: '38;2;102;102;102', // #666666
  error: '38;2;255;107;128', // #FF6B80
  muted: '38;2;153;153;153', // #999999
  prompt: '38;2;167;139;250', // #A78BFA
  userBg: '48;2;55;55;55' // #373737
}

const LIGHT = {
  dim: '38;2;118;118;118', // #767676
  error: '38;2;171;43;63', // #AB2B3F
  muted: '38;2;102;102;102', // #666666
  prompt: '38;2;124;58;237', // #7C3AED
  userBg: '48;2;240;240;240' // #F0F0F0
}

/** Render on the dark palette, switch to light the way entry.ts does, render
 *  again. Returns what the terminal was told the second time. */
function repaint(build: (theme: Theme, tui: TUI) => Component, darkGrey = DARK.muted): string {
  const theme = new Theme('dark', 3)
  const terminal = new FakeTerminal(80, 24)
  const tui = new TuiMainScreen(terminal)

  tui.addChild(build(theme, tui))
  tui.renderNow()

  expect(terminal.output()).toContain(darkGrey)

  terminal.writes.length = 0
  expect(theme.setScheme('light')).toBe(true)
  tui.invalidate()
  tui.renderNow()

  return terminal.output()
}

describe('a late theme swap repaints what is already on screen', () => {
  it('repaints a system line', () => {
    // A note is painted in `dim`, the grey pi writes its statuses in.
    const output = repaint(theme => systemLine(theme, 'read path=/etc/hosts'), DARK.dim)

    expect(output).toContain(LIGHT.dim)
    expect(output).not.toContain(DARK.dim)
  })

  it('repaints a system block, keeping its tone', () => {
    const theme = new Theme('dark', 3)
    const terminal = new FakeTerminal(80, 24)
    const tui = new TuiMainScreen(terminal)

    tui.addChild(systemBlock(theme, 'error: the gateway said no', 'error'))
    tui.renderNow()
    expect(terminal.output()).toContain(DARK.error)

    terminal.writes.length = 0
    theme.setScheme('light')
    tui.invalidate()
    tui.renderNow()

    expect(terminal.output()).toContain(LIGHT.error)
    expect(terminal.output()).not.toContain(DARK.error)
  })

  it('repaints the session panel', () => {
    const output = repaint(theme => new SessionPanel(theme, { commandCount: 4, sessionId: 'tui:abc123' }))

    expect(output).toContain(LIGHT.muted)
    expect(output).not.toContain(DARK.muted)
  })

  it('repaints a user message, background and all', () => {
    const theme = new Theme('dark', 3)
    const terminal = new FakeTerminal(80, 24)
    const tui = new TuiMainScreen(terminal)

    tui.addChild(new UserMessage(theme, 'summarise the run'))
    tui.renderNow()
    expect(terminal.output()).toContain(DARK.userBg)

    terminal.writes.length = 0
    theme.setScheme('light')
    tui.invalidate()
    tui.renderNow()

    expect(terminal.output()).toContain(LIGHT.userBg)
    expect(terminal.output()).toContain(LIGHT.prompt)
    expect(terminal.output()).not.toContain(DARK.userBg)
    expect(terminal.output()).not.toContain(DARK.prompt)
  })

  it("repaints an assistant message's thinking label and closing note", () => {
    const output = repaint(theme => {
      const message = new AssistantMessage(theme)

      message.appendThinking('weighing two options')
      message.setNote('stopped: max tokens')

      return message
    })

    expect(output).toContain(LIGHT.muted)
    expect(output).not.toContain(DARK.muted)
  })

  it('repaints the queued-message list', () => {
    const output = repaint(theme => {
      const queue = new MessageQueue({
        actions: { interrupt: () => {}, send: () => {} },
        isTurnActive: () => true,
        theme
      })

      queue.submit('the next thing to do')

      return queue.view
    })

    expect(output).toContain(LIGHT.muted)
    expect(output).not.toContain(DARK.muted)
  })

  it('repaints a tool panel', () => {
    const output = repaint((theme, tui) => {
      const tool = new ToolExecution(theme, tui, { arguments: { path: '/etc/hosts' }, name: 'read' })

      tool.complete({ result_preview: '127.0.0.1 localhost', truncated: false })

      return tool
    })

    expect(output).toContain(LIGHT.muted)
    expect(output).not.toContain(DARK.muted)
  })
})

/** The same swap with no `tui.invalidate()` at all.
 *
 *  Every component that holds painted lines keys them on the palette, so the
 *  right colors do not depend on a caller remembering to invalidate. This is
 *  the property, not an optimisation: the palette moves twice during startup,
 *  once when the terminal answers the background query and again if
 *  `tui.theme` names one, and a component that missed either would sit in the
 *  old palette for the rest of the session. */
function repaintUnprompted(build: (theme: Theme, tui: TUI) => Component, darkGrey = DARK.muted): string {
  const theme = new Theme('dark', 3)
  const terminal = new FakeTerminal(80, 24)
  const tui = new TuiMainScreen(terminal)

  tui.addChild(build(theme, tui))
  tui.renderNow()

  expect(terminal.output()).toContain(darkGrey)

  terminal.writes.length = 0
  expect(theme.setScheme('light')).toBe(true)
  tui.renderNow()

  return terminal.output()
}

describe('a palette that moves without anyone invalidating', () => {
  it('repaints a user message', () => {
    const theme = new Theme('dark', 3)
    const message = new UserMessage(theme, 'design something')

    expect(message.render(80).join('\n')).toContain(DARK.userBg)
    theme.setScheme('light')

    const painted = message.render(80).join('\n')

    expect(painted).toContain(LIGHT.userBg)
    expect(painted).not.toContain(DARK.userBg)
  })

  it('repaints a tool panel', () => {
    const output = repaintUnprompted((theme, tui) => {
      const tool = new ToolExecution(theme, tui, { arguments: { path: '/etc/hosts' }, name: 'read' })

      tool.complete({ result_preview: '127.0.0.1 localhost', truncated: false })

      return tool
    })

    expect(output).toContain(LIGHT.muted)
    expect(output).not.toContain(DARK.muted)
  })

  it('repaints the help panel', () => {
    const output = repaintUnprompted(
      theme =>
        new HelpPanel(theme, createKeybindings(), {
          catalog: ['compare'],
          commands: [{ description: 'exit OpenDDE Harness', name: 'quit' }],
          hotkeys: ['app.reset'],
          skillCount: 4
        })
    )

    expect(output).toContain(LIGHT.muted)
    expect(output).not.toContain(DARK.muted)
  })

  it('repaints the design task bar', () => {
    const tasks = {
      focusedTaskId: undefined,
      list: () => [
        {
          cycle: 2,
          phase: null,
          skill: null,
          status: 'running',
          summary: 'folding',
          target: 'VEGF binder',
          taskId: 'abcdef0123456789',
          timestamp: '2026-09-11T00:00:00.000Z',
          tool: null,
          totalCycles: 6
        }
      ],
      revision: 1,
      visible: true
    } as unknown as ProteinDesignTasks

    const output = repaintUnprompted(theme => new ProteinDesignTaskBar(theme, tasks))

    expect(output).toContain(LIGHT.muted)
    expect(output).not.toContain(DARK.muted)
  })

  it('repaints the footer', () => {
    const output = repaintUnprompted(
      theme =>
        new Footer(
          theme,
          () => ({
            branch: 'tui/pi-tui',
            cwd: '/home/dev/work/opendde-harness',
            model: 'gpt-5',
            provider: 'openai',
            usage: {
              cacheHitPercent: 74,
              contextTokens: 36_000,
              contextWindow: 200_000,
              costUsd: 0.0314,
              input: 12,
              output: 3
            }
          }),
          '/home/dev'
        ),
      DARK.dim
    )

    // The footer is painted in `dim`, not `muted`.
    expect(output).toContain(LIGHT.dim)
    expect(output).not.toContain(DARK.dim)
  })
})
