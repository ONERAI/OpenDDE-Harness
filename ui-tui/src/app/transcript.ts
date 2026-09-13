// The transcript: the chat container everything is drawn into, the few lines
// the app writes into it itself, and the prompt `/retry` would send again.
//
// The retry prompt lives here because it is a fact about the transcript rather
// than about the turn: it survives the turn that sent it, and `/undo` and a
// turn that stored nothing both change what it means without ending anything.

import type { Component, TUI } from '@earendil-works/pi-tui'

import { Container, Spacer } from '@earendil-works/pi-tui'

import type { RetryCandidate, TranscriptEntry } from '../commands/index.js'
import type { SystemTone } from '../components/systemLine.js'
import type { Theme } from '../theme.js'

import { AssistantMessage } from '../components/assistantMessage.js'
import { systemBlock, systemBody, systemLine } from '../components/systemLine.js'
import { ThemedLine } from '../components/themedText.js'
import { UserMessage } from '../components/userMessage.js'

export class TranscriptView {
  /** The component every message is added to. */
  readonly container = new Container()

  /** The prompt `/retry` resends, and whether its exchange has already been
   *  taken off screen by an undo. */
  private retryPrompt: null | RetryCandidate = null

  constructor(
    private readonly theme: Theme,
    private readonly tui: TUI
  ) {}

  print(text: string, tone?: SystemTone): void {
    this.container.addChild(systemLine(this.theme, text, tone))
  }

  /**
   * A block of output under a header, which is how `/status`, `/tasks` and an
   * approval preview reach the transcript.
   *
   * The header is bold in the terminal's own foreground and the body is grey.
   * Concatenating the two and painting the result muted made a report read as
   * one undifferentiated wall, with nothing saying where it began. `systemLine`
   * would not do: it paints muted, and bold grey is still grey.
   */
  printBlock(text: string, title?: string): void {
    const box = new Container()

    // One blank above the whole block, as pi puts one above a note -- not one
    // between the header and the body it belongs to.
    box.addChild(new Spacer(1))

    if (title) {
      // Inset by one column, like every other line of system output: the body
      // moved and a header flush against the edge of its own block reads as
      // belonging to something else.
      box.addChild(new ThemedLine(painted => this.theme.bold(painted), title, 1))
    }

    box.addChild(systemBody(this.theme, text))
    this.container.addChild(box)
  }

  /** A block of its own, for the lines boot writes before a session exists. */
  addBlock(text: string, tone?: SystemTone): void {
    this.container.addChild(systemBlock(this.theme, text, tone))
  }

  add(child: Component): void {
    this.container.addChild(child)
  }

  /** A fresh transcript for a session that replaces this one. */
  clear(): void {
    this.container.clear()
    this.retryPrompt = null
  }

  /** What is on screen, oldest first, as the commands read it. */
  history(): TranscriptEntry[] {
    const entries: TranscriptEntry[] = []

    for (const child of this.container.children) {
      if (child instanceof UserMessage) {
        entries.push({ role: 'user', text: child.text })
      } else if (child instanceof AssistantMessage && child.text) {
        entries.push({ role: 'assistant', text: child.text })
      }
    }

    return entries
  }

  retryCandidate(): null | RetryCandidate {
    return this.retryPrompt
  }

  /** Every prompt that goes out becomes what `/retry` resends. */
  remember(text: string): void {
    this.retryPrompt = { onServer: true, text }
  }

  /** A turn that stored nothing leaves no exchange to undo, so `/retry` of
   *  that prompt must resend it rather than undo the exchange before it. */
  forget(content: string): void {
    if (this.retryPrompt?.text === content) {
      this.retryPrompt = { onServer: false, text: content }
    }
  }

  /**
   * Drop the exchange `session.undo` removed on the gateway, and everything
   * shown after it.
   *
   * The gateway removes its last *saved* exchange, which is not always the
   * last one on screen: a turn that failed before it produced anything leaves
   * an echo here and stored nothing there. Taking the last echo regardless
   * removed the failed attempt and left the deleted exchange on display.
   * Anything after the saved one goes too — those are failed attempts the
   * gateway never held either.
   */
  dropLastExchange(): boolean {
    const children = this.container.children
    let start = -1

    for (let i = children.length - 1; i >= 0; i -= 1) {
      const child = children[i]

      if (child instanceof UserMessage && child.saved) {
        start = i
        break
      }
    }

    if (start < 0) {
      return false
    }

    for (const child of children.slice(start)) {
      this.container.removeChild(child as Component)
    }

    if (this.retryPrompt) {
      this.retryPrompt = { ...this.retryPrompt, onServer: false }
    }

    this.tui.requestRender()

    return true
  }
}
