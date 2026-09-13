// "Approval required" — the one-shot authority the shell tool asks for before
// it runs a command the policy singled out.
//
// The broker owns the command text and the deadline; this only shows them and
// reports a choice. There is no "always allow": the backend has no such state,
// so offering one would be a lie.

import type { Focusable, KeybindingsManager } from '@earendil-works/pi-tui'

import { Container, Spacer, Text } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

import { HorizontalRule, OptionalText } from '../components/selectorList.js'
import { joinHints, keyHint, rawKeyHint } from '../lib/keybindings.js'
import { ChoiceList, sanitizeForDisplay } from './choiceList.js'

export type ApprovalChoice = 'allow' | 'deny'

export interface ApprovalPromptOptions {
  command: string
  description: string
  keybindings: KeybindingsManager
  onChoose: (choice: ApprovalChoice) => void
  theme: Theme
  /** The inline preview is shorter than the command; a note says where the
   *  whole thing went, and it is only set when it really was appended. */
  truncated?: boolean
}

export class ApprovalPrompt extends Container implements Focusable {
  focused = false

  private busy = false
  private readonly list: ChoiceList
  private readonly options: ApprovalPromptOptions
  private seconds: null | number = null
  private readonly statusText = new OptionalText()
  private readonly titleText = new Text('', 1, 0)

  constructor(options: ApprovalPromptOptions) {
    super()
    this.options = options

    const { theme } = options

    this.list = new ChoiceList(theme, [
      { label: 'Allow once', value: 'allow' },
      { label: 'Deny', value: 'deny' }
    ])

    this.addChild(new HorizontalRule(theme))
    this.addChild(new Spacer(1))
    this.addChild(this.titleText)
    this.addChild(new Spacer(1))
    this.addChild(new Text(theme.fg('text', sanitizeForDisplay(options.command)), 3, 0))

    if (options.truncated) {
      this.addChild(new Text(theme.fg('muted', 'the full command is in the transcript above'), 3, 0))
    }

    this.addChild(new Spacer(1))
    this.addChild(this.list)
    this.addChild(this.statusText)
    this.addChild(new Spacer(1))
    this.addChild(
      new Text(
        joinHints([
          rawKeyHint(theme, '1/2', 'choose'),
          keyHint(options.keybindings, theme, 'tui.select.confirm', 'select'),
          keyHint(options.keybindings, theme, 'tui.select.cancel', 'deny')
        ]),
        1,
        0
      )
    )
    this.addChild(new HorizontalRule(theme))

    this.paintTitle()
  }

  setSeconds(seconds: number): void {
    this.seconds = seconds
    this.paintTitle()
  }

  /** A response is in flight; further keys do nothing. */
  setBusy(busy: boolean): void {
    this.busy = busy
    this.statusText.setText(busy ? this.options.theme.fg('muted', 'sending…') : '')
  }

  handleInput(data: string): void {
    if (this.busy) {
      return
    }

    const kb = this.options.keybindings

    if (kb.matches(data, 'tui.select.cancel')) {
      this.options.onChoose('deny')

      return
    }

    if (data === '1') {
      this.list.select(0)
      this.options.onChoose('allow')

      return
    }

    if (data === '2') {
      this.list.select(1)
      this.options.onChoose('deny')

      return
    }

    if (kb.matches(data, 'tui.select.up')) {
      this.list.move(-1)

      return
    }

    if (kb.matches(data, 'tui.select.down')) {
      this.list.move(1)

      return
    }

    if (kb.matches(data, 'tui.select.confirm')) {
      this.options.onChoose(this.list.selected()?.value === 'allow' ? 'allow' : 'deny')
    }
  }

  private paintTitle(): void {
    const { theme } = this.options
    const countdown = this.seconds === null ? '' : ` (${this.seconds}s)`

    this.titleText.setText(theme.bold(theme.fg('warn', `Approval required · ${this.options.description}${countdown}`)))
  }
}
