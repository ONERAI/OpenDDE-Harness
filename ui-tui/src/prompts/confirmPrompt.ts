// Yes/no, from two places: the gateway's `confirm.request`, which has a
// deadline and a default the broker applies when it expires, and this UI's own
// delete/replace/disconnect confirmations, which have neither and never reach
// the wire. The mode is explicit so the two can never be confused.

import type { Focusable, KeybindingsManager } from '@earendil-works/pi-tui'

import { Container, Spacer, Text } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

import { HorizontalRule, OptionalText } from '../components/selectorList.js'
import { joinHints, keyHint, rawKeyHint } from '../lib/keybindings.js'
import { ChoiceList, sanitizeForDisplay } from './choiceList.js'

export interface ConfirmPromptOptions {
  /** `true` labels: what Yes means, e.g. `Disconnect openai`. */
  confirmLabel?: string
  defaultAnswer: boolean
  denyLabel?: string
  keybindings: KeybindingsManager
  /** `local` has no timer and no respond RPC. */
  mode: 'local' | 'rpc'
  onAnswer: (value: boolean) => void
  prompt: string
  theme: Theme
  title?: string
}

export class ConfirmPrompt extends Container implements Focusable {
  focused = false

  private busy = false
  private readonly list: ChoiceList
  private readonly options: ConfirmPromptOptions
  private seconds: null | number = null
  private readonly statusText = new OptionalText()
  private readonly titleText = new Text('', 1, 0)

  constructor(options: ConfirmPromptOptions) {
    super()
    this.options = options

    const { theme } = options
    const yes = options.confirmLabel ?? 'Yes'
    const no = options.denyLabel ?? 'No'

    this.list = new ChoiceList(theme, [
      { label: no, value: 'no', ...(options.defaultAnswer ? {} : { note: theme.fg('muted', '· default') }) },
      { label: yes, value: 'yes', ...(options.defaultAnswer ? { note: theme.fg('muted', '· default') } : {}) }
    ])
    this.list.select(options.defaultAnswer ? 1 : 0)

    this.addChild(new HorizontalRule(theme))
    this.addChild(new Spacer(1))
    this.addChild(this.titleText)
    this.addChild(new Spacer(1))
    this.addChild(new Text(theme.fg('text', sanitizeForDisplay(options.prompt)), 3, 0))
    this.addChild(new Spacer(1))
    this.addChild(this.list)
    this.addChild(this.statusText)
    this.addChild(new Spacer(1))
    this.addChild(
      new Text(
        joinHints([
          rawKeyHint(theme, 'y/n', 'answer'),
          keyHint(options.keybindings, theme, 'tui.select.confirm', 'select'),
          keyHint(options.keybindings, theme, 'tui.select.cancel', 'no')
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
      this.options.onAnswer(false)

      return
    }

    if (data === 'y' || data === 'Y' || data === '2') {
      this.list.select(1)
      this.options.onAnswer(true)

      return
    }

    if (data === 'n' || data === 'N' || data === '1') {
      this.list.select(0)
      this.options.onAnswer(false)

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
      this.options.onAnswer(this.list.selected()?.value === 'yes')
    }
  }

  private paintTitle(): void {
    const { theme } = this.options
    const countdown = this.options.mode === 'rpc' && this.seconds !== null ? ` (${this.seconds}s)` : ''

    this.titleText.setText(theme.bold(theme.fg('text', `${this.options.title ?? 'Confirm'}${countdown}`)))
  }
}
