// The agent asking a question mid-turn. Its choices are answered by their exact
// text, never by the number shown beside them, and "Other" swaps the list for a
// plain input. Nothing typed here reaches the chat editor, the prompt history
// or `turn.send`.

import type { Focusable, KeybindingsManager } from '@earendil-works/pi-tui'

import { Container, Input, Spacer, Text } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

import { HorizontalRule, OptionalText } from '../components/selectorList.js'
import { joinHints, keyHint, rawKeyHint } from '../lib/keybindings.js'
import { ChoiceList, sanitizeForDisplay } from './choiceList.js'

/** A sentinel that cannot collide with a server-supplied choice: every real
 *  choice is submitted verbatim, and a leading space is trimmed out of one. */
const OTHER = ' other'

export interface ClarifyPromptOptions {
  choices: string[]
  keybindings: KeybindingsManager
  onAnswer: (text: string) => void
  onCancel: () => void
  question: string
  theme: Theme
}

export class ClarifyPrompt extends Container implements Focusable {
  private readonly body = new Container()
  private busy = false
  private readonly hintText = new Text('', 1, 0)
  private input: Input | undefined
  private isFocused = false
  private list: ChoiceList | undefined
  private readonly options: ClarifyPromptOptions
  private readonly statusText = new OptionalText()

  constructor(options: ClarifyPromptOptions) {
    super()
    this.options = options

    const { theme } = options

    this.addChild(new HorizontalRule(theme))
    this.addChild(new Spacer(1))
    this.addChild(new Text(theme.bold(theme.fg('text', 'The agent needs an answer')), 1, 0))
    this.addChild(new Spacer(1))
    this.addChild(new Text(theme.fg('text', sanitizeForDisplay(options.question)), 3, 0))
    this.addChild(new Spacer(1))
    this.addChild(this.body)
    this.addChild(this.statusText)
    this.addChild(new Spacer(1))
    this.addChild(this.hintText)
    this.addChild(new HorizontalRule(theme))

    if (options.choices.length > 0) {
      this.showChoices()
    } else {
      this.showInput()
    }
  }

  get focused(): boolean {
    return this.isFocused
  }

  set focused(value: boolean) {
    this.isFocused = value

    if (this.input) {
      this.input.focused = value
    }
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

    if (this.input) {
      if (kb.matches(data, 'tui.select.cancel')) {
        // Back to the choices when there were any; otherwise this is the root.
        if (this.options.choices.length > 0) {
          this.showChoices()
        } else {
          this.options.onCancel()
        }

        return
      }

      // Digits are text here, not choice numbers.
      this.input.handleInput(data)

      return
    }

    if (kb.matches(data, 'tui.select.cancel')) {
      this.options.onCancel()

      return
    }

    if (/^[1-9]$/.test(data)) {
      const index = Number(data) - 1

      if (this.list && index < this.list.length) {
        this.list.select(index)
        this.take()
      }

      return
    }

    if (kb.matches(data, 'tui.select.up')) {
      this.list?.move(-1)

      return
    }

    if (kb.matches(data, 'tui.select.down')) {
      this.list?.move(1)

      return
    }

    if (kb.matches(data, 'tui.select.confirm')) {
      this.take()
    }
  }

  private take(): void {
    const choice = this.list?.selected()

    if (!choice) {
      return
    }

    if (choice.value === OTHER) {
      this.showInput()

      return
    }

    this.options.onAnswer(choice.value)
  }

  private showChoices(): void {
    const { theme } = this.options

    this.input = undefined
    this.list = new ChoiceList(theme, [
      ...this.options.choices.map(choice => ({ label: sanitizeForDisplay(choice), value: choice })),
      { label: 'Other — type an answer', value: OTHER }
    ])

    this.body.clear()
    this.body.addChild(this.list)
    this.hintText.setText(
      joinHints([
        rawKeyHint(theme, '↑↓', 'navigate'),
        keyHint(this.options.keybindings, theme, 'tui.select.confirm', 'answer'),
        keyHint(this.options.keybindings, theme, 'tui.select.cancel', 'cancel')
      ])
    )
  }

  private showInput(): void {
    const { theme } = this.options
    const input = new Input({
      placeholder: 'your answer',
      placeholderStyle: text => theme.fg('muted', text),
      prompt: theme.fg('muted', '❯ ')
    })

    input.focused = this.isFocused
    input.onSubmit = value => {
      const answer = value.trim()

      if (answer) {
        this.options.onAnswer(answer)
      }
    }

    this.list = undefined
    this.input = input
    this.body.clear()
    this.body.addChild(input)
    this.hintText.setText(
      joinHints([
        keyHint(this.options.keybindings, theme, 'tui.input.submit', 'answer'),
        keyHint(
          this.options.keybindings,
          theme,
          'tui.select.cancel',
          this.options.choices.length > 0 ? 'back' : 'cancel'
        )
      ])
    )
  }
}
