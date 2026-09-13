// The sign-in stage: pi's own login flow, drawn where the editor was.
//
// A login is a conversation with a vendor that the gateway is holding: it
// reports what it is doing, and twice it stops and asks. So this view is a
// growing list of lines plus one slot for whatever is being asked — a list for
// pi's login-method menu, a field for the authorization code a browser login
// falls back to. Nothing here knows what a step means; the stage above decides
// what to show and what an answer is for.
//
// Esc is always cancel, even while a question is up: a sign-in nobody wants any
// more has to be abandonable without answering it first.

import type { Focusable, KeybindingsManager, SelectItem } from '@earendil-works/pi-tui'

import { Container, Input, SelectList, Spacer, Text } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

import { joinHints, keyHint, rawKeyHint } from '../lib/keybindings.js'
import { HorizontalRule, OptionalText } from './selectorList.js'

/** One option of pi's own menu, in pi's own shape. */
export interface LoginChoice {
  description?: string
  id: string
  label: string
}

export interface LoginViewOptions {
  keybindings: KeybindingsManager
  /** The string the open question asked for: an option id, or typed text. */
  onAnswer: (answer: string) => void
  onCancel: () => void
  /** The stage holding this view was replaced or the selector closed: the
   *  sign-in it was showing has nobody to show it to any more. */
  onDispose?: () => void
  subtitle?: string
  theme: Theme
  title: string
}

/** How many options of a menu are on screen at once. pi's menus are short. */
const VISIBLE_CHOICES = 8

export class LoginView extends Container implements Focusable {
  private readonly askSlot = new Container()
  private readonly errorText = new OptionalText()
  private readonly lines = new Container()
  private readonly options: LoginViewOptions
  private readonly promptText = new OptionalText()
  private readonly statusText = new OptionalText()

  /** The open menu's options, in pi's order, and which one is on. */
  private choices: LoginChoice[] = []
  private index = 0
  private input: Input | undefined
  private isFocused = false
  private list: SelectList | undefined

  constructor(options: LoginViewOptions) {
    super()
    this.options = options

    const { theme } = options

    this.addChild(new HorizontalRule(theme))
    this.addChild(new Spacer(1))
    this.addChild(new Text(theme.bold(theme.fg('text', options.title)), 1, 0))

    if (options.subtitle) {
      this.addChild(new Text(theme.fg('muted', options.subtitle), 1, 0))
    }

    this.addChild(new Spacer(1))
    this.addChild(this.lines)
    this.addChild(this.promptText)
    this.addChild(this.askSlot)
    this.addChild(this.statusText)
    this.addChild(this.errorText)
    this.addChild(new Spacer(1))
    this.addChild(
      new Text(
        joinHints([
          rawKeyHint(theme, '↑↓', 'navigate'),
          keyHint(options.keybindings, theme, 'tui.select.confirm', 'select'),
          keyHint(options.keybindings, theme, 'tui.select.cancel', 'cancel sign-in')
        ]),
        1,
        0
      )
    )
    this.addChild(new HorizontalRule(theme))
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

  /**
   * One more line of what the vendor is doing. Kept, so the code stays on screen.
   *
   * `tone` is how pi colours the same line in its own login dialog: the URL in
   * accent, the click hint dim, the code and any instructions as a warning.
   */
  addLine(text: string, tone: 'accent' | 'muted' | 'text' | 'warn' = 'text'): void {
    this.lines.addChild(new Text(this.options.theme.fg(tone, `  ${text}`), 1, 0))
  }

  /** A URL as pi writes one: an OSC 8 hyperlink, so a terminal can open it.
   *  With a label, the label is the link's text, as pi's `showInfo` draws a
   *  provider's own links. */
  addLink(url: string, label?: string): void {
    this.addLine(`\x1b]8;;${url}\x07${label ?? url}\x1b]8;;\x07`, 'accent')
    // pi's own hint, and pi's own platform test for which key it names.
    this.addLine(process.platform === 'darwin' ? 'Cmd+click to open' : 'Ctrl+click to open', 'muted')
  }

  /** pi's own menu: its options in its own order, its first one selected. */
  askSelect(message: string, choices: LoginChoice[]): void {
    this.clearAsk()

    if (choices.length === 0) {
      return
    }

    const items: SelectItem[] = choices.map(choice => ({
      label: choice.label,
      value: choice.id,
      ...(choice.description ? { description: choice.description } : {})
    }))
    const list = new SelectList(items, VISIBLE_CHOICES, this.options.theme.selectListTheme())

    // pi's own first option is the default, and pi's own label says so.
    this.choices = choices
    this.index = 0
    list.setSelectedIndex(0)
    list.onSelectionChange = item => {
      this.index = Math.max(
        0,
        choices.findIndex(choice => choice.id === item.value)
      )
    }
    list.onSelect = item => this.options.onAnswer(item.value)
    list.onCancel = () => this.options.onCancel()

    this.list = list
    this.promptText.setText(this.options.theme.fg('label', `  ${message}`))
    this.askSlot.addChild(list)
  }

  /** A one-line field: the authorization code, pasted back from the browser. */
  askText(message: string, placeholder?: string): void {
    this.clearAsk()

    const input = new Input({
      placeholder: placeholder ?? '',
      placeholderStyle: text => this.options.theme.fg('muted', text),
      prompt: this.options.theme.fg('muted', '❯ ')
    })

    input.focused = this.isFocused
    input.onSubmit = value => this.options.onAnswer(value)
    input.onEscape = () => this.options.onCancel()

    this.input = input
    this.promptText.setText(this.options.theme.fg('label', `  ${message}`))
    this.askSlot.addChild(input)
  }

  /** Nothing is being asked any more: the answer went, or the flow moved on. */
  clearAsk(): void {
    this.askSlot.clear()
    this.promptText.setText('')
    this.choices = []
    this.index = 0
    this.input = undefined
    this.list = undefined
  }

  setError(text?: string): void {
    this.errorText.setText(text ? this.options.theme.fg('error', text) : '')
  }

  setStatus(text?: string): void {
    this.statusText.setText(text ? this.options.theme.fg('muted', `  ${text}`) : '')
  }

  /** Called by the stage that holds this view when it is replaced, and by the
   *  selector when it closes with this view up. */
  dispose(): void {
    this.clearAsk()
    this.options.onDispose?.()
  }

  handleInput(data: string): void {
    const kb = this.options.keybindings

    if (kb.matches(data, 'tui.select.cancel')) {
      this.options.onCancel()

      return
    }

    // Routed here rather than by `SelectList.handleInput`, which reads the
    // process-wide keybindings instead of the manager this view was given —
    // the same reason `SelectorList` does its own routing.
    if (this.list) {
      if (kb.matches(data, 'tui.select.up')) {
        this.moveBy(-1)
      } else if (kb.matches(data, 'tui.select.down')) {
        this.moveBy(1)
      } else if (kb.matches(data, 'tui.select.confirm')) {
        const choice = this.choices[this.index]

        if (choice) {
          this.options.onAnswer(choice.id)
        }
      }

      return
    }

    if (this.input) {
      if (kb.matches(data, 'tui.input.submit')) {
        this.options.onAnswer(this.input.getValue())

        return
      }

      this.input.handleInput(data)
    }

    // Nothing is being asked: the flow is waiting on the vendor, and a key
    // press is not an answer to anything.
  }

  private moveBy(delta: number): void {
    if (!this.list || this.choices.length === 0) {
      return
    }

    this.index = Math.min(this.choices.length - 1, Math.max(0, this.index + delta))
    this.list.setSelectedIndex(this.index)
  }
}
