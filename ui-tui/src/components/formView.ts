// The small forms the model picker needs: an API key, a base URL, the wire
// choice.
//
// Tab and Shift+Tab move between fields and the submit action, and never escape
// to the chat editor — the slot is the form's for as long as it is open. Enter
// in a text or secret field submits the whole form, which is what a one-field
// form makes people expect; on the wire row Enter cycles the value instead, so
// the explicit submit row is how that form is sent.

import type { Component, Focusable, KeybindingsManager } from '@earendil-works/pi-tui'

import { Container, Input, SettingsList, Spacer, Text } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

import { joinHints, keyHint, rawKeyHint } from '../lib/keybindings.js'
import { SecretInput } from './secretInput.js'
import { HorizontalRule, OptionalText } from './selectorList.js'

const SHIFT_TAB = '\x1b[Z'

export interface FormFieldSpec {
  /** Dim note under the label. */
  hint?: string
  id: string
  kind: 'enum' | 'secret' | 'text'
  label: string
  placeholder?: string
  /** Enum only: the values Enter cycles through. */
  values?: string[]
  value?: string
}

export interface FormViewOptions {
  fields: FormFieldSpec[]
  keybindings: KeybindingsManager
  onCancel: () => void
  onSubmit: (values: Record<string, string>) => void
  submitLabel: string
  subtitle?: string
  theme: Theme
  title: string
}

interface Field {
  control: Component
  focus: (focused: boolean) => void
  label: OptionalText
  read: () => string
  spec: FormFieldSpec
}

export class FormView extends Container implements Focusable {
  private readonly actionText = new Text('', 1, 0)
  private busy = false
  private readonly errorText = new OptionalText()
  private readonly fields: Field[] = []
  private index = 0
  private isFocused = false
  private readonly options: FormViewOptions
  private readonly statusText = new OptionalText()

  constructor(options: FormViewOptions) {
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

    for (const spec of options.fields) {
      const field = this.buildField(spec)

      this.fields.push(field)

      if (spec.kind !== 'enum') {
        this.addChild(field.label)
      }

      this.addChild(field.control)
      this.addChild(new Spacer(1))
    }

    this.addChild(this.actionText)
    this.addChild(this.statusText)
    this.addChild(this.errorText)
    this.addChild(new Spacer(1))
    this.addChild(
      new Text(
        joinHints([
          rawKeyHint(theme, 'tab', 'next field'),
          keyHint(options.keybindings, theme, 'tui.input.submit', 'submit'),
          keyHint(options.keybindings, theme, 'tui.select.cancel', 'back')
        ]),
        1,
        0
      )
    )
    this.addChild(new HorizontalRule(theme))

    this.paint()
  }

  get focused(): boolean {
    return this.isFocused
  }

  set focused(value: boolean) {
    this.isFocused = value
    this.paint()
  }

  /** Every field's current text, keyed by field id. */
  values(): Record<string, string> {
    const out: Record<string, string> = {}

    for (const field of this.fields) {
      out[field.spec.id] = field.read()
    }

    return out
  }

  /** Refuse further submits while a mutation is in flight. */
  setBusy(busy: boolean): void {
    this.busy = busy
    this.paint()
  }

  setStatus(text?: string): void {
    this.statusText.setText(text ? this.options.theme.fg('muted', text) : '')
  }

  setError(text?: string): void {
    this.errorText.setText(text ? this.options.theme.fg('error', text) : '')
  }

  /** Called by the stage that holds this form when it is replaced. */
  dispose(): void {
    this.clearSecrets()
  }

  /** Drop typed secrets when the form is left. */
  clearSecrets(): void {
    for (const field of this.fields) {
      if (field.spec.kind === 'secret') {
        ;(field.control as SecretInput).clear()
      }
    }
  }

  handleInput(data: string): void {
    const kb = this.options.keybindings

    if (kb.matches(data, 'tui.select.cancel')) {
      this.options.onCancel()

      return
    }

    if (data === SHIFT_TAB) {
      this.move(-1)

      return
    }

    if (kb.matches(data, 'tui.input.tab')) {
      this.move(1)

      return
    }

    const field = this.fields[this.index]

    if (!field) {
      if (kb.matches(data, 'tui.input.submit')) {
        this.submit()
      }

      return
    }

    if (field.spec.kind === 'enum') {
      ;(field.control as SettingsList).handleInput(data)

      return
    }

    if (kb.matches(data, 'tui.input.submit')) {
      this.submit()

      return
    }

    ;(field.control as Input).handleInput(data)
  }

  private submit(): void {
    if (this.busy) {
      return
    }

    this.options.onSubmit(this.values())
  }

  private buildField(spec: FormFieldSpec): Field {
    const { theme } = this.options
    const label = new OptionalText()

    if (spec.kind === 'enum') {
      const values = spec.values ?? []
      let current = spec.value ?? values[0] ?? ''
      const list = new SettingsList(
        [
          {
            currentValue: current,
            id: spec.id,
            label: spec.label,
            values,
            ...(spec.hint ? { description: spec.hint } : {})
          }
        ],
        1,
        theme.settingsListTheme(),
        (_id, next) => {
          // The form draft only. Nothing reaches the gateway until submit.
          current = next
        },
        () => this.options.onCancel()
      )

      return {
        control: list,
        focus: () => {},
        label,
        read: () => current,
        spec
      }
    }

    if (spec.kind === 'secret') {
      const input = new SecretInput({
        theme,
        ...(spec.placeholder ? { placeholder: spec.placeholder } : {})
      })

      input.onSubmit = () => this.submit()
      input.onEscape = () => this.options.onCancel()

      return {
        control: input,
        focus: focused => {
          input.focused = focused
        },
        label,
        read: () => input.getValue(),
        spec
      }
    }

    const input = new Input({
      placeholder: spec.placeholder ?? '',
      placeholderStyle: text => theme.fg('muted', text),
      prompt: theme.fg('muted', '❯ ')
    })

    if (spec.value) {
      input.setValue(spec.value)
    }

    input.onSubmit = () => this.submit()
    input.onEscape = () => this.options.onCancel()

    return {
      control: input,
      focus: focused => {
        input.focused = focused
      },
      label,
      read: () => input.getValue(),
      spec
    }
  }

  private move(delta: number): void {
    const stops = this.fields.length + 1

    this.index = (this.index + delta + stops) % stops
    this.paint()
  }

  private paint(): void {
    const { theme } = this.options

    this.fields.forEach((field, position) => {
      const active = this.isFocused && position === this.index
      const hint = field.spec.hint ? theme.fg('muted', `  ${field.spec.hint}`) : ''

      field.label.setText(`${theme.fg(active ? 'accent' : 'label', field.spec.label)}${hint}`)
      field.focus(active)
    })

    const onAction = this.isFocused && this.index === this.fields.length
    const label = this.busy ? 'Working…' : this.options.submitLabel

    this.actionText.setText(onAction ? theme.fg('accent', `→ ${label}`) : theme.fg('muted', `  ${label}`))
  }
}
