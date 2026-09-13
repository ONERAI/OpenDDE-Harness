// The numbered list the three broker prompts share.
//
// It is not pi's `SelectList`: these prompts have no search box, and the digits
// are answers rather than text. Up/Down move, Enter takes the selection, and a
// digit takes that row directly.

import type { Component } from '@earendil-works/pi-tui'

import { truncateToWidth } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

export interface PromptChoice {
  /** Shown after the number, e.g. `Allow once`. */
  label: string
  /** Dim suffix, e.g. `· default`. */
  note?: string
  value: string
}

export class ChoiceList implements Component {
  private choices: PromptChoice[]
  private index = 0

  constructor(
    private readonly theme: Theme,
    choices: PromptChoice[]
  ) {
    this.choices = choices
  }

  get length(): number {
    return this.choices.length
  }

  get selectedIndex(): number {
    return this.index
  }

  selected(): PromptChoice | undefined {
    return this.choices[this.index]
  }

  setChoices(choices: PromptChoice[]): void {
    this.choices = choices
    this.index = Math.min(this.index, Math.max(0, choices.length - 1))
  }

  select(index: number): void {
    if (index >= 0 && index < this.choices.length) {
      this.index = index
    }
  }

  move(delta: number): void {
    if (this.choices.length === 0) {
      return
    }

    this.index = Math.min(this.choices.length - 1, Math.max(0, this.index + delta))
  }

  invalidate(): void {
    // Rendered from state every frame.
  }

  render(width: number): string[] {
    return this.choices.map((choice, position) => {
      const active = position === this.index
      const marker = active ? '→' : ' '
      const body = `${marker} ${position + 1} ${choice.label}${choice.note ? ` ${choice.note}` : ''}`

      return truncateToWidth(this.theme.fg(active ? 'accent' : 'text', ` ${body}`), Math.max(1, width), '…')
    })
  }
}

/** Terminal control bytes are shown, never sent onward: a command preview is
 *  text on a screen, not something to hand back to the terminal. */
export function sanitizeForDisplay(text: string): string {
  // eslint-disable-next-line no-control-regex
  return text.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '·')
}
