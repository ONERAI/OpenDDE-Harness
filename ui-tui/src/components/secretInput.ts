// A one-line input for API keys: pi's `Input` for the buffer, paste handling and
// submit, with masked rendering and a deliberately small key set on top.
//
// pi's Input has no password option, so this subclass supplies one without
// reaching into its private fields: only the keys that keep the cursor at the
// end of the value are forwarded (type, paste, backspace, Ctrl+U, Enter, Esc),
// and `render` draws one bullet per character instead of the value.
//
// What this does not claim: a JavaScript string cannot be wiped, so `clear()`
// drops the reference and the component, and nothing more. The value never
// reaches the transcript, the prompt history or an error message.

import {
  CURSOR_MARKER,
  decodeKittyPrintable,
  getKeybindings,
  Input,
  truncateToWidth,
  visibleWidth
} from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

const PASTE_START = '\x1b[200~'
const PASTE_END = '\x1b[201~'
const MASK = '•'

/** Printable input: a plain character run, or the kitty protocol's encoding of
 *  one. Anything with an escape or a control byte in it is a key, not text. */
export function isPrintableInput(data: string): boolean {
  if (decodeKittyPrintable(data) !== undefined) {
    return true
  }

  if (!data) {
    return false
  }

  for (const char of data) {
    const code = char.codePointAt(0) ?? 0

    if (code < 0x20 || code === 0x7f) {
      return false
    }
  }

  return true
}

/** Whitespace around a pasted key is an artefact; a control byte inside it is
 *  not something we are willing to send. */
export function validateSecret(raw: string): { error: string } | { value: string } {
  const value = raw.trim()

  // eslint-disable-next-line no-control-regex
  if (/[\x00-\x1f\x7f]/.test(value)) {
    return { error: 'that value contains control characters' }
  }

  return { value }
}

export interface SecretInputOptions {
  placeholder?: string
  prompt?: string
  theme: Theme
}

export class SecretInput extends Input {
  private pasting = false
  private readonly secretPrompt: string
  private readonly secretPlaceholder: string
  private readonly theme: Theme

  constructor(options: SecretInputOptions) {
    super({ prompt: options.prompt ?? '> ' })
    this.secretPrompt = options.prompt ?? '> '
    this.secretPlaceholder = options.placeholder ?? ''
    this.theme = options.theme
  }

  /** Drop what was typed. The component is discarded with the stage that owns it. */
  clear(): void {
    this.setValue('')
  }

  override handleInput(data: string): void {
    if (this.pasting || data.includes(PASTE_START)) {
      this.pasting = !data.includes(PASTE_END)
      // pi buffers the paste and strips its newlines, so a pasted key with a
      // trailing newline types the key instead of submitting the form.
      super.handleInput(data)

      return
    }

    const kb = getKeybindings()

    if (
      kb.matches(data, 'tui.select.cancel') ||
      kb.matches(data, 'tui.input.submit') ||
      data === '\n' ||
      kb.matches(data, 'tui.editor.deleteCharBackward') ||
      kb.matches(data, 'tui.editor.deleteToLineStart') ||
      isPrintableInput(data)
    ) {
      super.handleInput(data)

      return
    }

    // Cursor moves, word kills, yank and undo would put the cursor somewhere
    // the mask cannot show; the value is append-only instead.
  }

  override render(width: number): string[] {
    if (width <= 0) {
      return ['']
    }

    const prompt = this.theme.fg('muted', this.secretPrompt)
    const room = width - visibleWidth(this.secretPrompt)

    if (room <= 0) {
      return [truncateToWidth(prompt, width, '')]
    }

    const cursor = this.focused ? CURSOR_MARKER : ''
    const length = [...this.getValue()].length

    if (length === 0 && this.secretPlaceholder) {
      return [prompt + cursor + this.theme.fg('muted', truncateToWidth(this.secretPlaceholder, room, ''))]
    }

    // One bullet per character, right-anchored so the end of a long key stays
    // visible — the same place the cursor is.
    const masked = MASK.repeat(Math.min(length, Math.max(0, room - 1)))

    return [prompt + this.theme.fg('text', masked) + cursor]
  }
}
