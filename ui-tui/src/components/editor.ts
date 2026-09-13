// The prompt editor: pi's `Editor` with a placeholder in the empty prompt,
// persisted prompt history, and the external editor handoff.
//
// The turn's status is not in here. It used to be drawn into the top border,
// pi-style; the owner asked for it on its own line above the editor instead
// (`components/statusIndicator.ts`), so the border is pi's plain rule again and
// carries nothing but the scroll indicator.
//
// Key handling stays out of here. The app installs one global input listener
// (`lib/keybindings.ts`), which runs before the focused component, so this class
// only has to render and to expose the state that listener asks about.

import type { AutocompleteProvider, EditorOptions, EditorTheme, TUI } from '@earendil-works/pi-tui'

import { Editor, truncateToWidth, visibleWidth } from '@earendil-works/pi-tui'

import type { InputHistory } from '../lib/history.js'

import { editExternally, resolveEditor } from '../lib/externalEditor.js'

export interface HarnessEditorOptions extends EditorOptions {
  /** Slash-command and path completion. Installed here so the app makes one call. */
  autocomplete?: AutocompleteProvider
  /** Persisted prompt history, seeded into the editor and appended on submit. */
  history?: InputHistory
  /** Shown in the empty editor. One is drawn per session, never per keystroke. */
  placeholder?: string
}

/** The reverse-video block pi draws for the cursor. The empty editor's only
 *  content is this, which is what makes the placeholder easy to splice in
 *  after it without rebuilding the line and losing pi's IME cursor marker. */
const CURSOR_BLOCK = '\u001b[7m \u001b[0m'

export class HarnessEditor extends Editor {
  private placeholder: string
  private placeholderPaint: (text: string) => string = text => text
  private readonly promptHistory?: InputHistory

  constructor(tui: TUI, theme: EditorTheme, options: HarnessEditorOptions = {}) {
    const { autocomplete, history, placeholder, ...editorOptions } = options

    super(tui, theme, editorOptions)
    this.placeholder = placeholder ?? ''
    this.promptHistory = history

    if (autocomplete) {
      this.setAutocompleteProvider(autocomplete)
    }

    // Oldest first: `addToHistory` unshifts, so the newest entry ends up as the
    // first one Up reaches.
    for (const entry of history?.load() ?? []) {
      this.addToHistory(entry)
    }
  }

  /**
   * What the empty editor suggests, and how to paint it.
   *
   * The paint function comes from the app rather than the theme directly: this
   * class is handed a pi `EditorTheme`, which knows about borders and the
   * completion list and nothing else.
   */
  setPlaceholder(text: string, paint?: (text: string) => string): void {
    this.placeholder = text

    if (paint) {
      this.placeholderPaint = paint
    }

    this.tui.requestRender()
  }

  /** Record a sent prompt in both the in-memory ring and the history file. */
  rememberPrompt(text: string): void {
    this.addToHistory(text)
    this.promptHistory?.append(text)
  }

  /**
   * Hand the prompt to $VISUAL / $EDITOR and take back what was saved.
   *
   * `preserveScreen` leaves the rendered transcript where it is rather than
   * printing a closing newline under it, so opening the editor twice does not
   * push the conversation apart. `editExternally` holds signals for the
   * lifetime of the child, which is what stops Ctrl+C in vim from ending the
   * session. Returns whether the text changed.
   */
  async openExternal(argv: string[] = resolveEditor()): Promise<boolean> {
    const before = this.getExpandedText()

    this.tui.stop({ preserveScreen: true })

    try {
      const result = await editExternally(before, argv)

      if (result.status !== 'complete' || result.content === before) {
        return false
      }

      this.setText(result.content)

      return true
    } finally {
      this.tui.start()
      this.tui.requestRender(true)
    }
  }

  /**
   * The editor, with the placeholder laid over the empty first line.
   *
   * Spliced into the line pi rendered rather than rebuilt from scratch: the
   * line carries a zero-width marker that tells the terminal where to put the
   * hardware cursor for an IME candidate window, and rebuilding it would drop
   * that. Everything after the cursor block is blank padding of a known width,
   * so the placeholder replaces exactly as many columns as it takes.
   */
  override render(width: number): string[] {
    const lines = super.render(width)

    if (!this.placeholder || lines.length < 2 || this.getText().length > 0) {
      return lines
    }

    const line = lines[1]!
    const cursorEnd = line.lastIndexOf(CURSOR_BLOCK)

    if (cursorEnd === -1) {
      return lines
    }

    const head = line.slice(0, cursorEnd + CURSOR_BLOCK.length)
    const blanks = line.slice(cursorEnd + CURSOR_BLOCK.length)

    // Only ever paint over blanks. Anything else means pi changed how it draws
    // an empty line, and the editor is better off plain than corrupted.
    if (blanks.length === 0 || blanks.trim().length > 0) {
      return lines
    }

    const text = truncateToWidth(this.placeholder, blanks.length, '')
    const painted = this.placeholderPaint(text)

    lines[1] = head + painted + ' '.repeat(blanks.length - visibleWidth(text))

    return lines
  }
}
