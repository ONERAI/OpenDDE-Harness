// Editing a message that is already queued.
//
// A prompt typed while a turn is running waits in the queue, and by the time
// the turn ends it is often not what you would send any more. Alt+Up pulls the
// last queued message back into the editor; Enter puts the edited text back in
// the same place rather than appending a second copy; Alt+X drops it; Escape
// leaves it exactly as it was and gives back whatever draft was displaced.
//
// Two things make this more than a text swap. The message being edited is
// reserved from dispatch, because a turn that ends mid-edit would otherwise
// send the text the user is in the middle of replacing and strand the draft it
// displaced. And the draft is captured expanded, because `Editor.setText`
// clears the paste table: a draft holding a large paste would come back as a
// marker with nothing behind it.

import type { MessageQueue } from './components/queue.js'

/** Bracketed paste, which is how a paste marker and its contents are both
 *  rebuilt: there is no public API for putting an entry back in the editor's
 *  paste table, but there is the input path that filled it in the first place. */
const PASTE_START = '\x1b[200~'
const PASTE_END = '\x1b[201~'

/** A draft that is nothing but one large paste, as the editor spells it. */
const PASTE_MARKER_ONLY = /^\[paste #\d+(?: (?:\+\d+ lines|\d+ chars))?\]$/

/** The editor surface this needs; the real `HarnessEditor` satisfies it. */
export interface EditableInput {
  /** What submitting would send: paste markers replaced by their contents. */
  getExpandedText(): string
  /** What is on screen, paste markers and all. */
  getText(): string
  handleInput(data: string): void
  insertTextAtCursor(text: string): void
  setText(text: string): void
}

/** What the editor held when the edit started. The two differ exactly when the
 *  draft contained a paste large enough for the editor to have collapsed it. */
interface Draft {
  display: string
  expanded: string
}

export interface QueueEditingOptions {
  editor: EditableInput
  /** Called after anything changed, so the app can request a render. */
  onChange?: () => void
  /** Called when the edit lets go of its message. The queue may have been
   *  waiting on it: a turn can end while the head is reserved, and the drain
   *  that would have run then has to run now. */
  onReleased?: () => void
  queue: MessageQueue
}

export class QueueEditing {
  /** The draft that was in the editor when the edit started. */
  private draft: Draft | null = null
  private readonly editor: EditableInput
  private readonly onChange: (() => void) | undefined
  private readonly onReleased: (() => void) | undefined
  private readonly queue: MessageQueue

  constructor(options: QueueEditingOptions) {
    this.editor = options.editor
    this.onChange = options.onChange
    this.onReleased = options.onReleased
    this.queue = options.queue
  }

  get active(): boolean {
    return this.queue.editing !== null
  }

  /**
   * Step through the queue, newest first for `-1` and oldest first for `1`.
   *
   * Wraps at both ends, like the old app's queue cycling, and stops being a
   * no-op the moment the queue empties. Returns whether anything happened.
   */
  cycle(direction: -1 | 1): boolean {
    this.reconcile()

    const length = this.queue.length

    if (length === 0) {
      return false
    }

    const current = this.queue.editing

    if (current === null) {
      this.draft = this.capture()
    }

    const next = current === null ? (direction > 0 ? 0 : length - 1) : (current + direction + length) % length

    this.queue.setEditing(next)
    this.editor.setText(this.queue.list()[next] ?? '')
    this.onChange?.()

    return true
  }

  /** Put the edited text back where it came from. Returns whether it applied,
   *  so a plain submit with no edit open still goes through the queue. */
  apply(text: string): boolean {
    this.reconcile()

    const index = this.queue.editing

    if (index === null) {
      return false
    }

    const content = text.trim()

    // An edit emptied down to nothing means "drop it", not "queue a blank".
    if (!content) {
      this.queue.remove(index)
    } else {
      this.queue.replace(index, content)
    }

    this.finish()

    return true
  }

  /** Alt+X: drop the message being edited. */
  drop(): boolean {
    this.reconcile()

    const index = this.queue.editing

    if (index === null) {
      return false
    }

    this.queue.remove(index)
    this.finish()

    return true
  }

  /** Escape: leave the queued message alone and take the draft back. */
  cancel(): boolean {
    this.reconcile()

    if (this.queue.editing === null) {
      return false
    }

    this.finish()

    return true
  }

  /** The queue went away under the edit — a session switch, a clear. The draft
   *  goes with it: every caller of this is clearing the editor anyway. */
  reset(): void {
    this.draft = null
    this.queue.setEditing(null)
  }

  /**
   * The queue let go of the edit without going through apply, drop or cancel.
   *
   * Nothing should do that any more — the head is reserved while it is being
   * edited, and every other index moves with its message — but the draft is
   * the user's text and it is not allowed to disappear because some future
   * queue transition forgot about it.
   */
  private reconcile(): void {
    if (this.draft === null || this.queue.editing !== null) {
      return
    }

    this.restore(this.draft)
    this.draft = null
    this.onChange?.()
  }

  private capture(): Draft {
    return { display: this.editor.getText(), expanded: this.editor.getExpandedText() }
  }

  /**
   * Put the draft back in the editor.
   *
   * `setText` clears the editor's paste table, so a draft that held a large
   * paste cannot simply be written back: the marker would return with nothing
   * behind it, and submitting it would send `[paste #1 +20 lines]` rather than
   * the twenty lines. A draft that was exactly one paste goes back through the
   * paste path, which rebuilds the marker and its contents together. A draft
   * that mixed prose with a paste goes back expanded: longer on screen than it
   * was, but every character of it is there and editable.
   */
  private restore(draft: Draft): void {
    if (draft.display === draft.expanded) {
      this.editor.setText(draft.display)

      return
    }

    this.editor.setText('')

    if (PASTE_MARKER_ONLY.test(draft.display.trim())) {
      this.editor.handleInput(`${PASTE_START}${draft.expanded}${PASTE_END}`)

      return
    }

    this.editor.setText(draft.expanded)
  }

  private finish(): void {
    const draft = this.draft

    this.queue.setEditing(null)
    this.draft = null
    this.restore(draft ?? { display: '', expanded: '' })
    this.onChange?.()
    this.onReleased?.()
  }
}
