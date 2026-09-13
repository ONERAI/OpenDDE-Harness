// Terminal window title. `Terminal.setTitle` exists on pi-tui's terminal but is
// a no-op on the fake used in tests, so the app talks to this instead of the
// escape sequence.

import type { Terminal } from '@earendil-works/pi-tui'

/** Set the window/tab title, ignoring terminals that do not support it. */
export function setTerminalTitle(terminal: Pick<Terminal, 'setTitle'>, title: string): void {
  try {
    terminal.setTitle(title)
  } catch {
    // A terminal that refuses a title is not worth a message.
  }
}
