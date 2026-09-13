// What a selector is given. Everything is explicit: there is no global gateway,
// no ambient store and no reach into the app object, so each selector runs in a
// test against a fake transport and a fake terminal.

import type { KeybindingsManager, TUI } from '@earendil-works/pi-tui'

import type { SystemTone } from '../components/systemLine.js'
import type { Gateway, LoginStepPush } from '../gateway.js'
import type { Theme } from '../theme.js'

export interface SelectorSession {
  effort?: string
  epoch: number
  id: null | string
  model?: string
  provider?: string
}

export interface TranscriptSink {
  print(text: string, tone?: SystemTone): void
  printBlock(text: string, title?: string): void
}

export interface SelectorDeps {
  gateway: Gateway
  keybindings: KeybindingsManager
  /** Hear every step of a running sign-in. Returns the unsubscribe: the steps
   *  are pushed to the whole app, and only the stage that started the login
   *  wants them. */
  onLoginStep(handler: (step: LoginStepPush) => void): () => void
  /** Open a sign-in URL in the browser, as pi's login dialog does. */
  openBrowser(url: string): void
  /** The live session, read at the moment it is asked for. Selectors capture
   *  what they need at submission time rather than reading it again later. */
  session(): SelectorSession
  theme: Theme
  transcript: TranscriptSink
  tui: TUI
}

/** What an RPC failure should say. Never includes `error.data`: the model
 *  validation error echoes its input, which can be a key that was just typed. */
export function selectorError(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}
