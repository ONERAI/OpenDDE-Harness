import type { Component } from '@earendil-works/pi-tui'

import { Container, Spacer } from '@earendil-works/pi-tui'

import type { Theme, ThemeToken } from '../theme.js'

import { ThemedLine, ThemedText } from './themedText.js'

export type SystemTone = 'error' | 'muted' | 'ok' | 'warn'

/** pi's own: a status is `dim`, an error and a warning take their colour.
 *  `muted` names the tone, not the token -- pi writes a status in the grey one
 *  step below `muted`, the same one it paints its footer in. */
const TONE_TOKEN: Record<SystemTone, ThemeToken> = {
  error: 'error',
  muted: 'dim',
  ok: 'ok',
  warn: 'warn'
}

/** The inset every other block in the transcript sits at: the assistant's
 *  replies, the user's band, the queue. A note flush against the left edge
 *  reads as belonging to the frame rather than to the conversation. */
const INSET = 1

/**
 * A note with the blank row pi puts above one.
 *
 * pi writes every note into its transcript as a `Spacer(1)` and then the text
 * (`showStatus`, `showError`, its tool statuses). Every other block here
 * carries its own leading blank -- a reply, the user's band -- so a note
 * without one was the single thing in the transcript that sat on the line
 * before it, and a run of them read as one paragraph.
 */
function spaced(note: Component): Component {
  const box = new Container()

  box.addChild(new Spacer(1))
  box.addChild(note)

  return box
}

/** A one-line note in the transcript (tool activity, status). Truncated rather
 *  than wrapped so a long tool argument never takes over the screen. The text
 *  is stored raw and coloured at render time, so a late theme swap repaints it. */
export function systemLine(theme: Theme, text: string, tone: SystemTone = 'muted'): Component {
  return spaced(new ThemedLine(painted => theme.fg(TONE_TOKEN[tone], painted), text.replace(/\s+/g, ' ').trim(), INSET))
}

/** A multi-line note (errors, setup instructions), wrapped to the width. */
export function systemBlock(theme: Theme, text: string, tone: SystemTone = 'muted'): Component {
  return spaced(systemBody(theme, text, tone))
}

/** The body of a block, without the blank row: for a caller that draws its own
 *  header above it and spaces the two together. */
export function systemBody(theme: Theme, text: string, tone: SystemTone = 'muted'): Component {
  return new ThemedText(painted => theme.fg(TONE_TOKEN[tone], painted), text, INSET)
}
