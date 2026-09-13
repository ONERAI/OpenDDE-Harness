// The fullscreen layout: a scrolling document over a fixed input dock.
//
// Modelled on pi's `chat-viewport.ts`. The components are exactly the ones the
// app already owns — the same session panel, the same chat container, the same
// editor slot — so switching renderers is a remount, never a rebuild. Regular
// mode draws those same children straight down the main screen and lets the
// shell's scrollback be the history; fullscreen hands them to a ScrollView and
// pins the dock to the bottom.

import type { Component, ScrollView as ScrollViewType } from '@earendil-works/pi-tui'

import { ScrollView, VStack } from '@earendil-works/pi-tui'

import type { Theme } from './theme.js'

export interface ChatViewportOptions {
  /** The scrolling half: session panel plus the transcript. */
  document: Component
  /** The editor, or whatever is standing in its slot. */
  editor: Component
  footer: Component
  queue: Component
  /** The busy indicator's own row, directly above the editor. */
  status: Component
  taskBar: Component
  theme: Theme
}

export interface ChatViewport {
  /** The layout root handed to `TuiAltScreen.setLayoutRoot`. */
  root: Component
  /** Kept so scroll position and follow-end survive an off/on cycle. */
  transcript: ScrollViewType
}

export function createChatViewport(options: ChatViewportOptions): ChatViewport {
  const t = options.theme
  const transcript = new ScrollView(options.document, {
    follow: 'end',
    overscroll: 'chain',
    primary: true,
    scrollbar: 'auto',
    scrollbarThumbStyle: text => t.fg('border', text),
    scrollbarTrackStyle: text => t.fg('muted', text)
  })

  // On a short terminal the bar, the queue and the footer give way first; the
  // editor keeps three rows, which is its border plus one line to type on. The
  // status row is one line or none, so there is nothing in it to shrink.
  const dock = new VStack([
    { component: options.taskBar, minSize: 0, shrink: 1 },
    { component: options.queue, minSize: 0, shrink: 1 },
    { component: options.status, minSize: 0, shrink: 0 },
    { component: options.editor, minSize: 3, shrink: 1 },
    { component: options.footer, minSize: 0, shrink: 1 }
  ])

  return {
    root: new VStack([
      { basis: 0, component: transcript, grow: 1, minSize: 1, shrink: 1 },
      { basis: 'auto', component: dock, grow: 0, minSize: 1, shrink: 1 }
    ]),
    transcript
  }
}
