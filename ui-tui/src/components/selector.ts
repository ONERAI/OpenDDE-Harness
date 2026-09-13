// Who owns the editor slot.
//
// A selector or a broker prompt replaces the editor in place, pi-style, rather
// than opening an overlay. The editor object itself is never rebuilt: its draft,
// cursor, paste markers, autocomplete and history all survive, because the slot
// is a container whose single child is swapped.
//
// The hard part is ownership, not swapping. A selector's RPC can land after the
// user closed it, a timer can tick after a session switch, and a second
// selector can open while the first is still finishing. Each `show` mints a
// token; only calls that still hold the current token may close the slot, so a
// stale `done` cannot restore the editor over a newer selector or steal focus.

import type { Component, Container, TUI } from '@earendil-works/pi-tui'

/** Why a selector stopped. `done` and `cancel` come from the user; the rest are
 *  the app telling the slot that what it was showing no longer applies. */
export type SelectorClose = 'cancel' | 'disconnect' | 'done' | 'replaced' | 'session-switch' | 'shutdown' | 'turn-error'

export interface Selector {
  component: Component
  /** The component that receives keys. Normally the same object, and normally
   *  `Focusable`, so pi places the hardware cursor inside it. */
  focus: Component
  /** Release this instance's resources. Called at most once by the host. */
  dispose(): void
}

export interface SelectorLease {
  /** Still the slot's owner. Guard every asynchronous result with this. */
  isCurrent(): boolean
  /** Close the slot. Harmless after the lease has already ended. */
  done(reason?: SelectorClose): void
  readonly sessionEpoch: number
  readonly sessionId: null | string
  /** Aborted when the lease ends, for in-flight work that can be cancelled. */
  readonly signal: AbortSignal
  readonly token: symbol
}

export interface SelectorHostOptions {
  editor: Component
  /** The slot: its single child is the editor, a selector or a prompt. */
  editorContainer: Container
  /** Told whenever the slot changes hands, so the app can re-evaluate the
   *  queue gate and the key router. */
  onChange: (active: boolean) => void
  session: () => { epoch: number; id: null | string }
  tui: TUI
}

interface ActiveSelector {
  controller: AbortController
  disposed: boolean
  selector: Selector | undefined
  token: symbol
}

export class SelectorHost {
  private current: ActiveSelector | undefined

  constructor(private readonly options: SelectorHostOptions) {}

  get active(): boolean {
    return this.current !== undefined
  }

  /**
   * Put a selector in the slot.
   *
   * Ownership is reserved before the factory runs, so a factory that throws, or
   * one that calls `lease.done()` before it returns, leaves the editor back in
   * the slot rather than an empty container.
   */
  show(create: (lease: SelectorLease) => Selector): void {
    const token = Symbol('selector')
    const controller = new AbortController()
    const { epoch, id } = this.options.session()
    const record: ActiveSelector = { controller, disposed: false, selector: undefined, token }

    const lease: SelectorLease = {
      done: (reason: SelectorClose = 'done') => this.finish(token, reason),
      isCurrent: () => this.current?.token === token,
      sessionEpoch: epoch,
      sessionId: id,
      signal: controller.signal,
      token
    }

    const previous = this.current

    this.current = record
    this.release(previous, 'replaced')

    let created: Selector

    try {
      created = create(lease)
    } catch (err) {
      if (this.current === record) {
        this.current = undefined
        this.release(record, 'cancel')
        this.restoreEditor()
        this.options.onChange(false)
      }

      throw err
    }

    if (this.current !== record) {
      // The factory closed itself (or was replaced) before it returned. The
      // component it handed back was never mounted, so dispose it once here.
      disposeOnce(created)

      return
    }

    record.selector = created
    this.options.editorContainer.clear()
    this.options.editorContainer.addChild(created.component)
    this.options.tui.setFocus(created.focus)
    this.options.onChange(true)
    this.options.tui.requestRender()
  }

  /** Close whatever holds the slot and put the editor back. Idempotent. */
  reset(reason: SelectorClose): void {
    const record = this.current

    this.current = undefined
    this.release(record, reason)
    this.restoreEditor()

    if (record) {
      this.options.onChange(false)
    }
  }

  private finish(token: symbol, reason: SelectorClose): void {
    if (this.current?.token !== token) {
      // A stale `done`: a timer, a late RPC, or a second call from a selector
      // that already closed. It may not touch a newer selector or the editor.
      return
    }

    this.reset(reason)
  }

  /** Invalidate ownership and dispose, in that order, exactly once. */
  private release(record: ActiveSelector | undefined, reason: SelectorClose): void {
    if (!record || record.disposed) {
      return
    }

    record.disposed = true
    record.controller.abort(reason)

    if (record.selector) {
      disposeOnce(record.selector)
    }
  }

  private restoreEditor(): void {
    this.options.editorContainer.clear()
    this.options.editorContainer.addChild(this.options.editor)
    this.options.tui.setFocus(this.options.editor)
    this.options.tui.requestRender()
  }
}

function disposeOnce(selector: Selector): void {
  try {
    selector.dispose()
  } catch {
    // A selector that fails to clean up must not keep the editor off screen.
  }
}
