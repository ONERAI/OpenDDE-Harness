// One selector, several bodies.
//
// A picker's list, its form and its confirmation are stages of the same
// selector, not separate selectors: going back rebuilds the body in place, so
// there is no stack of abandoned pickers to unwind and the slot changes hands
// exactly once.

import type { Component, Focusable } from '@earendil-works/pi-tui'

import { Container } from '@earendil-works/pi-tui'

export interface StageBody extends Component {
  /** Release what this body holds — a typed secret, say — when it is replaced. */
  dispose?(): void
  focused?: boolean
  handleInput?(data: string): void
}

export class StageView extends Container implements Focusable {
  private body: StageBody | undefined
  private isFocused = false

  get current(): StageBody | undefined {
    return this.body
  }

  get focused(): boolean {
    return this.isFocused
  }

  set focused(value: boolean) {
    this.isFocused = value

    if (this.body) {
      this.body.focused = value
    }
  }

  /** Swap the body. The previous one is dropped, not stacked. */
  set(body: StageBody): void {
    if (this.body && this.body !== body) {
      this.body.dispose?.()
    }

    this.body = body
    body.focused = this.isFocused
    this.clear()
    this.addChild(body)
  }

  handleInput(data: string): void {
    this.body?.handleInput?.(data)
  }
}

/**
 * A stage that reads one or two of its own bindings before handing the rest to
 * the view underneath.
 *
 * `intercept` returns true when it consumed the key. Only the stage that shows
 * a binding in its hints reads it, which is why Ctrl+D can mean "remove model"
 * on one stage and "disconnect provider" on another without either leaking.
 */
export function keyedStage(
  view: Focusable & StageBody & { handleInput(data: string): void },
  intercept: (data: string) => boolean
): StageBody {
  return {
    get focused() {
      return view.focused
    },
    set focused(value: boolean) {
      view.focused = value
    },
    handleInput(data: string) {
      if (intercept(data)) {
        return
      }

      view.handleInput(data)
    },
    invalidate: () => view.invalidate(),
    render: (width: number) => view.render(width)
  }
}
