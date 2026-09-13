// Which screen the UI draws on, and how it changes its mind at runtime.
//
// pi has two renderers over one terminal: `TuiMainScreen`, which writes into the
// shell's own scrollback, and `TuiAltScreen`, which owns the alternate buffer
// and brings its own scrollback, search, mouse selection and jump-to-latest. The
// same component tree serves both — the switch stops one renderer, hands its
// children to the other and starts that, and nothing in the transcript is
// rebuilt.
//
// The catch is that components capture the TUI they were built with. So nothing
// outside this module ever holds a concrete renderer: they are given
// `RendererOwner.reference`, a forwarding object whose properties *and*
// already-captured methods resolve to whichever renderer is current.

import type { Terminal, TUI, TuiMainScreenRenderState } from '@earendil-works/pi-tui'

import { TuiAltScreen, TuiMainScreen } from '@earendil-works/pi-tui'

import type { Theme } from './theme.js'

export type RendererMode = 'fullscreen' | 'regular'

export type HarnessRenderer = TuiAltScreen | TuiMainScreen

/** Wheel lines per notch in fullscreen, before clamping. */
const DEFAULT_SCROLL_LINES = 3
const MIN_SCROLL_LINES = 1
const MAX_SCROLL_LINES = 20

export interface RendererOptions {
  /** Copy the fullscreen selection. Returns whether the clipboard took it. */
  copySelection?: (text: string) => Promise<boolean>
  /** `OPENDDE_HARNESS_TUI_DISABLE_MOUSE` / `_SCROLL_SPEED` live here. */
  env?: NodeJS.ProcessEnv
  showHardwareCursor?: boolean
}

/** The words that turn the default off. Everything else, the variable being
 *  absent included, leaves it on. */
const OFF = new Set(['0', 'false', 'no', 'off'])

/**
 * Fullscreen, unless the environment asks for the main screen.
 *
 * `OPENDDE_HARNESS_TUI_FULLSCREEN=0`, `false`, `no` or `off` starts on the main
 * screen, where the shell's own scrollback is the history and the terminal owns
 * selection and scrolling. `/fullscreen` moves between the two at runtime for
 * the rest of the process either way.
 */
export function initialRendererMode(env: NodeJS.ProcessEnv = process.env): RendererMode {
  return OFF.has((env.OPENDDE_HARNESS_TUI_FULLSCREEN ?? '').trim().toLowerCase()) ? 'regular' : 'fullscreen'
}

/** Mouse capture is on unless the environment turns it off. Keyboard
 *  navigation works either way; turning it off gives the terminal its own
 *  selection and scroll back. */
export function mouseEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  const raw = (env.OPENDDE_HARNESS_TUI_DISABLE_MOUSE ?? '').trim().toLowerCase()

  return !(raw === '1' || raw === 'true' || raw === 'yes')
}

/**
 * Logical wheel lines for the fullscreen viewport.
 *
 * Main-screen scrolling belongs to the terminal, so this applies to fullscreen
 * only. A finite positive number is rounded and clamped; anything else keeps
 * the default. The old `CLAUDE_CODE_SCROLL_SPEED` name is still read when the
 * current one is absent.
 */
export function scrollLines(env: NodeJS.ProcessEnv = process.env): number {
  const raw = (env.OPENDDE_HARNESS_TUI_SCROLL_SPEED ?? env.CLAUDE_CODE_SCROLL_SPEED ?? '').trim()
  const value = Number(raw)

  if (!raw || !Number.isFinite(value) || value <= 0) {
    return DEFAULT_SCROLL_LINES
  }

  return Math.min(MAX_SCROLL_LINES, Math.max(MIN_SCROLL_LINES, Math.round(value)))
}

export function createRenderer(
  mode: RendererMode,
  terminal: Terminal,
  theme: Theme,
  options: RendererOptions = {}
): HarnessRenderer {
  const env = options.env ?? process.env
  const showCursor = options.showHardwareCursor ?? true

  if (mode === 'regular') {
    return new TuiMainScreen(terminal, showCursor)
  }

  const match = (text: string) => theme.bg('selectionBg', theme.fg('text', text))

  return new TuiAltScreen(terminal, showCursor, undefined, {
    mouse: mouseEnabled(env),
    scrollToEndIndicator: () => theme.bg('selectionBg', theme.fg('text', ' ↓ Jump to latest · end ')),
    searchCurrentMatchStyle: text => theme.bold(match(text)),
    searchMatchStyle: match,
    searchNavigationButtonStyle: text => theme.fg('muted', text),
    wheelScrollLines: scrollLines(env),
    ...(options.copySelection ? { copySelection: options.copySelection } : {})
  })
}

/**
 * A TUI that always means the current renderer.
 *
 * Property reads forward. Method reads forward *and* rebind: a component that
 * pulled `requestRender` out once at construction still reaches the renderer
 * that is on screen now, which is the whole point — otherwise a switch would
 * leave half the tree drawing into a stopped instance. Ported from pi's
 * `createInteractiveTuiReference`.
 */
export function createTuiReference(getRenderer: () => TUI): TUI {
  return new Proxy({} as TUI, {
    get: (_target, property) => {
      const tui = getRenderer()
      const value = Reflect.get(tui, property, tui)

      if (typeof value !== 'function') {
        return value
      }

      let boundTo = tui
      let method = value

      return (...args: unknown[]) => {
        const current = getRenderer()

        if (current !== boundTo) {
          const replacement = Reflect.get(current, property, current)

          if (typeof replacement !== 'function') {
            throw new TypeError(`TUI property ${String(property)} is not callable`)
          }

          boundTo = current
          method = replacement
        }

        return Reflect.apply(method as (...rest: unknown[]) => unknown, boundTo, args)
      }
    },
    getPrototypeOf: () => Reflect.getPrototypeOf(getRenderer()),
    has: (_target, property) => Reflect.has(getRenderer(), property),
    set: (_target, property, value) => Reflect.set(getRenderer(), property, value, getRenderer())
  })
}

export interface RendererOwnerOptions extends RendererOptions {
  mode?: RendererMode
  terminal: Terminal
  theme: Theme
}

/** Why a switch was refused, or null when it happened. */
export type SwitchRefusal = null | string

export class RendererOwner {
  /** Hand this to everything that renders. It never goes stale. */
  readonly reference: TUI

  /** Called after a switch, so listeners registered on the old renderer are
   *  registered again on the new one. */
  onSwitched: (() => void) | undefined

  private readonly options: RendererOwnerOptions
  private current: HarnessRenderer
  /** The fullscreen layout root, kept across switches so scroll position and
   *  follow-end survive an off/on cycle. */
  private layoutRootValue: Parameters<TuiAltScreen['setLayoutRoot']>[0]
  /** The main screen's differential-render state, restored when we come back
   *  to it, so returning from fullscreen does not repaint the whole shell. */
  private disposed = false
  private mainScreenState: TuiMainScreenRenderState | undefined
  private switching = false

  constructor(options: RendererOwnerOptions) {
    this.options = options
    this.current = createRenderer(options.mode ?? 'regular', options.terminal, options.theme, options)
    this.reference = createTuiReference(() => this.current)
  }

  get mode(): RendererMode {
    return this.current.mode
  }

  /** The renderer on screen. Only the switch and the app's shutdown need it. */
  get renderer(): HarnessRenderer {
    return this.current
  }

  /**
   * Move to `mode`, keeping every component instance.
   *
   * `start` is false for the last switch before exiting: pi renders once into
   * the shell and stops, rather than opening a second input loop nobody will
   * type into.
   */
  switchTo(mode: RendererMode, start = true): SwitchRefusal {
    if (mode === this.current.mode) {
      return null
    }

    if (this.switching) {
      return 'a renderer switch is already running'
    }

    if (this.current.hasOverlayEntries) {
      return 'close the search or overlay first'
    }

    this.switching = true

    const previous = this.current
    const children = [...previous.children]
    const focus = previous.getFocusedComponent()
    const layoutRoot = this.layoutRoot
    // Held outside the try so the catch can unwind it. A renderer that failed
    // to start has usually already taken the terminal: pi enters the alternate
    // screen, turns mouse reporting on and hides the cursor before it calls
    // `Terminal.start`, so a throw from there leaves the screen in fullscreen
    // with no renderer that believes it is.
    let next: HarnessRenderer | undefined

    try {
      if (previous instanceof TuiMainScreen) {
        this.mainScreenState = previous.captureRenderState()
      }

      previous.stop({ preserveScreen: true })
      previous.setFocus(null)
      previous.clear()

      if (previous instanceof TuiAltScreen) {
        previous.setLayoutRoot(undefined)
      }

      next = createRenderer(mode, previous.terminal, this.options.theme, this.options)

      next.setClearOnShrink(previous.getClearOnShrink())
      next.onDebug = previous.onDebug

      if (next instanceof TuiMainScreen && this.mainScreenState) {
        next.restoreRenderState(this.mainScreenState)
      }

      this.current = next
      this.mount(children, layoutRoot)
      next.invalidate()
      next.setFocus(focus)

      if (start) {
        next.start()
        this.onSwitched?.()
      }

      return null
    } catch (err) {
      // Terminal ownership may never be ambiguous. Stop the half-started
      // renderer first — that is what leaves the alternate screen, turns the
      // mouse off and shows the cursor again — and only then give the screen,
      // the tree and the focus back to the one that had them.
      this.unwind(next)
      this.current = previous
      this.mount(children, layoutRoot)
      previous.setFocus(focus)

      if (start) {
        previous.start()
        this.onSwitched?.()
      }

      return err instanceof Error ? err.message : String(err)
    } finally {
      this.switching = false
    }
  }

  /**
   * Give back everything a renderer took before it failed.
   *
   * `preserveScreen` because there is nothing to hand over: the previous
   * renderer is about to repaint, and dumping a transcript the failed renderer
   * never drew would print it twice. Both steps are guarded — a renderer that
   * cannot even be stopped must not stop the rollback.
   */
  private unwind(renderer: HarnessRenderer | undefined): void {
    if (!renderer) {
      return
    }

    try {
      renderer.stop({ preserveScreen: true })
    } catch {
      // Nothing further to give back.
    }

    try {
      renderer.setFocus(null)
      renderer.clear()

      if (renderer instanceof TuiAltScreen) {
        renderer.setLayoutRoot(undefined)
      }
    } catch {
      // The tree is about to be remounted on the previous renderer regardless.
    }
  }

  get layoutRoot(): Parameters<TuiAltScreen['setLayoutRoot']>[0] {
    return this.layoutRootValue
  }

  setLayoutRoot(root: Parameters<TuiAltScreen['setLayoutRoot']>[0]): void {
    this.layoutRootValue = root

    if (this.current instanceof TuiAltScreen) {
      this.current.setLayoutRoot(root)
    }
  }

  /**
   * Leave fullscreen for good, with the conversation in the shell's scrollback.
   *
   * Stopping the alternate screen with `preserveScreen: false` replays its last
   * *frame* — one viewport — which is all pi promises by "preserve the screen".
   * A conversation is longer than a frame, so leaving it behind means taking
   * the route pi's own exit takes: hand the same component tree to the main
   * screen and render it once there, where every line is written into the
   * shell's buffer rather than into an alternate one that is about to vanish.
   *
   * Returns whether it moved, so a caller that has not rendered yet knows it
   * still has to.
   */
  exitToMainScreen(): boolean {
    if (this.current.mode === 'regular') {
      return false
    }

    // A switch is refused while an overlay is up, and the search overlay is
    // exactly the kind of thing that is still open when someone quits.
    while (this.current.hasOverlayEntries) {
      this.current.hideOverlay()
    }

    // Not started: nothing is going to be typed into it, and a second input
    // loop on the way out is how a terminal ends up with two readers.
    return this.switchTo('regular', false) === null
  }

  dispose(): void {
    // Called from the boot failure path, from the app's quit and from the
    // signal cleanups, and more than one of those can run for a single exit.
    if (this.disposed) {
      return
    }

    this.disposed = true

    if (this.exitToMainScreen()) {
      // The tree moved but was deliberately not started, so nothing has drawn
      // it yet. This is the render that puts it in the scrollback.
      this.current.renderNow()
    }

    this.current.stop()
  }

  private mount(children: readonly Parameters<TUI['addChild']>[0][], layoutRoot: this['layoutRoot']): void {
    this.current.clear()

    for (const child of children) {
      this.current.addChild(child)
    }

    if (this.current instanceof TuiAltScreen) {
      this.current.setLayoutRoot(layoutRoot)
    }
  }
}
