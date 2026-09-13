// The application: it owns the pi-tui component tree, the session, and the
// pieces that talk to each other — the turn model streaming into the chat, the
// editor above the footer, the queue between them, the commands, the keys.
// Everything it needs from the outside (gateway, terminal, theme, environment,
// how to exit) is injected, so the whole boot path runs headless in tests.
//
// Layout, top to bottom: session panel → transcript → queued messages →
// the busy indicator's own row → editor → footer.

import type { TUI } from '@earendil-works/pi-tui'
import type { Container } from '@earendil-works/pi-tui'

import type { CommandContext } from './commands/index.js'
import type { CountdownClock } from './components/countdownTimer.js'
import type { SystemTone } from './components/systemLine.js'
import type { Gateway, GatewayNotification } from './gateway.js'
import type { InputHistory } from './lib/history.js'
import type { GlobalKeyHandler } from './lib/keybindings.js'
import type { RendererOwner } from './renderer.js'
import type { JsonValue } from './rpc/index.js'
import type { Theme } from './theme.js'
import type { UsageTotals } from './turn.js'

import { CommandSurface } from './app/commandSurface.js'
import { Layout } from './app/layout.js'
import { SessionFacts } from './app/sessionFacts.js'
import { SessionHub } from './app/sessionHub.js'
import { AppShell, listenSigwinch } from './app/shell.js'
import { TranscriptView } from './app/transcript.js'
import { TurnRunner } from './app/turnRunner.js'
import { dispatch, looksLikeCommand, toggleYolo } from './commands/index.js'
import { systemClock } from './components/countdownTimer.js'
import { HarnessEditor } from './components/editor.js'
import { MessageQueue } from './components/queue.js'
import { pickPlaceholder } from './content/placeholders.js'
import { SETUP_REQUIRED_LINES } from './content/setup.js'
import { createGlobalKeyHandler, createKeybindings, QUIT_ARM_MS, QUIT_HINT } from './lib/keybindings.js'
import { Phase2 } from './phase2.js'
import { Phase3 } from './phase3.js'
import { applyConfiguredTheme, THEME_NAMES } from './theme.js'
import { TurnController } from './turn.js'

/** SIGWINCH bursts while a window is dragged; report the size once it settles. */
const RESIZE_DEBOUNCE_MS = 100

export interface HarnessAppOptions {
  /** Prompt countdowns read the time from here; tests inject a fake. */
  clock?: CountdownClock
  /** Where the git branch is read from. Defaults to the session's cwd. */
  cwd?: string
  env?: NodeJS.ProcessEnv
  gateway: Gateway
  /** Persisted prompt history; omitted in tests. */
  history?: InputHistory
  /** Subscribe to terminal resizes. Defaults to SIGWINCH; tests pass a stub. */
  listenResize?: (handler: () => void) => () => void
  /** Epoch milliseconds, for the status line's elapsed count. Tests inject
   *  one so the seconds it draws are exact. */
  now?: () => number
  /** Called to end the process. Defaults to a no-op so tests never exit. */
  onExit?: (code: number) => void
  /** The renderer owner, when this process has one. Without it the app draws
   *  on whatever `tui` it was given and `/fullscreen` says so. */
  renderer?: RendererOwner
  theme: Theme
  tui: TUI
}

export class HarnessApp {
  private readonly transcript: TranscriptView
  /** The transcript's own container, held here because so much of the wiring
   *  hands it straight to a collaborator. */
  private readonly chat: Container
  private readonly ctx: CommandContext
  private readonly editor: HarnessEditor
  private readonly env: NodeJS.ProcessEnv
  private readonly gateway: Gateway
  private readonly layout: Layout
  private readonly keys: GlobalKeyHandler
  private readonly now: () => number
  private readonly onExit: (code: number) => void
  private readonly phase2: Phase2
  private readonly phase3: Phase3
  private readonly queue: MessageQueue
  private readonly commands: CommandSurface
  private readonly facts: SessionFacts
  private readonly sessions: SessionHub
  private readonly shell: AppShell
  private readonly theme: Theme
  private readonly tui: TUI
  private readonly turn: TurnController
  private readonly turns: TurnRunner

  private usage: null | UsageTotals = null

  constructor(opts: HarnessAppOptions) {
    this.gateway = opts.gateway
    this.theme = opts.theme
    this.tui = opts.tui
    this.env = opts.env ?? process.env
    this.now = opts.now ?? Date.now
    this.onExit = opts.onExit ?? (() => {})
    this.transcript = new TranscriptView(this.theme, this.tui)
    this.chat = this.transcript.container
    // One clock for everything that waits: prompt countdowns, the task
    // monitor's polling and the tool panels' reassurance rows. Tests inject a
    // manual one so none of them ever sleep.
    const clock = opts.clock ?? systemClock()

    this.turn = new TurnController({ chat: this.chat, clock, gateway: this.gateway, theme: this.theme, tui: this.tui })
    this.turn.onStatus = status => this.turns.onTurnStatus(status)
    this.turn.onUsage = usage => {
      this.usage = usage
      this.tui.requestRender()
    }
    // Next macrotask, not now: the gateway drops its active-turn slot while
    // unwinding the turn this event ended, and a send that lands first is
    // refused. A refusal is handled too (see `startTurn`); this just avoids
    // paying that round trip on every turn.
    this.turn.onIdle = () => {
      this.turns.scheduleDrain(0)
      // Anything that was still connecting when the session opened has
      // settled by the time a turn ends, and nothing else says so.
      this.facts.settleAfterTurn()
    }
    // A turn that stored nothing leaves no exchange to undo, so `/retry` of
    // that prompt must resend it rather than undo the exchange before it.
    this.turn.onUnsaved = content => this.transcript.forget(content)

    this.queue = new MessageQueue({
      actions: { interrupt: () => this.turns.cancelTurn(), send: text => void this.turns.startTurn(text) },
      // A picker, a broker prompt, a sign-in or a session switch: nothing
      // leaves the queue until the slot is the editor's again.
      isDispatchBlocked: () => this.phase2.blocked || this.sessions.inTransition,
      // A pending cancel blocks sends the way a running turn does: the gateway
      // refuses a send until the cancelled turn has unwound.
      isTurnActive: () => this.turn.active || this.turns.cancelling,
      theme: this.theme
    })

    this.facts = new SessionFacts(
      {
        footer: () => this.layout.footer,
        gateway: this.gateway,
        generation: () => this.sessions.generation,
        panel: () => this.layout.panel,
        print: (text, tone) => this.print(text, tone),
        sessionId: () => this.sessions.id,
        tui: this.tui,
        turn: this.turn,
        turns: () => this.turns
      },
      opts.cwd ?? process.cwd()
    )
    this.sessions = new SessionHub({
      chat: this.chat,
      editor: () => this.editor,
      env: this.env,
      facts: this.facts,
      gateway: this.gateway,
      isTurnActive: () => this.turn.active,
      onSessionReplaced: () => {
        this.usage = null
      },
      phase2: () => this.phase2,
      phase3: () => this.phase3,
      print: (text, tone) => this.print(text, tone),
      queue: this.queue,
      theme: this.theme,
      transcript: this.transcript,
      tui: this.tui,
      turn: this.turn,
      turns: () => this.turns
    })

    const keybindings = createKeybindings()

    this.commands = new CommandSurface({
      editor: () => this.editor,
      enqueue: text => this.enqueue(text),
      env: this.env,
      footer: () => this.layout.footer,
      gateway: this.gateway,
      hasRenderer: opts.renderer !== undefined,
      keybindings,
      phase2: () => this.phase2,
      phase3: () => this.phase3,
      queue: () => this.queue,
      quit: code => this.shell.quit(code),
      facts: this.facts,
      sessions: this.sessions,
      theme: this.theme,
      transcript: this.transcript,
      tui: this.tui,
      turn: this.turn,
      turns: () => this.turns
    })
    this.ctx = this.commands.context
    this.editor = new HarnessEditor(this.tui, this.theme.editorTheme(), {
      autocomplete: this.commands.autocomplete(),
      ...(opts.history ? { history: opts.history } : {})
    })
    this.editor.setPlaceholder(pickPlaceholder(), text => this.theme.fg('muted', text))
    this.editor.onSubmit = text => void this.submit(text)

    this.keys = createGlobalKeyHandler({
      actions: {
        cancelQueueEdit: () => this.phase3.queueEditing.cancel(),
        cancelTurn: () => this.turns.cancelTurn(),
        clearPendingInput: () => this.turns.clearPendingInput(),
        copySelection: () => this.phase3.copySelection(),
        dropQueued: () => this.phase3.queueEditing.drop(),
        editQueued: direction => this.phase3.queueEditing.cycle(direction),
        forceReset: () => this.turn.forceReset('cancel produced no response — input restored'),
        openExternalEditor: () => void this.shell.openExternalEditor(),
        paste: () => void this.phase3.paste(),
        quit: () => this.shell.quit(0),
        redraw: () => this.tui.requestRender(true),
        // The standing offer to exit, in the row the busy indicator uses. It
        // takes itself down after the arm window, because an idle screen
        // redraws for nothing else.
        showQuitHint: show =>
          this.layout.status.setNotice(show ? this.theme.fg('muted', QUIT_HINT) : null, QUIT_ARM_MS),
        submitQueueHead: () => this.turns.submitQueueHead(),
        toggleThinking: () =>
          this.commands.setDetails(
            'thinking',
            this.commands.getDetails('thinking') === 'expanded' ? 'collapsed' : 'expanded'
          ),
        toggleTools: () =>
          this.commands.setDetails(
            'tools',
            this.commands.getDetails('tools') === 'expanded' ? 'collapsed' : 'expanded'
          ),
        toggleYolo: () => void toggleYolo(this.ctx)
      },
      editor: this.editor,
      hasQueued: () => this.queue.length > 0,
      // While a picker or a prompt owns the slot, only it reads keys.
      isSlotActive: () => this.phase2.slotActive,
      isTurnActive: () => this.turn.active,
      keybindings
    })

    this.layout = new Layout({
      clock,
      facts: () => this.facts,
      showTokenUsage: () => this.commands.showTokenUsage,
      theme: this.theme,
      tui: this.tui,
      usage: () => this.usage
    })

    this.layout.editorContainer.addChild(this.editor)

    this.phase2 = new Phase2({
      clock,
      editor: this.editor,
      editorContainer: this.layout.editorContainer,
      gateway: this.gateway,
      keybindings,
      onGateChange: () => this.onGateChange(),
      onPromptAttention: waiting => this.turns.setPromptAttention(waiting),
      onModelApplied: update => this.facts.applyModel(update),
      onThinkingApplied: effort => this.facts.applyThinking(effort),
      session: () => ({
        effort: this.facts.info.effort,
        epoch: this.sessions.generation,
        id: this.sessions.id,
        model: this.facts.info.model,
        provider: this.facts.info.provider
      }),
      sessionActions: {
        blockedReason: () => this.sessions.switchBlockedReason(),
        currentId: () => this.sessions.id,
        removeCurrent: id => this.sessions.removeSession(id),
        resume: (id, stillCurrent) => this.sessions.resumeSession(id, stillCurrent)
      },
      theme: this.theme,
      transcript: {
        print: (text, tone) => this.print(text, tone),
        printBlock: (text, title) => this.printBlock(text, title)
      },
      tui: this.tui
    })

    this.phase3 = new Phase3({
      chat: this.chat,
      clock,
      drainQueue: () => this.turns.drain(),
      editor: this.editor,
      editorContainer: this.layout.editorContainer,
      env: this.env,
      footer: this.layout.footer,
      keybindings,
      panel: this.layout.panel,
      print: (text, tone) => {
        this.print(text, tone)
        this.tui.requestRender()
      },
      queue: this.queue,
      queueView: this.queue.view,
      rebindInput: () => this.shell.installInputListener(),
      requestRender: () => this.tui.requestRender(),
      sessionEpoch: () => this.sessions.generation,
      status: this.layout.status,
      theme: this.theme,
      tui: this.tui,
      ...(opts.renderer ? { renderer: opts.renderer } : {})
    })

    this.turns = new TurnRunner({
      blockedReason: () => this.sessions.switchBlockedReason(),
      editor: this.editor,
      gateway: this.gateway,
      keybindings,
      keys: () => this.keys,
      model: () => this.facts.info.model,
      now: this.now,
      phase2: () => this.phase2,
      phase3: () => this.phase3,
      print: (text, tone) => this.print(text, tone),
      queue: this.queue,
      sessionKey: () => this.sessions.id,
      status: this.layout.status,
      theme: this.theme,
      tui: this.tui,
      turn: this.turn
    })

    this.turn.onProteinProgress = payload => this.phase3.onProteinProgress(payload)

    this.shell = new AppShell({
      editor: () => this.editor,
      gateway: this.gateway,
      keys: () => this.keys,
      listenResize: opts.listenResize ?? listenSigwinch,
      ...(opts.history ? { history: opts.history } : {}),
      onExit: this.onExit,
      phase2: () => this.phase2,
      phase3: () => this.phase3,
      facts: () => this.facts,
      transcript: this.transcript,
      tui: this.tui,
      turn: this.turn,
      turns: () => this.turns
    })

    this.phase3.mount()
    this.tui.setFocus(this.editor)
    this.shell.installInputListener()

    this.gateway.onNotification = (method, params) => this.handleNotification(method, params)
  }

  /** Handshake, catalog, config, setup gate, session, subscription. The order
   *  is the contract: `system.hello` must be the first call on the socket. */
  async boot(): Promise<void> {
    const hello = await this.gateway.hello()

    const catalog = await this.gateway.commandsCatalog()

    this.commands.setCatalog(catalog)
    // The popup now knows the gateway's commands too.
    this.editor.setAutocompleteProvider(this.commands.autocomplete())

    const config = await this.gateway.configGet()
    const setup = await this.gateway.setupStatus()

    // `config.get` with no keys already serves every whitelisted key, so the
    // footer's setting comes out of the boot call rather than a second one.
    const settings = config.config as Record<string, JsonValue> | undefined

    this.commands.adoptShowTokenUsage(settings?.['tui.show_token_usage'] !== false)
    this.applyStoredTheme(settings?.['tui.theme'])

    this.facts.describe(catalog, hello.server_version)

    if (catalog.warning) {
      this.print(catalog.warning, 'warn')
    }

    if (setup.provider_configured === false) {
      for (const line of SETUP_REQUIRED_LINES) {
        this.transcript.addBlock(line, 'warn')
      }

      this.tui.requestRender()

      return
    }

    if (setup.compute_configured === false) {
      this.print('Protein Design compute is not configured. Run `ddeharness onboard` before starting a design.', 'warn')
    }

    await this.sessions.adoptSession(await this.sessions.openSession(config.config ?? {}))
    this.shell.onResize()

    const query = (this.env.OPENDDE_HARNESS_TUI_QUERY ?? '').trim()

    if (query) {
      await this.submit(query)
    }
  }

  /**
   * Paint with the palette `tui.theme` names, from the config boot already has.
   *
   * The screen is up by now, so a change repaints it the way the terminal's
   * background reply does. An environment variable that named a palette
   * outranks the stored name, and a name that is not a palette is said here
   * rather than swallowed: the transcript is where this app says things.
   */
  private applyStoredTheme(stored: JsonValue | undefined): void {
    const outcome = applyConfiguredTheme(this.theme, stored)

    if (outcome.repaint) {
      this.tui.invalidate()
      this.tui.requestRender(true)
    }

    if (outcome.invalid) {
      this.print(`tui.theme: ${String(stored)} is not a palette (${THEME_NAMES.join(', ')})`, 'warn')
    }
  }

  /** What the editor submits: a command runs here, anything else is a prompt
   *  that goes out now or waits its turn in the queue. */
  async submit(text: string): Promise<void> {
    const content = text.trim()

    if (looksLikeCommand(content)) {
      // A command submitted programmatically while a prompt is up is refused
      // rather than run behind it — and never turned into model text.
      if (this.phase2.blocked) {
        this.print('finish what is on screen before running a command', 'warn')
        this.tui.requestRender()

        return
      }

      if (this.phase2.commandBusy) {
        this.print('a command is still running', 'warn')
        this.tui.requestRender()

        return
      }

      this.editor.rememberPrompt(content)

      // One interactive command at a time, so a `confirm.request` that arrives
      // while one runs belongs to exactly one of them.
      const token = this.phase2.beginCommand()
      // Most commands answer instantly; the ones that go to the gateway need
      // not, and a screen that shows nothing reads as a hang.
      const finished = this.turns.showActivity(content.slice(1).trim() || 'working')

      try {
        await dispatch(this.commands.registry, this.ctx, content)
      } finally {
        finished()
        this.phase2.endCommand(token)
      }

      this.tui.requestRender()

      return
    }

    // A queued message being edited goes back where it was rather than being
    // appended as a second copy of itself.
    if (this.phase3.queueEditing.apply(content)) {
      this.tui.requestRender()

      return
    }

    if (content) {
      this.editor.rememberPrompt(content)
    }

    if (content && !this.sessions.id) {
      this.print('no session — nothing was sent', 'error')
      this.tui.requestRender()

      return
    }

    this.enqueue(content)
    this.tui.requestRender()
  }

  /** Send a prompt, or hold it behind the turn in flight. Every prompt that
   *  goes out this way becomes what `/retry` resends. */
  private enqueue(text: string): void {
    if (text) {
      this.transcript.remember(text)
    }

    this.queue.submit(text)
  }

  /** The four top-level notifications. All of them are the prompt
   *  coordinator's: it owns the deadlines and decides what reaches the slot. */
  handleNotification(method: GatewayNotification['method'], params: GatewayNotification['params']): void {
    this.phase2.handleNotification(method, params)
    this.tui.requestRender()
  }

  /** The socket went away. Prompts and pickers go with it; nothing is sent. */
  onDisconnect(): void {
    if (this.shell.quitting) {
      return
    }

    this.phase2.onDisconnect()
    this.tui.requestRender()
  }

  // ── help and footer settings ───────────────────────────────────────

  /** Something that gates sending changed hands. The queue may now have a way
   *  out, and the key router may need to step back in. */
  private onGateChange(): void {
    this.phase3.setSlotActive(this.phase2.slotActive)

    if (!this.phase2.blocked) {
      this.turns.drain()

      return
    }

    this.tui.requestRender()
  }

  // ── commands ───────────────────────────────────────────────────────

  private print(text: string, tone?: SystemTone): void {
    this.transcript.print(text, tone)
  }

  private printBlock(text: string, title?: string): void {
    this.transcript.printBlock(text, title)
  }
}
