// The command surface: the registry, the context every command is handed, the
// completion popup behind the editor, and the two settings commands change
// that belong to no one else — the footer's counters and how much of the
// agent's working-out the transcript shows.

import type { KeybindingsManager, TUI } from '@earendil-works/pi-tui'

import type { CommandContext, DetailsMode, DetailsSection } from '../commands/index.js'
import type { HarnessEditor } from '../components/editor.js'
import type { Footer } from '../components/footer.js'
import type { BusyInputMode } from '../components/queue.js'
import type { MessageQueue } from '../components/queue.js'
import type { Gateway } from '../gateway.js'
import type { Phase2 } from '../phase2.js'
import type { Phase3 } from '../phase3.js'
import type { CommandsCatalogResult } from '../rpc/index.js'
import type { Theme } from '../theme.js'
import type { TurnController } from '../turn.js'
import type { SessionFacts } from './sessionFacts.js'
import type { SessionHub } from './sessionHub.js'
import type { TranscriptView } from './transcript.js'
import type { TurnRunner } from './turnRunner.js'

import { HOTKEY_ORDER } from '../commands/core.js'
import { CommandRegistry } from '../commands/index.js'
import { HarnessAutocompleteProvider, resolveFdPath } from '../components/autocomplete.js'
import { HelpPanel } from '../components/helpPanel.js'
import { taskAccess } from '../phase3.js'

export interface CommandSurfaceOptions {
  editor: () => HarnessEditor
  env: NodeJS.ProcessEnv
  footer: () => Footer
  gateway: Gateway
  hasRenderer: boolean
  keybindings: KeybindingsManager
  /** A prompt a command sends goes through the queue, like any other. */
  enqueue: (text: string) => void
  /** Built after this; reached at call time. */
  phase2: () => Phase2
  phase3: () => Phase3
  queue: () => MessageQueue
  quit: (code: number) => void
  facts: SessionFacts
  sessions: SessionHub
  theme: Theme
  transcript: TranscriptView
  tui: TUI
  turn: TurnController
  turns: () => TurnRunner
}

export class CommandSurface {
  readonly registry = new CommandRegistry()
  readonly context: CommandContext

  private catalogue: CommandsCatalogResult | null = null
  private tokenUsage = true
  private thinkingVisible = false
  private toolsExpanded = false

  constructor(private readonly opts: CommandSurfaceOptions) {
    this.context = this.commandContext(opts.keybindings, opts.hasRenderer)
  }

  get catalog(): CommandsCatalogResult | null {
    return this.catalogue
  }

  get showTokenUsage(): boolean {
    return this.tokenUsage
  }

  /** What the gateway says it can run, learned at boot. */
  setCatalog(catalog: CommandsCatalogResult): void {
    this.catalogue = catalog
  }

  /** The stored setting, read from config at boot rather than stored again. */
  adoptShowTokenUsage(value: boolean): void {
    this.tokenUsage = value
  }

  /** Slash commands from the registry and the gateway's catalog, paths from
   *  `complete.path` with pi's local walk behind it. */
  autocomplete(): HarnessAutocompleteProvider {
    return new HarnessAutocompleteProvider({
      basePath: this.opts.facts.cwd,
      commands: this.registry.slashCommands(this.context),
      completePath: word => this.completePath(word),
      fdPath: resolveFdPath(this.opts.env)
    })
  }

  /** `/help`. A block in the transcript, not a selector: focus stays where the
   *  user left it and the terminal keeps the panel in its scrollback. */
  showHelp(): void {
    const catalog = this.catalogue
    const local = new Set(this.registry.commands.map(command => command.name))

    this.opts.transcript.add(
      new HelpPanel(this.opts.theme, this.context.keybindings, {
        // A local command of the same name wins: it is the one that runs.
        catalog: (catalog?.pairs ?? [])
          .map(pair =>
            String(pair[0] ?? '')
              .replace(/^\//, '')
              .trim()
          )
          .filter(name => name && !local.has(name)),
        commands: this.registry.commands
          .filter(command => !command.hidden)
          .map(command => ({
            description: command.description,
            name: command.name,
            ...(command.argumentHint ? { argumentHint: command.argumentHint } : {})
          })),
        hotkeys: HOTKEY_ORDER,
        ...(catalog?.skill_count ? { skillCount: catalog.skill_count } : {})
      })
    )
    this.opts.tui.requestRender()
  }

  /** `tui.show_token_usage`. The footer changes only once the gateway says it
   *  stored the value. */
  private async applyShowTokenUsage(value: boolean): Promise<boolean> {
    const result = await this.opts.gateway.configSet({ key: 'tui.show_token_usage', value })

    if (!result.applied) {
      return false
    }

    this.tokenUsage = value
    this.opts.footer().invalidate()
    this.opts.tui.requestRender()

    return true
  }

  /** Built before phase 3 is, because the editor's completion popup needs it.
   *  Everything phase 3 owns is therefore reached through `phase3()` at call
   *  time rather than captured here. */
  private commandContext(keybindings: KeybindingsManager, hasRenderer: boolean): CommandContext {
    return {
      busyMode: () => this.opts.queue().busyMode,
      catalog: () => this.catalogue,
      gateway: this.opts.gateway,
      keybindings,
      openLoginPicker: provider => this.opts.phase2().openLoginPicker(provider),
      openLogoutPicker: () => this.opts.phase2().openLogoutPicker(),
      openModelPicker: (scope, query) => this.opts.phase2().openModelPicker(scope, query),
      openScopedModelsPicker: () => this.opts.phase2().openScopedModelsPicker(),
      openThinkingPicker: () => this.opts.phase2().openThinkingPicker(),
      quietTools: {
        get: () => this.opts.turn.quietTools,
        set: quiet => this.opts.turn.setToolsQuiet(quiet)
      },
      quit: () => this.opts.quit(0),
      ...(hasRenderer
        ? {
            renderer: {
              mode: () => this.opts.phase3().renderer!.mode(),
              switchTo: mode => this.opts.phase3().renderer!.switchTo(mode)
            }
          }
        : {}),
      session: {
        adopt: (id, title) => this.opts.sessions.adoptFork(id, title),
        // "Busy" is anything that makes a session switch unsafe, not just a
        // running turn: a cancel that has not come back, a switch already in
        // flight, or messages still waiting to be sent.
        blockedReason: () => this.opts.sessions.switchBlockedReason(),
        busy: () => this.opts.sessions.switchBlockedReason() !== null || this.opts.phase2().blocked,
        create: title => this.opts.sessions.createSession(title),
        id: () => this.opts.sessions.id,
        info: () => this.opts.facts.info,
        patchInfo: patch => this.opts.facts.patchInfo(patch),
        pick: () => this.opts.phase2().openSessionPicker(),
        remove: id => this.opts.sessions.removeSession(id),
        resume: (id, stillCurrent) => this.opts.sessions.resumeSession(id, stillCurrent),
        send: text => this.opts.enqueue(text)
      },
      setBusyMode: (mode: BusyInputMode) => this.opts.queue().setMode(mode),
      setShowTokenUsage: value => this.applyShowTokenUsage(value),
      showHelp: () => this.showHelp(),
      showTokenUsage: () => this.tokenUsage,
      tasks: taskAccess(
        () => this.opts.phase3(),
        () => this.opts.sessions.generation
      ),
      transcript: {
        dropLastExchange: () => this.opts.transcript.dropLastExchange(),
        history: () => this.opts.transcript.history(),
        print: (text, tone) => this.opts.transcript.print(text, tone),
        printBlock: (text, title) => this.opts.transcript.printBlock(text, title),
        retryCandidate: () => this.opts.transcript.retryCandidate()
      }
    }
  }

  getDetails(section: DetailsSection): DetailsMode {
    const expanded = section === 'thinking' ? this.thinkingVisible : this.toolsExpanded

    return expanded ? 'expanded' : 'collapsed'
  }

  /**
   * `hidden` reads as collapsed: pi keeps a one-line label rather than
   * nothing, so there is always something to click.
   *
   * Applied to the components every time, not only when the stored default
   * moves. A click on one tool panel or one thinking label changes that
   * component alone, so `/details tools collapsed` has to reach it even though
   * the default already says collapsed.
   */
  setDetails(section: DetailsSection, mode: DetailsMode): void {
    const expanded = mode === 'expanded'

    if (section === 'thinking') {
      this.thinkingVisible = expanded
      this.opts.turn.setThinkingCollapsed(!expanded)
    } else {
      this.toolsExpanded = expanded
      this.opts.turn.setToolsExpanded(expanded)
    }
  }

  private completePath(word: string) {
    // The handler returns never[]; retain the existing adapter for remote suggestions.
    return this.opts.gateway.completePath(word).then(result =>
      (result.items ?? []).map((item: { display?: string; text: string }) => ({
        label: item.display ?? item.text,
        value: item.text
      }))
    )
  }
}
