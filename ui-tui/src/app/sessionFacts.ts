// What is known about the live session: its model and provider, the window
// they run in, how they are billed, where it is running and what it can reach.
//
// Everything here arrives with a session and changes when the session does, or
// when a command changes one of them on purpose. The panel and the footer read
// it; nothing else owns a copy.

import type { TUI } from '@earendil-works/pi-tui'

import type { SessionInfoView } from '../commands/index.js'
import type { Footer } from '../components/footer.js'
import type { SessionPanel, SessionPanelData } from '../components/sessionPanel.js'
import type { SystemTone } from '../components/systemLine.js'
import type { Gateway } from '../gateway.js'
import type { CommandsCatalogResult, McpServerInfo, ProjectInstructionFile, SessionCreateResult } from '../rpc/index.js'
import type { TurnController } from '../turn.js'
import type { TurnRunner } from './turnRunner.js'

import { GitBranch } from '../lib/gitBranch.js'

/** The init bundle, plus the legacy notices current handlers omit. */
export type SessionInit = SessionCreateResult['info'] & {
  config_notices?: string[]
  config_warning?: string
  credential_warning?: string
}

/**
 * What the session had already spent when it was opened.
 *
 * A resumed session's earlier turns ran in another process, and until its first
 * turn here reports, nothing the turn model holds knows about them: the footer
 * opened a conversation with forty calls behind it at `$0.000`. The gateway
 * totals the session's stored records for exactly this, and seeds its own
 * tracker with the same sum — so this figure and the one the next
 * `message.complete` brings are the same accounting, not two.
 *
 * Null costs are the gateway's unknown: a plan states no per-token price, and a
 * model in no price table has no list price. Neither is zero.
 */
export interface OpeningTotals {
  cost: null | number
  input: number
  listCost: null | number
  output: number
}

export interface SessionFactsOptions {
  /** Built after this; reached at call time. */
  footer: () => Footer
  gateway: Gateway
  /** The session these facts describe, for the call that asks about it again. */
  generation: () => number
  panel: () => SessionPanel
  print: (text: string, tone?: SystemTone) => void
  sessionId: () => null | string
  tui: TUI
  turn: TurnController
  turns: () => TurnRunner
}

export class SessionFacts {
  private branch: GitBranch | null = null
  private facts: SessionInfoView = {}
  /** See {@link OpeningTotals}. Zeroed for a session with no history. */
  private totals: OpeningTotals = { cost: null, input: 0, listCost: null, output: 0 }
  private workdir: string
  /** `session.info`'s grouped tools and skills, and its update notice. */
  private skills: Record<string, string[]> | undefined
  private tools: Record<string, string[]> | undefined
  private notice: { command?: string; version: string } | undefined
  /** Which refresh is the current one. Two model switches in a row leave two
   *  calls in flight, and the wire does not promise the order they come back
   *  in: the older answer described a model this session has already left. */
  private refreshSeq = 0
  /** The AGENTS.md / ODH.md files this session is following, as of the last
   *  answer about them. Kept so a file whose content moves can be reported
   *  once, when it moves, rather than every turn after it. */
  private instructions: ProjectInstructionFile[] = []
  /** The MCP servers this session was told about, and whether they had
   *  connected when it was told. */
  private servers: McpServerInfo[] = []
  /** The session generation whose first turn has already been settled, so the
   *  question is asked once per session rather than after every turn. */
  private settledFor: number | undefined

  constructor(
    private readonly opts: SessionFactsOptions,
    cwd: string
  ) {
    this.workdir = cwd
  }

  get cwd(): string {
    return this.workdir
  }

  get info(): SessionInfoView {
    return this.facts
  }

  /** What this session had spent before the turn model counted anything. */
  get opening(): OpeningTotals {
    return this.totals
  }

  get update(): { command?: string; version: string } | undefined {
    return this.notice
  }

  /** The branch name for the footer, or null outside a repository. */
  branchName(): null | string {
    return this.branch?.get() ?? null
  }

  /** A title the gateway gave a session after it was opened. */
  setTitle(title: string): void {
    this.facts.title = title
  }

  // Preserve the existing display of legacy notices, which current handlers omit.
  applySessionInfo(sessionId: string, info?: SessionInit): void {
    // A fork carries no facts of its own: `session.branch` answers with the
    // child's id, title and message count, and the child runs on everything it
    // inherited. Replacing the facts with nothing blanked the model and
    // provider in the panel and the footer for a session that had not changed
    // either. Only a real init bundle replaces them.
    if (info) {
      this.facts = {
        autoCompact: info.auto_compact,
        // Null is the gateway saying nothing measures this window; zero is a
        // session with nothing in it, which is not the same answer.
        contextTokens: info.usage?.context_used === null ? null : info.usage?.context_used || undefined,
        contextWindow: info.context_window,
        effort: info.reasoning_effort ?? undefined,
        model: info.model,
        provider: info.provider,
        subscription: info.subscription
      }

      this.totals = openingTotals(info)

      if (info.cwd) {
        this.workdir = info.cwd
      }

      this.branch?.dispose()
      this.branch = new GitBranch(this.workdir, () => this.opts.tui.requestRender())

      // The real inventory and the update notice, which only a full init
      // bundle carries. A fork inherits both from the session it came from.
      this.tools = info.tools
      this.skills = info.skills
      this.servers = info.mcp_servers ?? []
      this.instructions = info.project_instructions ?? []
      this.notice = info.update_available
        ? { version: info.update_available, ...(info.update_command ? { command: info.update_command } : {}) }
        : undefined
      this.opts.footer().invalidate()

      // Said once, in the transcript, as well as kept in the status bar: a
      // status-bar glyph is easy to sit under for weeks.
      if (this.notice) {
        this.opts.print(
          `OpenDDE Harness ${this.notice.version} is on PyPI; this is ${info.version}.` +
            (this.notice.command ? ` Update with: ${this.notice.command}` : '')
        )
      }
    }

    this.opts.panel().update({ ...this.panelPatch(info), sessionId })

    for (const warning of [info?.credential_warning, info?.config_warning].filter(Boolean) as string[]) {
      this.opts.print(`warning: ${warning}`, 'warn')
    }

    for (const notice of info?.config_notices ?? []) {
      this.opts.print(notice)
    }

    this.opts.turns().updateTitle('idle')
  }

  /**
   * Everything the panel shows about the session, from one bundle.
   *
   * Absent fields are sent as they arrived rather than dropped: an empty
   * server list is the gateway saying there is none, and a panel that kept the
   * last session's value would be showing something this session does not
   * have. A fork carries no bundle at all, and then only what this object
   * already holds is re-sent.
   *
   * No working directory and no system-prompt size: the footer already says
   * where the session runs, and the panel is the brand and what the session is,
   * not a second copy of the footer.
   */
  private panelPatch(info?: SessionInit): SessionPanelData {
    return {
      model: this.facts.model,
      provider: this.facts.provider,
      reasoningEffort: this.facts.effort,
      ...(this.skills ? { skills: this.skills } : {}),
      ...(this.tools ? { tools: this.tools } : {}),
      ...(info
        ? {
            mcpServers: info.mcp_servers,
            releaseDate: info.release_date,
            version: info.version
          }
        : {})
    }
  }

  /** What the gateway itself is, for the panel a boot fills in. */
  describe(catalog: CommandsCatalogResult, serverVersion: string): void {
    this.opts.panel().update({
      commandCount: catalog.pairs?.length,
      skillCount: catalog.skill_count,
      version: serverVersion
    })
  }

  /** A command changed one of the facts. */
  patchInfo(patch: Partial<SessionInfoView>): void {
    Object.assign(this.facts, patch)
    this.opts.panel().update({ model: this.facts.model, reasoningEffort: this.facts.effort })
    this.opts.tui.requestRender()

    // A different model is a different window, a different way of being
    // billed and possibly a different answer about compaction. The
    // command that changed it knows none of those.
    if ('model' in patch || 'provider' in patch) {
      void this.refreshSessionFacts()
    }

    // The stored conversation changed size (`/undo`). It supersedes the
    // last call's figure, which described the longer one.
    if ('contextTokens' in patch) {
      this.opts.turn.setContextBaseline(patch.contextTokens === undefined ? 0 : patch.contextTokens)
    }
  }

  /** The model picker applied a model, which may also be a new provider. */
  applyModel(update: Partial<SessionInfoView>): void {
    this.facts = { ...this.facts, ...update }
    this.opts.panel().update({
      model: this.facts.model,
      provider: this.facts.provider,
      reasoningEffort: this.facts.effort
    })
    this.opts.tui.requestRender()
    void this.refreshSessionFacts()
  }

  /** The thinking picker applied an effort. Nothing else about the session
   *  changed, so nothing else is asked for. */
  applyThinking(effort: string | undefined): void {
    this.facts.effort = effort
    this.opts.panel().update({ reasoningEffort: effort })
    this.opts.tui.requestRender()
  }

  /**
   * Ask the gateway for this session's facts again.
   *
   * A live model switch reports the model, the provider and the thinking
   * level, and that is all it knows: whether the new provider is billed by
   * plan, what its context window is, and whether anything will compact the
   * session are answers only the gateway has. Without this the footer kept
   * the previous provider's, so switching to the Codex login left the cost
   * unmarked until the next session was opened.
   *
   * The same call answers a second question: MCP servers connect on the
   * session's first turn, so the bundle a boot saw listed them all as not
   * connected. Asking again once a turn has run is how the panel learns what
   * they brought.
   *
   * Best effort: a refusal leaves the facts as they were, which is what they
   * already showed.
   */
  async refresh(): Promise<void> {
    return this.refreshSessionFacts()
  }

  /**
   * A turn has ended, so anything that was still connecting has settled.
   *
   * MCP servers connect during the session's first turn, which is after the
   * bundle that opened the session was built: it listed them configured and
   * not connected, and nothing tells the panel when that changes. Asked once
   * per session, and only when there is a server whose state could still move
   * — a session with none, or with all of them already connected, has nothing
   * to learn and is not worth the call.
   */
  settleAfterTurn(): void {
    void this.checkInstructions()

    const epoch = this.opts.generation()

    if (this.settledFor === epoch || !this.servers.some(server => !server.connected)) {
      return
    }

    this.settledFor = epoch
    void this.refreshSessionFacts()
  }

  /**
   * Say so when an instruction file's content has moved under this session.
   *
   * These files are read into the prompt as the user's own standing
   * instructions, unfenced. A write to one goes through an approval prompt, so
   * the ordinary case is already visible; this is what covers the rest --
   * a shell command that wrote one, an editor, a branch switch. Said once,
   * when it changes, because the gateway keeps reporting it changed for the
   * rest of the session.
   *
   * Cheap by construction: it asks only when this session has such files, and
   * `session.instructions` is a directory walk, not the session-info bundle.
   */
  private async checkInstructions(): Promise<void> {
    const id = this.opts.sessionId()

    if (!id || !this.instructions.length) {
      return
    }

    try {
      const result = await this.opts.gateway.sessionInstructions({ session_id: id })
      const reported = new Set(this.instructions.filter(file => file.changed).map(file => file.path))

      this.instructions = result.files

      for (const file of result.files) {
        if (file.changed && !reported.has(file.path)) {
          this.opts.print(
            `${file.display} has changed since this session started — /memory shows what is in force`,
            'warn'
          )
        }
      }
    } catch {
      // A list this could not fetch is a line it cannot print. The next turn
      // asks again.
    }
  }

  private async refreshSessionFacts(): Promise<void> {
    const epoch = this.opts.generation()
    const seq = (this.refreshSeq += 1)

    try {
      const { info } = await this.opts.gateway.sessionInfo(this.opts.sessionId())

      // A session adopted while the call was in flight has its own facts.
      if (epoch !== this.opts.generation()) {
        return
      }

      // And a switch inside this session, which does not change the epoch,
      // leaves an older call in flight describing the model it has left.
      // Answers arrive in whatever order the gateway finishes them in.
      if (seq !== this.refreshSeq) {
        return
      }

      this.facts = {
        ...this.facts,
        autoCompact: info.auto_compact,
        contextWindow: info.context_window,
        effort: info.reasoning_effort ?? undefined,
        model: info.model,
        provider: info.provider,
        subscription: info.subscription
      }
      // The same tracker the completions report from, asked again: a switch mid
      // session does not reset what the session has spent.
      this.totals = openingTotals(info)
      // The inventory too: MCP servers connect on the session's first turn, so
      // the bundle a boot saw said "not connected" for all of them and only a
      // later snapshot has the tools they brought.
      this.tools = info.tools
      this.skills = info.skills
      this.servers = info.mcp_servers ?? []
      this.instructions = info.project_instructions ?? []
      this.notice = info.update_available
        ? { version: info.update_available, ...(info.update_command ? { command: info.update_command } : {}) }
        : undefined
      this.opts.panel().update(this.panelPatch(info))
      // The measured window belongs to whichever model answered last, which
      // after a switch is not this one. Until the next completion replaced it,
      // the footer divided this session's occupancy by the old model's
      // capacity; the gateway has just said how big this model's is.
      this.opts.turn.setContextWindow(info.context_window ?? 0)
      this.opts.footer().invalidate()
      this.opts.tui.requestRender()
    } catch {
      // The facts on screen are the ones it had; none of them is wrong enough
      // to interrupt the user over.
    }
  }

  dispose(): void {
    this.branch?.dispose()
  }
}

/** The gateway's `info.usage` as the footer's opening totals. */
function openingTotals(info: SessionInit): OpeningTotals {
  return {
    cost: info.usage?.cost_usd ?? null,
    input: info.usage?.input ?? 0,
    listCost: info.usage?.list_cost_usd ?? null,
    output: info.usage?.output ?? 0
  }
}
