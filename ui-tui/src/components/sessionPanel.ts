import type { Component } from '@earendil-works/pi-tui'

import { truncateToWidth } from '@earendil-works/pi-tui'

import type { McpServerInfo } from '../rpc/index.js'
import type { Theme } from '../theme.js'

import { BRAND } from '../content/brand.js'
import { Branding } from './branding.js'

export interface SessionPanelData {
  commandCount?: number
  /** When this version shipped, shown beside it on the lockup. */
  releaseDate?: null | string
  /** `session.info`'s MCP servers, for the count. They connect on the first
   *  turn, so a fresh session has them configured and not yet connected. */
  mcpServers?: McpServerInfo[]
  model?: string
  provider?: string
  /** JSON null when the model has no effort setting (see the generated `SessionInfo`). */
  reasoningEffort?: null | string
  sessionId?: string
  /** `session.info`'s grouped skills, for the count: group name to skill names. */
  skills?: Record<string, string[]>
  /** Fallback when the session has served no grouped skills yet. */
  skillCount?: number
  /** `session.info`'s grouped tools, for the count: group name to tool names. */
  tools?: Record<string, string[]>
  version?: string
}

function total(groups: Record<string, string[]> | undefined): number {
  return groups ? Object.values(groups).reduce((sum, names) => sum + names.length, 0) : 0
}

/** `1 command`, `2 commands`. */
function count(n: number, noun: string): string {
  return `${n} ${noun}${n === 1 ? '' : 's'}`
}

export interface SessionPanelOptions {
  /** Terminal height, which decides how much of the brand lockup fits. */
  rows?: () => number
}

/** The header above the transcript: the brand lockup, then who we are, what we
 *  run on, and where we are. */
export class SessionPanel implements Component {
  private readonly branding: Branding

  private cachedKey?: string
  private cachedLines?: string[]

  constructor(
    private readonly theme: Theme,
    private data: SessionPanelData = {},
    options: SessionPanelOptions = {}
  ) {
    this.branding = new Branding(theme, {
      ...(options.rows ? { rows: options.rows } : {}),
      ...(data.releaseDate ? { releaseDate: data.releaseDate } : {}),
      ...(data.version ? { version: data.version } : {})
    })
  }

  update(patch: SessionPanelData): void {
    this.data = { ...this.data, ...patch }

    if ('version' in patch || 'releaseDate' in patch) {
      this.branding.setVersion(this.data.version, this.data.releaseDate)
    }

    this.invalidate()
  }

  invalidate(): void {
    this.branding.invalidate()
    this.cachedKey = undefined
    this.cachedLines = undefined
  }

  render(width: number): string[] {
    // Keyed on the palette as well as the width: see `Branding.render`.
    const key = `${width}:${this.theme.scheme}`

    if (this.cachedLines && this.cachedKey === key) {
      return this.cachedLines
    }

    const t = this.theme
    const d = this.data
    const dot = t.fg('muted', ' · ')

    // The lockup owns the product name and the version; everything below it is
    // what this particular session is.
    const lines = [...this.branding.render(width)]

    const model = [d.model, d.provider, d.reasoningEffort].filter(Boolean) as string[]
    const identity = model.map(part => t.fg('text', part))

    if (d.sessionId) {
      identity.push(t.fg('dim', 'session ') + t.fg('text', d.sessionId))
    }

    if (identity.length > 0) {
      lines.push(identity.join(dot))
    }

    const session: string[] = []

    if (d.commandCount !== undefined) {
      session.push(t.fg('muted', count(d.commandCount, 'command')))
    }

    // The real count when the session served its tools; the catalog's is only a
    // stand-in until it has.
    const toolCount = total(d.tools)

    if (toolCount > 0) {
      session.push(t.fg('muted', count(toolCount, 'tool')))
    }

    const skillCount = d.skills ? total(d.skills) : d.skillCount

    if (skillCount !== undefined) {
      session.push(t.fg('muted', count(skillCount, 'skill')))
    }

    // Only worth a slot when some are configured: most installs have none.
    if (d.mcpServers && d.mcpServers.length > 0) {
      session.push(t.fg('muted', count(d.mcpServers.length, 'MCP server')))
    }

    if (session.length > 0) {
      lines.push(session.join(dot))
    }

    lines.push(t.fg('muted', BRAND.welcome), '')

    this.cachedKey = key
    this.cachedLines = lines.map(line => truncateToWidth(line, width))

    return this.cachedLines
  }
}
