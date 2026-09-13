// The MCP servers this session can reach, and reloading them.
//
// `/mcp` is the view and `/mcp:reload` is the action, which is pi's shape for
// a feature with more than one verb. The data is the session's own: the
// gateway already reports every configured server, its transport, whether it
// has connected and how many tools it brought, so the view asks for the
// session's facts again rather than keeping a copy that goes stale.

import type { Command } from './types.js'

import { errorMessage } from './passthrough.js'

/** How the connector talks to a server, in the words its config uses. */
const TRANSPORT: Record<string, string> = {
  sse: 'SSE',
  stdio: 'stdio',
  streamableHttp: 'HTTP'
}

/** Servers connect during the session's first turn, so "not connected" on a
 *  fresh session means "not yet", not "broken". */
function line(server: { connected: boolean; name: string; tool_count: number; transport: string }): string {
  const transport = TRANSPORT[server.transport] ?? server.transport
  const state = server.connected ? `${server.tool_count} tool${server.tool_count === 1 ? '' : 's'}` : 'not connected'

  return `${server.name.padEnd(18)} ${transport.padEnd(7)} ${state}`
}

export const mcpCommands: Command[] = [
  {
    description: 'the MCP servers this session can reach',
    name: 'mcp',
    run: async ctx => {
      try {
        const { info } = await ctx.gateway.sessionInfo(ctx.session.id())
        const servers = info.mcp_servers ?? []

        if (!servers.length) {
          return ctx.transcript.print('no MCP servers configured')
        }

        ctx.transcript.printBlock(servers.map(line).join('\n'), 'MCP servers')
      } catch (err) {
        ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
      }
    }
  },

  {
    argumentHint: '[always]',
    description: 'reload the MCP servers in this session',
    getArgumentCompletions: prefix =>
      'always'.startsWith(prefix.trim()) ? [{ label: 'always', value: 'always' }] : [],
    name: 'mcp:reload',
    run: async (ctx, arg) => {
      const always = arg.trim().toLowerCase() === 'always'

      try {
        const result = await ctx.gateway.reloadMcp({
          confirm: true,
          session_id: ctx.session.id(),
          ...(always ? { always: true } : {})
        })

        if (!result.ok) {
          return ctx.transcript.print('MCP reload did not run', 'warn')
        }

        const changed = result.tools_changed ? ' · tool list changed' : ''

        ctx.transcript.print(`reloaded ${result.reloaded ?? 0} MCP server(s)${changed}`)
      } catch (err) {
        ctx.transcript.print(`error: ${errorMessage(err)}`, 'error')
      }
    }
  }
]
