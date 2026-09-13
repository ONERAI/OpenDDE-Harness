// Protein design: the product's own work, from the screen where its tasks run.
//
// `/design` is the view — what the plugin will do with a configuration, its
// defaults and its example paths — and the two actions validate a file or
// launch it. They run on the gateway, which is where the plugin lives; the
// tasks they start appear in `/tasks` without anything else being typed.
//
// `/design:context` is not a separate command: the bare feature is the view,
// and the view is exactly what the CLI's `protein-design context` prints.

import type { Command } from './types.js'

import { runOnGateway } from './passthrough.js'

/** A YAML path is the argument both actions take, and the thing a user has
 *  open in another window while they type. */
const CONFIG_HINT = '<config.yaml>'

function needsConfig(name: string): string {
  return `usage: /${name} ${CONFIG_HINT}`
}

export const designCommands: Command[] = [
  {
    description: 'what a protein-design run would use: defaults, paths, compute',
    name: 'design',
    run: ctx => runOnGateway(ctx, 'protein-design context', 'Protein design')
  },

  {
    argumentHint: CONFIG_HINT,
    description: 'check a design configuration without starting anything',
    name: 'design:validate',
    run: (ctx, arg) => {
      const config = arg.trim()

      if (!config) {
        return ctx.transcript.print(needsConfig('design:validate'), 'warn')
      }

      return runOnGateway(ctx, `protein-design validate --config ${config}`, 'Protein design')
    }
  },

  {
    argumentHint: CONFIG_HINT,
    description: 'launch a design run as a detached task',
    name: 'design:start',
    run: (ctx, arg) => {
      const config = arg.trim()

      if (!config) {
        return ctx.transcript.print(needsConfig('design:start'), 'warn')
      }

      // The task it starts is the one `/tasks` lists a moment later.
      return runOnGateway(ctx, `protein-design start --config ${config}`, 'Protein design')
    }
  }
]
