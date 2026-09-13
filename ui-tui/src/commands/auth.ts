// `/login` and `/logout`: pi's own two authentication commands.
//
// Both open the picker by their own door and then run pi's steps — the
// authentication method, the provider, the sign-in; or the sign-in to remove.
// Neither does any work of its own: the flow belongs to the selector, which is
// the one place that holds the login stage.
//
// The names, the descriptions and the argument hint are pi's own
// (`core/slash-commands.ts`), so somebody who knows pi types what they know.

import type { Command } from './types.js'

export const authCommands: Command[] = [
  {
    argumentHint: '<provider>',
    description: 'Configure provider authentication',
    name: 'login',
    // With a provider named, pi goes straight to that one; bare, it asks which
    // way in first. Both decisions are the selector's.
    run: (ctx, arg) => ctx.openLoginPicker(arg.trim() || undefined)
  },

  {
    description: 'Remove provider authentication',
    name: 'logout',
    run: ctx => ctx.openLogoutPicker()
  }
]
