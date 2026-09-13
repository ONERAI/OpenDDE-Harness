/**
 * How many colors we may emit, and where that number comes from.
 *
 * Detection of the terminal's real capability is chalk's job (mainstream
 * `supports-color`). This module layers our overrides and the two corrections
 * chalk gets wrong on top of it, and hands the answer to `theme.ts`, which
 * builds a `Chalk` instance pinned to that level. Nothing here mutates the
 * global chalk singleton, so import order is not load-bearing.
 *
 * Channels:
 *   - `NO_COLOR` (any value, per no-color.org) — colors off, beats everything.
 *   - `OPENDDE_HARNESS_TUI_COLOR` = auto | truecolor | 256 | 16 | none — the
 *     `--color` flag forwards here. Pins the tier exactly.
 *   - `OPENDDE_HARNESS_TUI_TRUECOLOR` = 1/true/yes/on — legacy alias for
 *     `OPENDDE_HARNESS_TUI_COLOR=truecolor`.
 */

export type ColorTier = 0 | 1 | 2 | 3

const TRUE_RE = /^(?:1|true|yes|on)$/i

/**
 * The tier the user asked for, or `null` for "auto" (defer to detection).
 * `NO_COLOR` is not consulted here — see `resolveColorTier`.
 */
export function parseColorOverride(env: NodeJS.ProcessEnv): ColorTier | null {
  const raw = (env.OPENDDE_HARNESS_TUI_COLOR ?? '').trim().toLowerCase()

  switch (raw) {
    case '0':
    case 'none':
    case 'off':
      return 0
    case '1':
    case '16':
    case 'ansi':
      return 1
    case '2':
    case '256':
    case 'ansi256':
      return 2
    case '24bit':
    case '3':
    case 'rgb':
    case 'truecolor':
      return 3
    default:
      break
  }

  if (TRUE_RE.test((env.OPENDDE_HARNESS_TUI_TRUECOLOR ?? '').trim())) {
    return 3
  }

  return null
}

/**
 * The tier to render at, given the environment and chalk's auto-detected level.
 *
 * Corrections applied to the detected level (skipped entirely when the user
 * pinned a tier):
 *   - xterm.js (VS Code / Cursor / code-server) has been truecolor since 2017
 *     but often doesn't set COLORTERM, and supports-color doesn't know
 *     `TERM_PROGRAM=vscode` — it lands on 2, where `chalk.hex()` collapses onto
 *     the 6x6x6 cube. Boost to 3.
 *   - tmux only re-emits truecolor SGR when the outer terminal advertises RGB,
 *     which the default config doesn't. Clamp to 256, which passes through.
 *   - Terminal.app before macOS Tahoe 26 approximates 24-bit SGR to its own
 *     256 palette, and many shells export COLORTERM=truecolor globally. Clamp.
 *
 * Order matters: the boost runs first so that tmux inside VS Code still clamps.
 */
export function resolveColorTier(env: NodeJS.ProcessEnv, detectedLevel: number): ColorTier {
  if ('NO_COLOR' in env) {
    return 0
  }

  const pinned = parseColorOverride(env)

  if (pinned !== null) {
    return pinned
  }

  let level = Math.max(0, Math.min(3, Math.trunc(detectedLevel))) as ColorTier

  if (env.TERM_PROGRAM === 'vscode' && level === 2) {
    level = 3
  }

  if (env.TMUX && level > 2) {
    level = 2
  }

  if (env.TERM_PROGRAM === 'Apple_Terminal' && level > 2) {
    level = 2
  }

  return level
}
