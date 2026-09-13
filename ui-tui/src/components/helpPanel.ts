// `/help`, as a block appended to the transcript.
//
// It is not a selector: it never takes the editor slot and never takes focus,
// so the prompt stays where it was and the terminal's own scrollback keeps it.
// At narrow widths the description drops to its own indented line rather than
// being squeezed into a column that fits nothing.

import type { Component, KeybindingsManager } from '@earendil-works/pi-tui'
import type { Keybinding } from '@earendil-works/pi-tui'

import { sliceByColumn, truncateToWidth, visibleWidth } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

/** The narrowest terminal that still gets a two-column layout. */
const TWO_COLUMN_MIN = 52

export interface HelpCommand {
  argumentHint?: string
  description: string
  name: string
}

export interface HelpPanelData {
  /** Commands the gateway serves that this UI does not implement itself. */
  catalog: string[]
  commands: HelpCommand[]
  hotkeys: Keybinding[]
  skillCount?: number
}

export class HelpPanel implements Component {
  private cachedKey = ''
  private lines: string[] = []

  constructor(
    private readonly theme: Theme,
    private readonly keybindings: KeybindingsManager,
    private readonly data: HelpPanelData
  ) {}

  invalidate(): void {
    this.cachedKey = ''
  }

  render(width: number): string[] {
    // Keyed on the palette as well as the width: see `UserMessage.render`.
    const key = `${width}:${this.theme.scheme}`

    if (key === this.cachedKey) {
      return this.lines
    }

    this.cachedKey = key
    this.lines = this.build(width)

    return this.lines
  }

  private build(width: number): string[] {
    const { theme } = this
    const out: string[] = ['', theme.bold(theme.fg('text', 'Help')), '']
    const names = this.data.commands.map(command =>
      command.argumentHint ? `/${command.name} ${command.argumentHint}` : `/${command.name}`
    )
    const column = Math.max(0, ...names.map(name => visibleWidth(name)))

    this.data.commands.forEach((command, index) => {
      out.push(...this.row(names[index] ?? `/${command.name}`, command.description, column, width))
    })

    if (this.data.catalog.length) {
      out.push(
        '',
        ...wrapToWidth(`From the CLI: ${this.data.catalog.map(name => `/${name}`).join(' ')}`, width).map(line =>
          theme.fg('muted', line)
        )
      )
    }

    if (this.data.skillCount) {
      out.push(
        '',
        ...wrapToWidth(`${this.data.skillCount} skill commands are available.`, width).map(line =>
          theme.fg('muted', line)
        )
      )
    }

    out.push('', theme.bold(theme.fg('text', 'Keys')), '')

    const keys = this.data.hotkeys.map(binding => this.keybindings.getKeys(binding).join('/'))
    const keyColumn = Math.max(0, ...keys.map(text => visibleWidth(text)))

    this.data.hotkeys.forEach((binding, index) => {
      out.push(
        ...this.row(keys[index] ?? '', this.keybindings.getDefinition(binding)?.description ?? '', keyColumn, width)
      )
    })

    out.push('')

    // A backstop, not the wrapping: the renderer throws on a line wider than
    // the terminal, and no heading or label is worth that.
    return out
      .flatMap(line => line.split('\n'))
      .map(line => (visibleWidth(line) > width ? truncateToWidth(line, width, '') : line))
  }

  /** `<name>  <what it does>`: the name in the terminal's own foreground, the
   *  description in grey. Neither column is identity, so neither is tinted. */
  private row(left: string, right: string, column: number, width: number): string[] {
    const { theme } = this

    if (width < TWO_COLUMN_MIN || column + 4 > width) {
      // Stack and wrap rather than truncate: a description clipped to six
      // characters is worse than one on its own lines.
      const indent = Math.min(4, Math.max(0, width - 1))

      return [
        truncateToWidth(theme.fg('text', left), width, '…'),
        ...wrapToWidth(right, width - indent).map(line => theme.fg('muted', `${' '.repeat(indent)}${line}`))
      ]
    }

    // Columns are measured in display cells, not characters: a CJK label is
    // two cells wide per character and `padEnd` would under-pad it.
    const padded = left + ' '.repeat(Math.max(0, column - visibleWidth(left)))
    const room = width - column - 2

    return [`${theme.fg('text', padded)}  ${truncateToWidth(theme.fg('muted', right), Math.max(1, room), '…')}`]
  }
}

/**
 * Wrap to a display width, breaking a word that is wider than the line.
 *
 * Wrapping on spaces alone is not enough: an unbroken CJK description, or a
 * long catalogue command name, is one word wider than the whole viewport, and
 * the renderer refuses a line wider than the terminal. Width is counted in
 * cells, so one CJK character costs two.
 */
export function wrapToWidth(text: string, width: number): string[] {
  const max = Math.max(1, width)
  const lines: string[] = []
  let line = ''

  for (const word of text.split(/\s+/).filter(Boolean)) {
    const candidate = line ? `${line} ${word}` : word

    if (visibleWidth(candidate) <= max) {
      line = candidate
      continue
    }

    if (line) {
      lines.push(line)
      line = ''
    }

    let rest = word

    while (visibleWidth(rest) > max) {
      // `strict` keeps a double-width character whole rather than splitting it
      // across two lines, so a head can come back one cell short.
      const head = sliceByColumn(rest, 0, max, true) || [...rest][0] || ''

      if (!head) {
        break
      }

      lines.push(head)
      rest = rest.slice(head.length)
    }

    line = rest
  }

  if (line) {
    lines.push(line)
  }

  return lines.length > 0 ? lines : ['']
}
