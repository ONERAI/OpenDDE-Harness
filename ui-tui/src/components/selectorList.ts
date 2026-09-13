// The list every selector is made of: a title, a search input, rows between two
// horizontal rules, and a line of key hints — pi's selector shape.
//
// Search is fuzzy. `SelectList.setFilter` is a prefix match on the row value,
// which would hide `claude-opus-5` from the query `opus`, so the rows are
// filtered here with pi's `fuzzyFilter` and the list child is rebuilt from the
// result, exactly as pi's own thinking and model selectors do.

import type { Component, Focusable, KeybindingsManager, SelectItem, TUI } from '@earendil-works/pi-tui'

import { Container, fuzzyFilter, Input, SelectList, Spacer, Text } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

import { joinHints, keyHint, rawKeyHint } from '../lib/keybindings.js'

/** One row. `id` is the caller's own identifier and is never parsed here;
 *  `kind` is how the owning selector tells an action apart from a thing. */
export interface SelectorRow {
  description?: string
  id: string
  kind: 'action' | 'item'
  label: string
  searchText: string
}

export interface SelectorListOptions {
  /** Shown instead of the list when there are no rows at all. */
  emptyText?: string
  keybindings: KeybindingsManager
  onCancel: () => void
  onSelect: (id: string) => void
  placeholder?: string
  rows: SelectorRow[]
  /** Dim line under the title. */
  subtitle?: string
  theme: Theme
  title: string
  tui: TUI
}

/** Rows that fit without pushing the transcript off the screen. */
export function visibleRowCount(terminalRows: number): number {
  return Math.min(10, Math.max(1, terminalRows - 10))
}

/** The standard hint line: navigate, select, cancel, plus whatever the stage adds. */
export function standardHints(theme: Theme, keybindings: KeybindingsManager, extra: string[] = []): string {
  return joinHints([
    rawKeyHint(theme, '↑↓', 'navigate'),
    keyHint(keybindings, theme, 'tui.select.confirm', 'select'),
    keyHint(keybindings, theme, 'tui.select.cancel', 'back'),
    ...extra
  ])
}

export class SelectorList extends Container implements Focusable {
  private readonly border: Component
  private readonly errorText = new OptionalText()
  private readonly hintText = new OptionalText()
  private readonly input: Input
  private readonly listSlot = new Container()
  private readonly statusText = new OptionalText()
  private readonly subtitleText = new OptionalText()
  private readonly titleText = new Text('', 1, 0)

  private allRows: SelectorRow[]
  private builtFor = 0
  private filtered: SelectorRow[] = []
  private isFocused = false
  private list: SelectList | undefined
  private readonly options: SelectorListOptions
  private selected: null | string = null

  constructor(options: SelectorListOptions) {
    super()
    this.options = options
    this.allRows = options.rows
    this.border = new HorizontalRule(options.theme)
    this.input = new Input({
      placeholder: options.placeholder ?? 'type to search',
      placeholderStyle: text => options.theme.fg('muted', text),
      prompt: options.theme.fg('muted', '❯ ')
    })
    this.input.onSubmit = () => this.confirm()
    this.input.onEscape = () => this.options.onCancel()

    this.setTitle(options.title)

    if (options.subtitle) {
      this.setSubtitle(options.subtitle)
    }

    this.addChild(this.border)
    this.addChild(new Spacer(1))
    this.addChild(this.titleText)
    this.addChild(this.subtitleText)
    this.addChild(new Spacer(1))
    this.addChild(this.input)
    this.addChild(new Spacer(1))
    this.addChild(this.listSlot)
    this.addChild(this.statusText)
    this.addChild(this.errorText)
    this.addChild(new Spacer(1))
    this.addChild(this.hintText)
    this.addChild(new HorizontalRule(options.theme))

    this.setHints(standardHints(options.theme, options.keybindings))
    this.rebuild()
  }

  get focused(): boolean {
    return this.isFocused
  }

  set focused(value: boolean) {
    this.isFocused = value
    this.input.focused = value
  }

  get query(): string {
    return this.input.getValue()
  }

  get selectedId(): null | string {
    return this.selected
  }

  setTitle(text: string): void {
    this.titleText.setText(this.options.theme.bold(this.options.theme.fg('text', text)))
  }

  setSubtitle(text: string): void {
    this.subtitleText.setText(text ? this.options.theme.fg('muted', text) : '')
  }

  setHints(text: string): void {
    this.hintText.setText(text)
  }

  /** A dim line under the list: "loading…", "3 providers configured". */
  setStatus(text?: string): void {
    this.statusText.setText(text ? this.options.theme.fg('muted', text) : '')
  }

  setError(text?: string): void {
    this.errorText.setText(text ? this.options.theme.fg('error', text) : '')
  }

  setQuery(text: string): void {
    this.input.setValue(text)
    this.rebuild()
  }

  /** Replace the data. The selected row keeps its identity when it survives. */
  setRows(rows: SelectorRow[]): void {
    this.allRows = rows
    this.rebuild()
  }

  handleInput(data: string): void {
    const kb = this.options.keybindings

    if (kb.matches(data, 'tui.select.cancel')) {
      this.options.onCancel()

      return
    }

    if (kb.matches(data, 'tui.select.confirm')) {
      this.confirm()

      return
    }

    if (kb.matches(data, 'tui.select.up')) {
      this.moveBy(-1)

      return
    }

    if (kb.matches(data, 'tui.select.down')) {
      this.moveBy(1)

      return
    }

    if (kb.matches(data, 'tui.select.pageUp')) {
      this.moveBy(-this.pageSize())

      return
    }

    if (kb.matches(data, 'tui.select.pageDown')) {
      this.moveBy(this.pageSize())

      return
    }

    // Everything else is search text — `d`, `j` and `k` included, because this
    // list has an input in it and they are letters people type.
    this.input.handleInput(data)
    this.rebuild()
  }

  override render(width: number): string[] {
    const wanted = this.pageSize()

    if (wanted !== this.builtFor) {
      // The terminal was resized. Rebuild the list, not the query or the
      // selection: both survive the new height.
      this.rebuild()
    }

    return super.render(width)
  }

  private pageSize(): number {
    return visibleRowCount(this.options.tui.terminal.rows)
  }

  private confirm(): void {
    if (this.selected === null) {
      return
    }

    this.options.onSelect(this.selected)
  }

  private moveBy(delta: number): void {
    if (this.filtered.length === 0) {
      return
    }

    const index = this.filtered.findIndex(row => row.id === this.selected)
    const next = Math.min(this.filtered.length - 1, Math.max(0, (index < 0 ? 0 : index) + delta))

    this.selected = this.filtered[next]?.id ?? null
    this.list?.setSelectedIndex(next)
  }

  private rebuild(): void {
    const query = this.input.getValue().trim()

    // Things first, in match order; actions after them, in the order they were
    // given. An action row matches a query like any other, and scored among
    // the things it landed wherever its score fell -- "Add…" between two
    // providers -- rather than where pi keeps such a row: at the end.
    const matched = query ? fuzzyFilter(this.allRows, query, row => row.searchText) : this.allRows

    this.filtered = [...matched.filter(row => row.kind !== 'action'), ...matched.filter(row => row.kind === 'action')]

    if (!this.filtered.some(row => row.id === this.selected)) {
      this.selected = this.filtered[0]?.id ?? null
    }

    this.builtFor = this.pageSize()
    this.listSlot.clear()

    if (this.filtered.length === 0) {
      this.listSlot.addChild(
        new Text(
          this.options.theme.fg('muted', query ? '  no match' : `  ${this.options.emptyText ?? 'nothing here'}`),
          1,
          0
        )
      )
      this.list = undefined

      return
    }

    const items: SelectItem[] = this.filtered.map(row => ({
      label: row.label,
      value: row.id,
      ...(row.description ? { description: row.description } : {})
    }))

    const list = new SelectList(items, this.builtFor, this.options.theme.selectListTheme())
    const index = Math.max(
      0,
      this.filtered.findIndex(row => row.id === this.selected)
    )

    list.setSelectedIndex(index)
    list.onSelectionChange = item => {
      this.selected = item.value
    }
    list.onSelect = item => this.options.onSelect(item.value)
    list.onCancel = () => this.options.onCancel()

    this.list = list
    this.listSlot.addChild(list)
  }
}

/** Wrapped text that takes up no room at all when it is empty, so an absent
 *  subtitle or error does not leave a blank line behind. */
export class OptionalText implements Component {
  private inner = new Text('', 1, 0)
  private text = ''

  setText(text: string): void {
    this.text = text
    this.inner.setText(text)
  }

  invalidate(): void {
    this.inner.invalidate()
  }

  render(width: number): string[] {
    return this.text ? this.inner.render(width) : []
  }
}

/** A rule the width of the viewport, pi's selector frame. */
export class HorizontalRule implements Component {
  constructor(private readonly theme: Theme) {}

  invalidate(): void {
    // Nothing cached.
  }

  render(width: number): string[] {
    return [this.theme.fg('border', '─'.repeat(Math.max(1, width)))]
  }
}
