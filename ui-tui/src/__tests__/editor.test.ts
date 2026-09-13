import { TuiMainScreen } from '@earendil-works/pi-tui'
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { CommandRegistry } from '../commands/index.js'
import { completionWord, HarnessAutocompleteProvider, pathPrefixBefore } from '../components/autocomplete.js'
import { HarnessEditor } from '../components/editor.js'
import { InputHistory } from '../lib/history.js'
import { Theme } from '../theme.js'
import { createTestContext } from './commandContext.js'
import { FakeTerminal } from './fakes.js'

const theme = new Theme('dark', 0)
const registry = new CommandRegistry()

function suggest(provider: HarnessAutocompleteProvider, line: string, force = false) {
  return provider.getSuggestions([line], 0, line.length, { force, signal: new AbortController().signal })
}

describe('HarnessEditor', () => {
  let dir: string
  let terminal: FakeTerminal
  let tui: TuiMainScreen

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'harness-editor-'))
    terminal = new FakeTerminal(80, 24)
    tui = new TuiMainScreen(terminal)
  })

  afterEach(() => {
    rmSync(dir, { force: true, recursive: true })
  })

  it('draws a plain rule for a border: the turn status has its own line', () => {
    const editor = new HarnessEditor(tui, theme.editorTheme())

    expect(editor.render(40)[0]).toBe('─'.repeat(40))
  })

  it('seeds the prompt ring from the history file and appends to it', () => {
    const file = join(dir, 'history')
    const seed = new InputHistory(file)

    seed.append('an older prompt')

    const editor = new HarnessEditor(tui, theme.editorTheme(), { history: new InputHistory(file) })

    editor.rememberPrompt('a new prompt')

    expect(new InputHistory(file).load()).toEqual(['an older prompt', 'a new prompt'])
  })

  it('takes back what the external editor saved, and stops the TUI while it runs', async () => {
    const editor = new HarnessEditor(tui, theme.editorTheme())
    const script = join(dir, 'fake-editor.sh')

    writeFileSync(script, '#!/bin/sh\nprintf "edited elsewhere" > "$1"\n')
    chmodSync(script, 0o755)

    editor.setText('draft')

    const stopped: unknown[] = []
    const originalStop = tui.stop.bind(tui)

    tui.stop = options => {
      stopped.push(options)
      originalStop(options)
    }

    expect(await editor.openExternal(['sh', script])).toBe(true)
    expect(editor.getText()).toBe('edited elsewhere')
    expect(stopped).toEqual([{ preserveScreen: true }])

    tui.stop = originalStop
    originalStop()
  })

  it('leaves the prompt alone when the editor exits without saving a change', async () => {
    const editor = new HarnessEditor(tui, theme.editorTheme())

    editor.setText('draft')

    expect(await editor.openExternal(['sh', '-c', 'exit 1'])).toBe(false)
    expect(editor.getText()).toBe('draft')

    tui.stop()
  })

  it('hands the submitted text over with paste markers already expanded', () => {
    const editor = new HarnessEditor(tui, theme.editorTheme())
    const onSubmit = vi.fn()

    editor.onSubmit = onSubmit
    editor.handleInput('\x1b[200~pasted body\x1b[201~')
    editor.handleInput('\r')

    expect(onSubmit).toHaveBeenCalledWith('pasted body')
  })

  it('treats a trailing backslash before Enter as a line continuation', () => {
    const editor = new HarnessEditor(tui, theme.editorTheme())
    const submitted: string[] = []

    editor.onSubmit = text => submitted.push(text)
    editor.setText('first line\\')
    // pi's Editor puts the cursor at the end of the text it is given.
    editor.handleInput('\r')

    expect(submitted).toEqual([])

    editor.setText(`${editor.getText()}second line`)
    editor.handleInput('\r')

    expect(submitted).toEqual(['first line\nsecond line'])
  })
})

describe('path prefixes', () => {
  it('picks up @ and path-shaped tokens', () => {
    expect(pathPrefixBefore('look at @src/app')).toBe('@src/app')
    expect(pathPrefixBefore('read ./notes')).toBe('./notes')
    expect(pathPrefixBefore('read ~/notes')).toBe('~/notes')
    expect(pathPrefixBefore('src/app.ts')).toBe('src/app.ts')
  })

  it('ignores plain words unless completion is forced', () => {
    expect(pathPrefixBefore('hello')).toBeNull()
    expect(pathPrefixBefore('hello', true)).toBe('hello')
  })

  it('strips the trigger from the word sent to the gateway', () => {
    expect(completionWord('@src/')).toBe('src/')
    expect(completionWord('@"my dir/')).toBe('my dir/')
    expect(completionWord('"my dir/')).toBe('my dir/')
    expect(completionWord('src/')).toBe('src/')
  })
})

describe('autocomplete', () => {
  let dir: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'harness-complete-'))
    mkdirSync(join(dir, 'src'))
    writeFileSync(join(dir, 'src', 'app.ts'), 'x')
    writeFileSync(join(dir, 'README.md'), 'x')
  })

  afterEach(() => {
    rmSync(dir, { force: true, recursive: true })
  })

  function provider(completePath?: (word: string) => Promise<{ label: string; value: string }[]>) {
    const { context } = createTestContext()

    return new HarnessAutocompleteProvider({
      basePath: dir,
      commands: registry.slashCommands(context),
      ...(completePath ? { completePath } : {})
    })
  }

  it('offers every command after a bare slash', async () => {
    const result = await suggest(provider(), '/')

    expect(result?.prefix).toBe('/')

    const names = result!.items.map(item => item.value)

    expect(names).toContain('help')
    expect(names).toContain('thinking')
    // The CLI surface is not here any more; `compare` is a terminal command.
    expect(names).not.toContain('compare')
  })

  it('completes a namespaced action from its feature', async () => {
    const result = await suggest(provider(), '/task:')

    // pi matches the whole name, colon included, so `feature:action` needs no
    // special handling in the popup — but it is the shape the set now uses,
    // so it is pinned.
    expect(result?.items.map(item => item.value).sort()).toEqual(['task:follow', 'task:logs'])
  })

  it('completes a feature and its actions together', async () => {
    const result = await suggest(provider(), '/mcp')

    expect(result?.items.map(item => item.value).sort()).toEqual(['mcp', 'mcp:reload'])
  })

  it('filters as the name is typed, and shows the hint and description', async () => {
    const result = await suggest(provider(), '/think')

    expect(result!.items.map(item => item.value)).toEqual(['thinking'])
    expect(result!.items[0]!.description).toContain('[level]')
    expect(result!.items[0]!.description).toContain('thinking level')
  })

  it('offers a command its own argument completions', async () => {
    const result = await suggest(provider(), '/thinking h')

    expect(result!.items.map(item => item.value)).toEqual(['high'])
    expect(result!.prefix).toBe('h')
  })

  it('says nothing for a name that matches nothing', async () => {
    expect(await suggest(provider(), '/zzzz')).toBeNull()
  })

  it('asks the gateway for paths first', async () => {
    const completePath = vi.fn(async () => [{ label: 'src/app.ts', value: 'src/app.ts' }])
    const result = await suggest(provider(completePath), 'open @src/')

    expect(completePath).toHaveBeenCalledWith('src/')
    expect(result!.items.map(item => item.value)).toEqual(['src/app.ts'])
    expect(result!.prefix).toBe('@src/')
  })

  it('falls back to the local walk when the gateway has nothing', async () => {
    const completePath = vi.fn(async () => [])
    const result = await suggest(provider(completePath), 'open ./RE')

    expect(completePath).toHaveBeenCalledWith('./RE')
    expect(result!.items.map(item => item.label)).toContain('README.md')
  })

  it('falls back when the gateway call fails', async () => {
    const completePath = vi.fn(async () => {
      throw new Error('no such method')
    })

    const result = await suggest(provider(completePath), 'open ./RE')

    expect(result!.items.map(item => item.label)).toContain('README.md')
  })

  it('completes a path with no gateway completer at all', async () => {
    const result = await suggest(provider(), 'open ./src/')

    expect(result!.items.map(item => item.label)).toContain('app.ts')
  })

  it('never sends a slash command to the path completer', async () => {
    const completePath = vi.fn(async () => [{ label: 'x', value: 'x' }])

    await suggest(provider(completePath), '/help')

    expect(completePath).not.toHaveBeenCalled()
  })

  it('completes a command name into the line with a trailing space', () => {
    const applied = provider().applyCompletion(['/thin'], 0, 5, { label: 'thinking', value: 'thinking' }, '/thin')

    expect(applied.lines[0]).toBe('/thinking ')
  })
})
