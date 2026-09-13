// The product's own wording: the questions the empty prompt suggests, the verb
// that heads a tool panel, and the phrases a long wait is allowed to use.

import { stripTerminalSequences, TuiMainScreen, visibleWidth } from '@earendil-works/pi-tui'
import { describe, expect, it } from 'vitest'

import { HarnessEditor } from '../components/editor.js'
import { ToolExecution } from '../components/toolExecution.js'
import { GENERIC_WAIT, waitPhrases } from '../content/charms.js'
import { pick } from '../content/pick.js'
import { pickPlaceholder, PLACEHOLDERS } from '../content/placeholders.js'
import { pickVerb, VERBS } from '../content/verbs.js'
import { Theme } from '../theme.js'
import { FakeTerminal } from './fakes.js'

const theme = new Theme('dark', 0)

function editor(placeholder?: string) {
  const tui = new TuiMainScreen(new FakeTerminal(80, 24))
  const harnessEditor = new HarnessEditor(tui, theme.editorTheme(), {
    ...(placeholder ? { placeholder } : {})
  })

  tui.setFocus(harnessEditor)

  return harnessEditor
}

/** The editor's single content row, without its border. */
const promptRow = (component: HarnessEditor, width = 80) => stripTerminalSequences(component.render(width)[1] ?? '')

describe('the editor placeholder', () => {
  it('suggests one of this product’s own questions', () => {
    expect(PLACEHOLDERS.length).toBeGreaterThan(20)
    expect(PLACEHOLDERS).toContain(pickPlaceholder())
    // Deterministic when the caller brings its own source of randomness.
    expect(pickPlaceholder(() => 0)).toBe(PLACEHOLDERS[0])
    expect(pickPlaceholder(() => 0.999999)).toBe(PLACEHOLDERS.at(-1))
  })

  it('shows the suggestion in an empty prompt and nowhere else', () => {
    const component = editor('Design a VHH against human CRLF2')

    expect(promptRow(component)).toContain('Design a VHH against human CRLF2')

    component.setText('humanize this')
    expect(promptRow(component)).toContain('humanize this')
    expect(promptRow(component)).not.toContain('Design a VHH')

    component.setText('')
    expect(promptRow(component)).toContain('Design a VHH')
  })

  it('is exactly as wide as the row it sits in, however long the sentence', () => {
    const component = editor(PLACEHOLDERS.reduce((a, b) => (a.length > b.length ? a : b)))

    for (const width of [40, 72, 120]) {
      const lines = component.render(width)

      for (const line of lines) {
        expect(visibleWidth(line)).toBe(width)
      }
    }
  })

  it('draws nothing extra when there is no suggestion', () => {
    const bare = editor()
    const suggested = editor('x')

    expect(promptRow(bare).trim()).toBe('')
    expect(promptRow(suggested).trim()).toBe('x')
  })

  it('takes a new suggestion when the app hands it one', () => {
    const component = editor('first')

    component.setPlaceholder('second')
    expect(promptRow(component)).toContain('second')
    expect(promptRow(component)).not.toContain('first')
  })
})

describe('the activity verbs', () => {
  it('heads a tool panel with the tool name alone, which is already the verb', () => {
    const tui = new TuiMainScreen(new FakeTerminal(80, 24))
    const known = new ToolExecution(theme, tui, { display: '/etc/hosts', name: 'read' })
    const unknown = new ToolExecution(theme, tui, { display: 'anything', name: 'fold_backbone' })
    const header = (tool: ToolExecution) => stripTerminalSequences(tool.render(80)[1] ?? '')

    // "reading read /etc/hosts" said the same thing twice.
    expect(header(known).trim()).toMatch(/^read \/etc\/hosts/)
    expect(header(unknown).trim()).toMatch(/^fold_backbone anything/)
  })

  it('keeps a domain vocabulary for the turn’s own verb', () => {
    expect(VERBS.length).toBeGreaterThan(100)
    expect(new Set(VERBS).size).toBe(VERBS.length)
    expect(VERBS.some(verb => /fold/.test(verb))).toBe(true)
    expect(VERBS.some(verb => /CDR|paratope|epitope/.test(verb))).toBe(true)
    expect(VERBS).toContain(pickVerb())
  })

  it('picks the first entry rather than nothing from an empty draw', () => {
    expect(pick(['only'], () => 1)).toBe('only')
  })
})

describe('the long-wait phrases', () => {
  it('ships the old TUI’s three domain phrases and no invented ones', () => {
    const domain = new Set(['fold_structure', 'rank_designs', 'lease_gpu'].flatMap(name => waitPhrases(name)))

    expect([...domain].sort()).toEqual(['scoring candidates', 'still folding', 'waiting on the compute service'])
  })

  it('says something neutral about a tool that claims no domain', () => {
    expect(waitPhrases('grep')).toEqual(GENERIC_WAIT)
  })
})
