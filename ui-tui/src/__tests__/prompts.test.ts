// The broker prompts: one answer each, whatever order answer, expiry and close
// arrive in.

import { describe, expect, it } from 'vitest'

import { SelectorHost } from '../components/selector.js'
import { CONFIRM_VISIBLE_MS, PromptCoordinator } from '../prompts/coordinator.js'
import { decodeApprovalRequest, decodeConfirmRequest } from '../prompts/decode.js'
import { createHarness, KEYS, settle } from './selectorHarness.js'

const SESSION = 'tui:abc'

function build(results: Record<string, unknown> = {}) {
  const harness = createHarness({
    'approval.respond': { ok: true },
    'clarify.respond': { ok: true },
    'confirm.respond': { ok: true },
    ...results
  })
  const host = new SelectorHost({
    editor: harness.editor,
    editorContainer: harness.editorContainer,
    onChange: () => {},
    session: () => ({ epoch: harness.session.epoch, id: harness.session.id }),
    tui: harness.tui
  })
  const prompts = new PromptCoordinator({
    clock: harness.clock,
    gateway: harness.gateway,
    host,
    keybindings: harness.deps.keybindings,
    sessionId: () => harness.session.id,
    theme: harness.deps.theme,
    transcript: harness.deps.transcript,
    tui: harness.tui
  })

  const screen = () => harness.draw(harness.editorContainer)
  const type = (data: string) => {
    const focused = harness.tui.getFocusedComponent()

    ;(focused as { handleInput?(data: string): void }).handleInput?.(data)
  }

  return { harness, host, prompts, screen, type }
}

function approval(overrides: Record<string, unknown> = {}, clockMs = 1_000_000) {
  return {
    action_digest: 'digest',
    approval_id: 'a1',
    command: 'rm -rf /tmp/x',
    conversation_id: SESSION,
    created_at: clockMs / 1000,
    description: 'Delete files using a shell command',
    expires_at: clockMs / 1000 + 30,
    tool_call_id: 'tool-1',
    turn_id: 'turn-1',
    ...overrides
  }
}

const calls = (harness: ReturnType<typeof createHarness>, method: string) =>
  harness.transport.calls.filter(call => call.method === method)

describe('approval prompts', () => {
  it('shows the request and sends exactly one allow', async () => {
    const { harness, prompts, screen, type } = build()

    prompts.handleNotification('approval.request', approval())

    expect(screen()).toContain('Approval required')
    expect(screen()).toContain('rm -rf /tmp/x')

    type('1')
    type('1')
    type(KEYS.enter)
    await settle()

    expect(calls(harness, 'approval.respond')).toHaveLength(1)
    expect(calls(harness, 'approval.respond')[0]?.params).toEqual({
      approval_id: 'a1',
      choice: 'allow',
      session_id: SESSION
    })
    expect(harness.editorContainer.children[0]).toBe(harness.editor)
  })

  it('denies on escape', async () => {
    const { harness, prompts, type } = build()

    prompts.handleNotification('approval.request', approval())
    type(KEYS.escape)
    await settle()

    expect(calls(harness, 'approval.respond')[0]?.params).toMatchObject({ choice: 'deny' })
  })

  it('denies at the deadline, and a later Enter changes nothing', async () => {
    const { harness, prompts, type } = build()

    prompts.handleNotification('approval.request', approval())
    harness.clock.advance(30_000)
    type(KEYS.enter)
    await settle()

    const sent = calls(harness, 'approval.respond')

    expect(sent).toHaveLength(1)
    expect(sent[0]?.params).toMatchObject({ choice: 'deny' })
    expect(harness.printed.some(line => line.text.includes('expired'))).toBe(true)
  })

  it('counts down from the request deadline, not from when it was shown', () => {
    const { harness, prompts, screen } = build()

    // Ten seconds of the thirty were already spent in transit.
    prompts.handleNotification('approval.request', approval({ expires_at: harness.clock.nowMs() / 1000 + 20 }))

    expect(screen()).toContain('(20s)')
  })

  it('refuses a request whose deadline has already passed', async () => {
    const { harness, prompts } = build()

    prompts.handleNotification('approval.request', approval({ expires_at: harness.clock.nowMs() / 1000 - 5 }))
    await settle()

    expect(calls(harness, 'approval.respond')[0]?.params).toMatchObject({ choice: 'deny' })
    expect(harness.editorContainer.children[0]).toBe(harness.editor)
  })

  it('denies a request for another conversation without showing it', async () => {
    const { harness, prompts } = build()

    prompts.handleNotification('approval.request', approval({ approval_id: 'other', conversation_id: 'tui:zzz' }))
    await settle()

    expect(calls(harness, 'approval.respond')[0]?.params).toMatchObject({
      approval_id: 'other',
      choice: 'deny',
      session_id: 'tui:zzz'
    })
    expect(harness.editorContainer.children[0]).toBe(harness.editor)
  })

  it('drops a malformed request instead of drawing an unanswerable prompt', () => {
    const { harness, prompts } = build()

    prompts.handleNotification('approval.request', { command: 'ls' })

    expect(decodeApprovalRequest({ command: 'ls' })).toBeNull()
    expect(calls(harness, 'approval.respond')).toHaveLength(0)
    expect(harness.editorContainer.children[0]).toBe(harness.editor)
  })

  it('closes on approval.closed without answering, and ignores another id', async () => {
    const { harness, prompts } = build()

    prompts.handleNotification('approval.request', approval())
    prompts.handleNotification('approval.closed', {
      approval_id: 'someone-else',
      conversation_id: SESSION,
      reason: 'timeout'
    })

    expect(prompts.blocked).toBe(true)

    prompts.handleNotification('approval.closed', { approval_id: 'a1', conversation_id: SESSION, reason: 'timeout' })
    await settle()

    expect(calls(harness, 'approval.respond')).toHaveLength(0)
    expect(prompts.blocked).toBe(false)
  })

  it('ignores a repeated approval id', () => {
    const { prompts } = build()

    prompts.handleNotification('approval.request', approval())
    prompts.handleNotification('approval.request', approval())

    expect(prompts.blocked).toBe(true)
  })

  it('puts a long command in the transcript and says so', () => {
    const { harness, prompts, screen } = build()
    const command = `echo ${'x'.repeat(400)}`

    prompts.handleNotification('approval.request', approval({ command }))

    expect(harness.printed.some(line => line.title === 'Command awaiting approval')).toBe(true)
    expect(screen()).toContain('the full command is in the transcript above')
  })

  it('shows control characters as text rather than passing them through', () => {
    const { prompts, screen } = build()

    prompts.handleNotification('approval.request', approval({ command: 'echo \u001b[31mred' }))

    // The escape byte is drawn as a dot; the terminal is never asked to run it.
    expect(screen()).toContain('\u00b7[31mred')
  })

  it('reports a refusal rather than treating it as approval', async () => {
    const { harness, prompts, type } = build({ 'approval.respond': { ok: false } })

    prompts.handleNotification('approval.request', approval())
    type('1')
    await settle()

    expect(harness.printed.some(line => line.text.includes('already closed'))).toBe(true)
  })
})

describe('clarify prompts', () => {
  const question = { choices: ['red', 'blue'], conversation_id: SESSION, question: 'Which?', request_id: 'q1' }

  it('answers with the choice text, not its number, and sends no conversation id', async () => {
    const { harness, prompts, type } = build()

    prompts.handleNotification('clarify.request', question)
    type('2')
    await settle()

    expect(calls(harness, 'clarify.respond')[0]?.params).toEqual({ answer: 'blue', request_id: 'q1' })
  })

  it('has no countdown, because the broker never sends one', () => {
    const { prompts, screen } = build()

    prompts.handleNotification('clarify.request', question)

    expect(screen()).not.toMatch(/\(\d+s\)/)
  })

  it('takes free text from Other, where digits are text', async () => {
    const { harness, prompts, type } = build()

    prompts.handleNotification('clarify.request', question)
    type(KEYS.down)
    type(KEYS.down)
    type(KEYS.enter)
    type('4')
    type('2')
    type(KEYS.enter)
    await settle()

    expect(calls(harness, 'clarify.respond')[0]?.params).toEqual({ answer: '42', request_id: 'q1' })
  })

  it('goes back to the choices from Other, and cancels from the root', async () => {
    const { harness, prompts, screen, type } = build()

    prompts.handleNotification('clarify.request', question)
    type(KEYS.down)
    type(KEYS.down)
    type(KEYS.enter)
    type(KEYS.escape)

    expect(screen()).toContain('red')

    type(KEYS.escape)
    await settle()

    expect(calls(harness, 'clarify.respond')[0]?.params).toEqual({ cancelled: true, request_id: 'q1' })
  })

  it('replaces the previous question for the same conversation without answering it', async () => {
    const { harness, prompts, screen } = build()

    prompts.handleNotification('clarify.request', question)
    prompts.handleNotification('clarify.request', { ...question, question: 'Which now?', request_id: 'q2' })
    await settle()

    expect(calls(harness, 'clarify.respond')).toHaveLength(0)
    expect(screen()).toContain('Which now?')
  })

  it('cancels a question from another conversation by its own id', async () => {
    const { harness, prompts } = build()

    prompts.handleNotification('clarify.request', { ...question, conversation_id: 'tui:zzz', request_id: 'q9' })
    await settle()

    expect(calls(harness, 'clarify.respond')[0]?.params).toEqual({ cancelled: true, request_id: 'q9' })
  })
})

describe('confirm prompts', () => {
  const ask = { default: true, prompt: 'Reload MCP servers?', request_id: 'c1' }

  it('preselects the default and sends it on Enter', async () => {
    const { harness, prompts, screen, type } = build()

    prompts.handleNotification('confirm.request', ask)

    expect(screen()).toContain('default')

    type(KEYS.enter)
    await settle()

    expect(calls(harness, 'confirm.respond')[0]?.params).toEqual({ answer: true, request_id: 'c1' })
  })

  it('sends the request default at the deadline, including true', async () => {
    const { harness, prompts } = build()

    prompts.handleNotification('confirm.request', ask)
    harness.clock.advance(CONFIRM_VISIBLE_MS)
    await settle()

    expect(calls(harness, 'confirm.respond')[0]?.params).toEqual({ answer: true, request_id: 'c1' })
  })

  it('sends false at the deadline when false is the default', async () => {
    const { harness, prompts } = build()

    prompts.handleNotification('confirm.request', { ...ask, default: false })
    harness.clock.advance(CONFIRM_VISIBLE_MS)
    await settle()

    expect(calls(harness, 'confirm.respond')[0]?.params).toEqual({ answer: false, request_id: 'c1' })
  })

  it('does not extend the countdown when keys are pressed', () => {
    const { harness, prompts, screen, type } = build()

    prompts.handleNotification('confirm.request', ask)
    harness.clock.advance(10_000)
    type('x')
    type('z')

    expect(screen()).toContain('(20s)')
  })

  it('is explicit cancellation on escape, not the default', async () => {
    const { harness, prompts, type } = build()

    prompts.handleNotification('confirm.request', ask)
    type(KEYS.escape)
    await settle()

    expect(calls(harness, 'confirm.respond')[0]?.params).toEqual({ answer: false, request_id: 'c1' })
  })

  it('refuses a default that is not a boolean', () => {
    const { harness, prompts } = build()

    expect(decodeConfirmRequest({ default: 'yes', prompt: 'x', request_id: 'c2' })).toBeNull()

    prompts.handleNotification('confirm.request', { default: 'yes', prompt: 'x', request_id: 'c2' })

    expect(prompts.blocked).toBe(false)
    expect(harness.editorContainer.children[0]).toBe(harness.editor)
  })

  it('drops a confirm left over from a command that has finished', async () => {
    const { harness, prompts } = build()
    const token = prompts.beginCommand()

    prompts.handleNotification('confirm.request', ask)
    prompts.endCommand(token)
    await settle()

    expect(calls(harness, 'confirm.respond')).toHaveLength(0)
    expect(prompts.blocked).toBe(false)
  })
})

describe('queueing and lifecycle', () => {
  it("keeps a second prompt's own deadline while it waits", async () => {
    const { harness, prompts, screen, type } = build()

    prompts.handleNotification('approval.request', approval())
    prompts.handleNotification('confirm.request', { default: false, prompt: 'Second?', request_id: 'c1' })

    harness.clock.advance(10_000)
    type(KEYS.escape)
    await settle()

    // The confirm waited ten of its thirty seconds behind the approval.
    expect(screen()).toContain('Second?')
    expect(screen()).toContain('(20s)')
  })

  it('answers a queued approval that expired before it was ever shown', async () => {
    const { harness, prompts } = build()

    prompts.handleNotification('approval.request', approval())
    prompts.handleNotification('approval.request', approval({ approval_id: 'a2' }))

    harness.clock.advance(30_000)
    await settle(8)

    const sent = calls(harness, 'approval.respond')

    expect(sent.map(call => (call.params as { approval_id: string }).approval_id).sort()).toEqual(['a1', 'a2'])
    expect(prompts.blocked).toBe(false)
  })

  it('cancels everything outstanding on a session switch', async () => {
    const { harness, prompts } = build()

    prompts.handleNotification('approval.request', approval())
    prompts.handleNotification('clarify.request', {
      choices: [],
      conversation_id: SESSION,
      question: 'Which?',
      request_id: 'q1'
    })
    prompts.handleNotification('confirm.request', { default: true, prompt: 'p', request_id: 'c1' })

    prompts.onSessionChange()
    await settle(8)

    expect(calls(harness, 'approval.respond')[0]?.params).toMatchObject({ choice: 'deny' })
    expect(calls(harness, 'clarify.respond')[0]?.params).toEqual({ cancelled: true, request_id: 'q1' })
    expect(calls(harness, 'confirm.respond')[0]?.params).toEqual({ answer: false, request_id: 'c1' })
  })

  it('leaves a command confirm alone when a turn errors', async () => {
    const { harness, prompts } = build()

    prompts.handleNotification('confirm.request', { default: true, prompt: 'p', request_id: 'c1' })
    prompts.onTurnError(SESSION)
    await settle()

    expect(calls(harness, 'confirm.respond')).toHaveLength(0)
    expect(prompts.blocked).toBe(true)
  })

  it('sends nothing when the socket is gone', async () => {
    const { harness, prompts } = build()

    prompts.handleNotification('approval.request', approval())
    prompts.onDisconnect()
    await settle()

    expect(calls(harness, 'approval.respond')).toHaveLength(0)
    expect(prompts.blocked).toBe(false)
    expect(harness.editorContainer.children[0]).toBe(harness.editor)
  })

  it('reports an RPC failure and restores the editor', async () => {
    const { harness, prompts, type } = build({ 'confirm.respond': new Error('socket gone') })

    prompts.handleNotification('confirm.request', { default: false, prompt: 'p', request_id: 'c1' })
    type('y')
    await settle()

    expect(harness.printed.some(line => line.text.includes('socket gone'))).toBe(true)
    expect(harness.editorContainer.children[0]).toBe(harness.editor)
  })

  it('raises the attention cue once, whatever is queued behind it', async () => {
    const harness = createHarness({ 'approval.respond': { ok: true }, 'confirm.respond': { ok: true } })
    const host = new SelectorHost({
      editor: harness.editor,
      editorContainer: harness.editorContainer,
      onChange: () => {},
      session: () => ({ epoch: harness.session.epoch, id: harness.session.id }),
      tui: harness.tui
    })
    const cues: boolean[] = []
    const prompts = new PromptCoordinator({
      clock: harness.clock,
      gateway: harness.gateway,
      host,
      keybindings: harness.deps.keybindings,
      onAttention: waiting => cues.push(waiting),
      sessionId: () => harness.session.id,
      theme: harness.deps.theme,
      transcript: harness.deps.transcript,
      tui: harness.tui
    })

    prompts.handleNotification('approval.request', approval())
    prompts.handleNotification('confirm.request', { default: false, prompt: 'p', request_id: 'c1' })

    // One cue for the pair: the title says "something is waiting", not how many.
    expect(cues).toEqual([true])

    prompts.onDisconnect()
    await settle()

    expect(cues).toEqual([true, false])
  })

  it('never shows more seconds than are actually left', () => {
    const { harness, prompts, screen } = build()

    // The broker computes expires_at from a fractional time.time(), so the
    // window measured here is a hair over thirty seconds.
    prompts.handleNotification('approval.request', approval({ expires_at: (harness.clock.nowMs() + 30_001) / 1000 }))

    expect(screen()).toContain('(30s)')
    expect(screen()).not.toContain('(31s)')
  })

  it('counts down to zero without ever going negative', () => {
    const { harness, prompts, screen } = build()

    prompts.handleNotification('approval.request', approval())
    harness.clock.advance(29_500)

    expect(screen()).toContain('(0s)')
  })
})
