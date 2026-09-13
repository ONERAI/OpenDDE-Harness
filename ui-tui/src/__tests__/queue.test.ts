import { visibleWidth } from '@earendil-works/pi-tui'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { MessageQueue, normalizeBusyInputMode } from '../components/queue.js'
import { Theme } from '../theme.js'

describe('normalizeBusyInputMode', () => {
  it('accepts the two modes, case and padding aside', () => {
    expect(normalizeBusyInputMode('interrupt')).toBe('interrupt')
    expect(normalizeBusyInputMode(' QUEUE ')).toBe('queue')
  })

  it('falls back to queue for anything else', () => {
    expect(normalizeBusyInputMode('drop')).toBe('queue')
    expect(normalizeBusyInputMode(undefined)).toBe('queue')
    expect(normalizeBusyInputMode(7)).toBe('queue')
  })
})

describe('MessageQueue', () => {
  const theme = new Theme('dark', 0)

  let interrupt: ReturnType<typeof vi.fn<() => void>>
  let send: ReturnType<typeof vi.fn<(text: string) => void>>
  let turnActive: boolean
  let blocked: boolean
  let clock: number

  function build(mode: 'interrupt' | 'queue' = 'queue') {
    return new MessageQueue({
      actions: { interrupt, send },
      isDispatchBlocked: () => blocked,
      isTurnActive: () => turnActive,
      mode,
      now: () => clock,
      theme
    })
  }

  beforeEach(() => {
    interrupt = vi.fn<() => void>()
    send = vi.fn<(text: string) => void>()
    turnActive = false
    blocked = false
    clock = 1000
  })

  it('sends straight away while idle', () => {
    expect(build().submit('hello')).toBe('sent')
    expect(send).toHaveBeenCalledWith('hello')
  })

  it('holds a message typed during a turn and drains it when the turn ends', () => {
    turnActive = true

    const queue = build()

    expect(queue.submit('later')).toBe('queued')
    expect(send).not.toHaveBeenCalled()
    expect(queue.list()).toEqual(['later'])

    turnActive = false
    expect(queue.drain()).toBe(true)
    expect(send).toHaveBeenCalledWith('later')
    expect(queue.length).toBe(0)
  })

  it('cancels the running turn in interrupt mode and still queues the text', () => {
    turnActive = true

    const queue = build('interrupt')

    expect(queue.submit('stop and do this')).toBe('interrupted')
    expect(interrupt).toHaveBeenCalledTimes(1)
    expect(send).not.toHaveBeenCalled()
    expect(queue.list()).toEqual(['stop and do this'])
  })

  it('follows a mode change made at runtime', () => {
    turnActive = true

    const queue = build()

    queue.setMode('interrupt')
    queue.submit('now')

    expect(interrupt).toHaveBeenCalledTimes(1)
  })

  it('drains in order, one turn at a time', () => {
    turnActive = true

    const queue = build()

    queue.submit('one')
    queue.submit('two')

    turnActive = false
    queue.drain()
    expect(send).toHaveBeenLastCalledWith('one')

    queue.drain()
    expect(send).toHaveBeenLastCalledWith('two')
    expect(send).toHaveBeenCalledTimes(2)
  })

  it('does not drain while a turn is still running', () => {
    turnActive = true

    const queue = build()

    queue.submit('waiting')

    expect(queue.drain()).toBe(false)
    expect(send).not.toHaveBeenCalled()
  })

  it('sends the head on demand once the turn is over', () => {
    turnActive = true

    const queue = build()

    queue.submit('head')
    queue.submit('tail')

    turnActive = false
    expect(queue.submitHead()).toBe('sent')
    expect(send).toHaveBeenCalledWith('head')
    expect(queue.list()).toEqual(['tail'])

    queue.clear()
    expect(queue.submitHead()).toBe('empty')
  })

  it('refuses Ctrl+K while a turn is running, and keeps the queue', () => {
    turnActive = true

    const queue = build()

    queue.submit('head')
    queue.submit('tail')

    // Shifting the head and only then discovering that nothing could be sent
    // is how a queued prompt used to disappear.
    expect(queue.submitHead()).toBe('blocked')
    expect(send).not.toHaveBeenCalled()
    expect(queue.list()).toEqual(['head', 'tail'])

    turnActive = false
    expect(queue.submitHead()).toBe('sent')
    expect(send).toHaveBeenCalledWith('head')
  })

  it('interrupts on a second Enter over an empty editor', () => {
    turnActive = true

    const queue = build()

    expect(queue.submit('')).toBe('ignored')
    expect(interrupt).not.toHaveBeenCalled()

    clock += 100
    expect(queue.submit('')).toBe('interrupted')
    expect(interrupt).toHaveBeenCalledTimes(1)
  })

  it('does not interrupt when the two Enters are far apart', () => {
    turnActive = true

    const queue = build()

    queue.submit('')
    clock += 5000
    expect(queue.submit('')).toBe('ignored')
    expect(interrupt).not.toHaveBeenCalled()
  })

  it('does not interrupt when a real prompt came between the Enters', () => {
    turnActive = true

    const queue = build()

    queue.submit('')
    queue.submit('a real prompt')
    clock += 100
    expect(queue.submit('')).toBe('ignored')
    expect(interrupt).not.toHaveBeenCalled()
  })

  it('ignores Enter on an empty editor while idle', () => {
    const queue = build()

    expect(queue.submit('')).toBe('ignored')
    expect(queue.submit('   ')).toBe('ignored')
    expect(interrupt).not.toHaveBeenCalled()
    expect(send).not.toHaveBeenCalled()
  })

  it('renders the pending list with a hint, and nothing when empty', () => {
    turnActive = true

    const queue = new MessageQueue({
      actions: { interrupt, send },
      isTurnActive: () => turnActive,
      now: () => clock,
      submitHint: 'ctrl+k',
      theme
    })

    expect(queue.view.render(40)).toEqual([])

    queue.submit('a queued line')

    const lines = queue.view.render(40)

    expect(lines.some(line => line.includes('Queued: a queued line'))).toBe(true)
    expect(lines.some(line => line.includes('ctrl+k to send the first now'))).toBe(true)
    expect(lines.every(line => visibleWidth(line) <= 40)).toBe(true)
  })
})

describe('MessageQueue while the editor slot is taken', () => {
  const theme = new Theme('dark', 0)

  let interrupt: ReturnType<typeof vi.fn<() => void>>
  let send: ReturnType<typeof vi.fn<(text: string) => void>>
  let blocked: boolean
  let turnActive: boolean

  function build(mode: 'interrupt' | 'queue' = 'queue') {
    return new MessageQueue({
      actions: { interrupt, send },
      isDispatchBlocked: () => blocked,
      isTurnActive: () => turnActive,
      mode,
      theme
    })
  }

  beforeEach(() => {
    interrupt = vi.fn<() => void>()
    send = vi.fn<(text: string) => void>()
    blocked = true
    turnActive = false
  })

  it('queues instead of sending, even while idle', () => {
    const queue = build()

    expect(queue.submit('hello')).toBe('queued')
    expect(send).not.toHaveBeenCalled()
    expect(queue.list()).toEqual(['hello'])
  })

  it('does not interrupt the turn for a message it cannot send', () => {
    turnActive = true

    const queue = build('interrupt')

    expect(queue.submit('hello')).toBe('queued')
    expect(interrupt).not.toHaveBeenCalled()
  })

  it('leaves the head where it is on drain and on Ctrl+K', () => {
    const queue = build()

    queue.submit('first')
    queue.submit('second')

    expect(queue.drain()).toBe(false)
    expect(queue.submitHead()).toBe('blocked')
    expect(queue.list()).toEqual(['first', 'second'])
  })

  it('sends in order once the slot is free again', () => {
    const queue = build()

    queue.submit('first')
    queue.submit('second')
    blocked = false

    expect(queue.drain()).toBe(true)
    expect(send).toHaveBeenCalledWith('first')
    expect(queue.list()).toEqual(['second'])
  })

  it('ignores Enter on an empty editor rather than interrupting', () => {
    turnActive = true

    const queue = build()

    expect(queue.submit('')).toBe('ignored')
    expect(queue.submit('')).toBe('ignored')
    expect(interrupt).not.toHaveBeenCalled()
  })
})
