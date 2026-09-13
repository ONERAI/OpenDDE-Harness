// Test doubles: a terminal that records bytes instead of owning a tty, and a
// transport that answers RPC calls from a script while recording the order.

import type { Terminal } from '@earendil-works/pi-tui'

import type { RpcTransport } from '../gateway.js'

/** A `Terminal` that writes into an array. No stdin, no raw mode, no tty. */
export class FakeTerminal implements Terminal {
  readonly writes: string[] = []

  onInput?: (data: string) => void

  private readonly columnCount: number
  private readonly rowCount: number

  constructor(columns = 80, rows = 24) {
    this.columnCount = columns
    this.rowCount = rows
  }

  get columns(): number {
    return this.columnCount
  }

  get rows(): number {
    return this.rowCount
  }

  get kittyProtocolActive(): boolean {
    return false
  }

  /** Total bytes the TUI has written so far. */
  get byteCount(): number {
    return this.writes.reduce((sum, chunk) => sum + Buffer.byteLength(chunk, 'utf8'), 0)
  }

  /** Everything written, as one string. */
  output(): string {
    return this.writes.join('')
  }

  start(onInput: (data: string) => void, _onResize: () => void): void {
    this.onInput = onInput
  }

  stop(): void {
    this.onInput = undefined
  }

  async drainInput(): Promise<void> {}

  write(data: string): void {
    this.writes.push(data)
  }

  moveBy(): void {}
  hideCursor(): void {}
  showCursor(): void {}
  clearLine(): void {}
  clearFromCursor(): void {}
  clearScreen(): void {}
  /** Window titles the app asked for, newest last. */
  readonly titles: string[] = []

  setTitle(title: string): void {
    this.titles.push(title)
  }
  setProgress(): void {}
}

export interface RecordedCall {
  method: string
  params: unknown
}

/** An `RpcTransport` that answers from a table of canned results and records
 *  every call in order. Unlisted methods resolve to `{}`.
 *
 *  A canned `Error` is thrown; a canned function is called with the params and
 *  its result awaited, which is how a method that only answers later — a
 *  sign-in the test has to drive first — is scripted. */
export class FakeTransport implements RpcTransport {
  readonly calls: RecordedCall[] = []

  /** Handler for the events of the last `turn.subscribe`. */
  emit: (event: unknown) => void = () => {}

  constructor(readonly results: Record<string, unknown> = {}) {}

  /** The methods that were called, in order. */
  get methods(): string[] {
    return this.calls.map(call => call.method)
  }

  async rpc<R = unknown, P = unknown>(method: string, params: P): Promise<R> {
    this.calls.push({ method, params })

    const result = this.results[method]

    if (result instanceof Error) {
      throw result
    }

    if (typeof result === 'function') {
      return (await (result as (params: unknown) => unknown)(params)) as R
    }

    return (result ?? {}) as R
  }

  async subscribe<E = unknown, P = unknown>(method: string, params: P, handler: (event: E) => void) {
    this.calls.push({ method, params })
    this.emit = event => handler(event as E)

    const id = `sub-${this.calls.length}`

    return {
      subscription_id: id,
      // Recorded like any other call, so a test can see where the detach fell
      // in a session switch and whether there was exactly one of them.
      unsubscribe: async () => {
        this.calls.push({ method: `${method.split('.')[0]}.unsubscribe`, params: { subscription_id: id } })
      }
    }
  }
}
