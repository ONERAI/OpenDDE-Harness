// OpenDDE Harness TUI RPC — production JSON-RPC 2.0 client.
//
// Transport: TCP loopback (cross-platform — Windows has no usable AF_UNIX).
// The Python parent listens on 127.0.0.1:<ephemeral>; the Node child connects
// via `net.createConnection({host, port})`, sends OPENDDE_HARNESS_RPC_TOKEN as the first
// line (the parent validates it before any dispatch), then speaks JSON frames.
// A unix-socket path is still accepted for legacy setups. Bare FD inheritance
// (pass_fds=(3,4)) was rejected — Node can't reliably wrap inherited pipe FDs.
//
// Framing: newline-delimited UTF-8 JSON (specs §2.5). Each frame is
// `JSON.stringify(obj) + '\n'`. Single frame limit: 1 MiB of UTF-8 bytes,
// newline included — the same thing `tui_rpc/server.py` measures on the frames
// it reads, and not the same as a JavaScript string's `.length`.
//
// Writes are serialized through a `writeQueue` (single-writer model) so
// concurrent `rpc()` / `subscribe()` calls never interleave bytes.
//
// Error mapping: incoming JSON-RPC error frames are converted to typed
// `RpcError` subclasses via `errors.ts::rpcErrorFromFrame`.

import type { Socket } from 'node:net'

import { createConnection } from 'node:net'

import type { EventNotificationParams, JsonRpcErrorResponse, JsonRpcRequest, JsonRpcResponse } from './generated.js'

import { rpcErrorFromFrame } from './errors.js'
import { isJsonRpcError } from './generated.js'
import { SubscriptionRegistry } from './subscriptions.js'

const MAX_FRAME_BYTES = 1024 * 1024 // 1 MiB (specs §2.5)
const NEWLINE = 0x0a

type Pending = {
  /** Runs while the response frame is being processed, before the awaiting
   *  caller is resumed. `subscribe` uses it to install its handler in time for
   *  an event that shares the response's chunk. */
  onResult?: (result: unknown) => void
  resolve: (value: unknown) => void
  reject: (err: Error) => void
}

export interface RpcClientOptions {
  /** RPC target: "host:port" (TCP loopback) or a unix socket path. Defaults to env `OPENDDE_HARNESS_RPC_SOCKET`. */
  socketPath?: string
  /** Optional logger for non-fatal protocol oddities. Defaults to stderr. */
  warn?: (msg: string) => void
  /**
   * Sink for server-initiated notifications whose method is NOT `event`
   * (the subscription-stream envelope). The confirm round-trip
   * (`confirm.request`) arrives this way — a first-class top-level method,
   * not a per-subscription stream event. When omitted, such notifications
   * are logged as unknown and dropped.
   */
  onNotification?: (method: string, params: unknown) => void
  /**
   * Called once when the socket is gone — peer close, transport error, or
   * `close()` from this side. The UI uses it to drop broker prompts and
   * pickers that can no longer be answered.
   */
  onClose?: (err: Error) => void
}

export class RpcClient {
  private readonly socket: Socket
  private readonly registry = new SubscriptionRegistry()
  private readonly pending = new Map<number, Pending>()
  private readonly warn: (msg: string) => void
  private readonly onNotification?: (method: string, params: unknown) => void
  private readonly onClose?: (err: Error) => void

  private nextId = 1
  private readBuffer: Buffer = Buffer.alloc(0)
  private writeQueue: Promise<void> = Promise.resolve()
  private closed = false
  private connected = false
  private readonly connectPromise: Promise<void>

  constructor(opts: RpcClientOptions = {}) {
    const target = opts.socketPath ?? process.env.OPENDDE_HARNESS_RPC_SOCKET
    if (!target) {
      throw new Error(
        'RpcClient: no RPC target supplied; pass `socketPath` or set ' + 'OPENDDE_HARNESS_RPC_SOCKET env var.'
      )
    }
    this.warn = opts.warn ?? (m => process.stderr.write(`[rpc-client] ${m}\n`))
    this.onNotification = opts.onNotification
    this.onClose = opts.onClose

    // Cross-platform transport: the parent exports either a TCP-loopback
    // "host:port" (current Python parent; works on Windows too) or, for legacy
    // setups, a unix socket path. A trailing ":<digits>" disambiguates TCP.
    const tcp = /^(.+):(\d+)$/.exec(target)
    if (tcp) {
      this.socket = createConnection({ host: tcp[1], port: Number(tcp[2]) })
    } else {
      this.socket = createConnection(target)
    }
    // Shared secret the parent validates as the first line before any frame
    // (the loopback port is reachable by any local process, so this gates it).
    const authToken = process.env.OPENDDE_HARNESS_RPC_TOKEN

    this.connectPromise = new Promise<void>((resolve, reject) => {
      const onConnect = () => {
        this.connected = true
        this.socket.off('error', onError)
        // Must be the very first bytes on the wire, ahead of any RPC frame.
        if (authToken) {
          this.socket.write(authToken + '\n')
        }
        resolve()
      }
      const onError = (err: Error) => {
        this.socket.off('connect', onConnect)
        reject(err)
      }
      this.socket.once('connect', onConnect)
      this.socket.once('error', onError)
    })

    // Bytes, not decoded text: the frame limit is a byte limit, and a
    // multibyte character must not be split across the newline scan.
    this.socket.on('data', (chunk: string | Buffer) => {
      const bytes = typeof chunk === 'string' ? Buffer.from(chunk, 'utf-8') : chunk
      this.readBuffer = this.readBuffer.length === 0 ? bytes : Buffer.concat([this.readBuffer, bytes])
      if (this.readBuffer.length > MAX_FRAME_BYTES * 2) {
        // Defensive: if peer is flooding without newlines, abort rather than OOM.
        this.warn(
          `incoming read buffer exceeded ${MAX_FRAME_BYTES * 2} bytes without ` + 'newline — closing connection'
        )
        this.failAll(new Error('rpc-client: frame size limit exceeded'))
        this.socket.destroy()
        return
      }
      this.drainBuffer()
    })
    this.socket.on('end', () => this.failAll(new Error('socket closed by peer')))
    this.socket.on('error', err => this.failAll(err))
  }

  /** Awaitable handle that resolves once the socket connection is established. */
  ready(): Promise<void> {
    return this.connectPromise
  }

  private drainBuffer(): void {
    let nl = this.readBuffer.indexOf(NEWLINE)
    while (nl !== -1) {
      const line = this.readBuffer.subarray(0, nl)
      this.readBuffer = this.readBuffer.subarray(nl + 1)
      if (line.length > 0) {
        this.handleFrame(line)
      }
      nl = this.readBuffer.indexOf(NEWLINE)
    }
  }

  /** `raw` is one frame's bytes without its terminating newline. */
  private handleFrame(raw: Buffer): void {
    // The newline counts towards the limit, as it does on the Python side.
    if (raw.length + 1 > MAX_FRAME_BYTES) {
      this.warn(`oversized frame (${raw.length + 1} bytes) dropped`)
      return
    }
    const line = raw.toString('utf-8').trim()
    if (line.length === 0) {
      return
    }
    let frame: unknown
    try {
      frame = JSON.parse(line)
    } catch {
      this.warn(`malformed frame ignored: ${line.slice(0, 120)}`)
      return
    }
    if (!frame || typeof frame !== 'object') {
      this.warn('non-object frame ignored')
      return
    }
    const obj = frame as Record<string, unknown>

    // Notification frame (no `id`, has `method`)
    if (obj.id === undefined && typeof obj.method === 'string') {
      if (obj.method === 'event') {
        const params = obj.params as EventNotificationParams<unknown> | undefined
        if (params && typeof params.subscription_id === 'string') {
          this.registry.dispatch(params)
        } else {
          this.warn('event notification missing subscription_id/event')
        }
      } else if (this.onNotification) {
        // First-class top-level notifications (e.g. confirm.request) are not
        // subscription-stream events; hand them to the consumer's sink.
        this.onNotification(obj.method, obj.params)
      } else {
        this.warn(`unknown notification method: ${obj.method}`)
      }
      return
    }

    // Response frame (has `id`)
    const resp = frame as JsonRpcResponse<unknown>
    const id = resp.id
    if (typeof id !== 'number' && typeof id !== 'string') {
      this.warn('response frame has no valid id')
      return
    }
    const idKey = typeof id === 'number' ? id : Number(id)
    const pending = this.pending.get(idKey)
    if (!pending) {
      this.warn(`response for unknown id ${String(id)}`)
      return
    }
    this.pending.delete(idKey)
    if (isJsonRpcError(resp)) {
      pending.reject(rpcErrorFromFrame((resp as JsonRpcErrorResponse).error))
    } else {
      // Before resolving: the rest of this chunk is dispatched synchronously,
      // so anything the result has to be registered for must happen now.
      pending.onResult?.(resp.result)
      pending.resolve(resp.result)
    }
  }

  private failAll(err: Error): void {
    if (this.closed) {
      return
    }
    this.closed = true
    for (const [, p] of this.pending) {
      p.reject(err)
    }
    this.pending.clear()
    this.registry.clear()
    this.onClose?.(err)
  }

  private async writeFrame(frame: string): Promise<void> {
    const size = Buffer.byteLength(frame, 'utf-8')
    if (size > MAX_FRAME_BYTES) {
      throw new Error(`rpc-client: outgoing frame ${size} bytes exceeds ${MAX_FRAME_BYTES} limit`)
    }
    // Serialize all writes — even when the socket itself is happy with
    // concurrent writes, we don't want two frames interleaved on the wire.
    const prev = this.writeQueue
    this.writeQueue = (async () => {
      await prev
      if (!this.connected) {
        await this.connectPromise
      }
      await new Promise<void>((resolve, reject) => {
        this.socket.write(frame, err => (err ? reject(err) : resolve()))
      })
    })()
    return this.writeQueue
  }

  /** Invoke a JSON-RPC method and await the typed result. */
  rpc<R = unknown, P = unknown>(method: string, params: P): Promise<R> {
    return this.request<R, P>(method, params)
  }

  /** `onResult` runs synchronously while the response frame is handled. */
  private async request<R, P>(method: string, params: P, onResult?: (result: R) => void): Promise<R> {
    if (this.closed) {
      throw new Error('rpc-client: closed')
    }
    const id = this.nextId++
    const req: JsonRpcRequest<P> = { jsonrpc: '2.0', id, method, params }
    const frame = JSON.stringify(req) + '\n'
    const result = new Promise<R>((resolve, reject) => {
      this.pending.set(id, {
        onResult: onResult && (value => onResult(value as R)),
        resolve: v => resolve(v as R),
        reject
      })
    })
    try {
      await this.writeFrame(frame)
    } catch (err) {
      this.pending.delete(id)
      throw err
    }
    return result
  }

  /**
   * Subscribe to a server-push stream (e.g. `turn.subscribe`).
   *
   * The server returns a `{subscription_id}` result; this method registers
   * the handler against that id and returns an `unsubscribe()` thunk that
   * both calls the paired server method (if `unsubscribeMethod` is given)
   * and detaches the handler locally.
   *
   * Registration happens while the response frame is processed, not after the
   * await resumes: the server may put the stream's first event in the same TCP
   * chunk as the response, and `drainBuffer` dispatches that chunk's lines
   * synchronously — a handler installed a microtask later would miss it.
   */
  async subscribe<E = unknown, P = unknown, R extends { subscription_id: string } = { subscription_id: string }>(
    method: string,
    params: P,
    handler: (event: E) => void,
    opts: { unsubscribeMethod?: string } = {}
  ): Promise<{ subscription_id: string; unsubscribe: () => Promise<void> }> {
    const result = await this.request<R, P>(method, params, ({ subscription_id }) =>
      this.registry.register<E>(subscription_id, handler)
    )
    const subscriptionId = result.subscription_id
    const unsubscribeMethod = opts.unsubscribeMethod
    const unsubscribe = async (): Promise<void> => {
      this.registry.unregister(subscriptionId)
      if (unsubscribeMethod && !this.closed) {
        await this.rpc<unknown, { subscription_id: string }>(unsubscribeMethod, {
          subscription_id: subscriptionId
        })
      }
    }
    return { subscription_id: subscriptionId, unsubscribe }
  }

  /** Number of pending requests (mainly for tests). */
  pendingCount(): number {
    return this.pending.size
  }

  /** Number of active subscriptions (mainly for tests). */
  subscriptionCount(): number {
    return this.registry.size()
  }

  /** Tear down the socket and reject every pending promise. */
  close(): void {
    this.failAll(new Error('rpc-client: closed by caller'))
    try {
      this.socket.end()
    } catch {
      /* noop */
    }
    try {
      this.socket.destroy()
    } catch {
      /* noop */
    }
  }
}
