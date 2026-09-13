// The transport's two edge cases, over real loopback sockets: the 1 MiB frame
// limit is a limit on UTF-8 bytes (a JavaScript string's `.length` counts
// UTF-16 units, so 400k Han characters look small and are not), and a
// subscription's first event can arrive in the same TCP chunk as the response
// that names its id.

import type { AddressInfo, Socket } from 'node:net'

import { createServer } from 'node:net'
import { afterEach, describe, expect, it } from 'vitest'

import { RpcClient } from '../rpc/client.js'

const MAX_FRAME_BYTES = 1024 * 1024

/** Bytes of `{"jsonrpc":"2.0","id":1,"method":"echo","params":{"text":""}}\n`,
 *  so a test can size a payload to land exactly on the limit. */
const ENVELOPE_BYTES =
  Buffer.byteLength(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'echo', params: { text: '' } }), 'utf-8') + 1

/** ASCII: one byte per character. */
function ascii(bytes: number): string {
  return 'x'.repeat(bytes)
}

/** Han characters: three UTF-8 bytes each, one JSON character each. */
function cjk(bytes: number): string {
  return '漢'.repeat(Math.floor(bytes / 3)) + ascii(bytes % 3)
}

interface Peer {
  close: () => Promise<void>
  port: number
  /** Push raw bytes at the client, exactly as written. */
  send: (text: string) => void
}

/** A server that answers every request with an empty result, unless the test
 *  takes over with `onRequest`. */
async function peer(onRequest?: (frame: { id: number; method: string }, write: (text: string) => void) => void) {
  const queued: string[] = []

  let socket: Socket | undefined

  const server = createServer(connection => {
    socket = connection

    for (const text of queued.splice(0)) {
      connection.write(text)
    }

    let buffer = ''

    connection.on('data', chunk => {
      buffer += chunk.toString('utf-8')

      let nl = buffer.indexOf('\n')

      while (nl !== -1) {
        const line = buffer.slice(0, nl)

        buffer = buffer.slice(nl + 1)
        nl = buffer.indexOf('\n')

        let frame: { id: number; method: string }

        try {
          // The first line is the auth token when the env sets one.
          frame = JSON.parse(line)
        } catch {
          continue
        }

        if (onRequest) {
          onRequest(frame, text => connection.write(text))
        } else {
          connection.write(`{"jsonrpc":"2.0","id":${frame.id},"result":{}}\n`)
        }
      }
    })
    connection.on('error', () => {})
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  const harness: Peer = {
    close: () =>
      new Promise<void>(resolve => {
        socket?.destroy()
        server.close(() => resolve())
      }),
    port: (server.address() as AddressInfo).port,
    send: text => {
      if (socket) {
        socket.write(text)
      } else {
        queued.push(text)
      }
    }
  }

  return harness
}

const open: { client: RpcClient; server: Peer }[] = []

/** A connected client, torn down after the test. */
async function connect(server: Peer, opts: { onNotification?: (method: string, params: unknown) => void } = {}) {
  const warnings: string[] = []
  const client = new RpcClient({
    socketPath: `127.0.0.1:${server.port}`,
    warn: message => warnings.push(message),
    ...opts
  })

  open.push({ client, server })
  await client.ready()

  return { client, warnings }
}

afterEach(async () => {
  for (const { client, server } of open.splice(0)) {
    client.close()
    await server.close()
  }
})

describe('frame limits count UTF-8 bytes', () => {
  it('sends a frame that lands just below and exactly on the limit', async () => {
    const server = await peer()
    const { client } = await connect(server)

    await expect(client.rpc('echo', { text: ascii(MAX_FRAME_BYTES - ENVELOPE_BYTES - 1) })).resolves.toEqual({})
    await expect(client.rpc('echo', { text: ascii(MAX_FRAME_BYTES - ENVELOPE_BYTES) })).resolves.toEqual({})
  })

  it('refuses an ASCII frame one byte over the limit', async () => {
    const server = await peer()
    const { client } = await connect(server)

    await expect(client.rpc('echo', { text: ascii(MAX_FRAME_BYTES - ENVELOPE_BYTES + 1) })).rejects.toThrow(
      /1048577 bytes exceeds 1048576/
    )
  })

  it('refuses a CJK frame whose string length is under the limit', async () => {
    const server = await peer()
    const { client } = await connect(server)

    // 400k characters — well under the limit as UTF-16 units, 1.2 MB as UTF-8.
    await expect(client.rpc('echo', { text: '漢'.repeat(400_000) })).rejects.toThrow(/1200\d{3} bytes exceeds/)
  })

  it('accepts an incoming frame that ends exactly on the limit', async () => {
    const server = await peer()
    const notifications: unknown[] = []
    const { client } = await connect(server, { onNotification: (_method, params) => notifications.push(params) })

    const envelope = Buffer.byteLength(JSON.stringify({ jsonrpc: '2.0', method: 'note', params: { text: '' } })) + 1

    server.send(
      JSON.stringify({ jsonrpc: '2.0', method: 'note', params: { text: ascii(MAX_FRAME_BYTES - envelope) } }) + '\n'
    )
    await expect(client.rpc('ping', {})).resolves.toEqual({})

    expect(notifications).toHaveLength(1)
  })

  it('drops an incoming CJK frame whose bytes exceed the limit', async () => {
    const server = await peer()
    const notifications: unknown[] = []
    const { client, warnings } = await connect(server, { onNotification: (_m, params) => notifications.push(params) })

    server.send(JSON.stringify({ jsonrpc: '2.0', method: 'note', params: { text: cjk(MAX_FRAME_BYTES + 3) } }) + '\n')

    // Round-trips after the oversized frame, so the drop is not just slow.
    await expect(client.rpc('ping', {})).resolves.toEqual({})

    expect(notifications).toEqual([])
    expect(warnings.join('\n')).toMatch(/oversized frame \(10\d{5} bytes\) dropped/)
  })
})

describe('subscribe', () => {
  it('keeps an event that arrives in the same chunk as the subscribe response', async () => {
    const events: unknown[] = []
    const server = await peer((frame, write) => {
      if (frame.method === 'turn.subscribe') {
        // One write: the response and the stream's first event coalesce into a
        // single chunk, which `drainBuffer` dispatches synchronously.
        write(
          `{"jsonrpc":"2.0","id":${frame.id},"result":{"subscription_id":"s1"}}\n` +
            '{"jsonrpc":"2.0","method":"event","params":{"subscription_id":"s1","event":{"type":"first"}}}\n'
        )

        return
      }

      write(`{"jsonrpc":"2.0","id":${frame.id},"result":{}}\n`)
    })
    const { client } = await connect(server)

    const subscription = await client.subscribe('turn.subscribe', {}, event => events.push(event))

    server.send('{"jsonrpc":"2.0","method":"event","params":{"subscription_id":"s1","event":{"type":"second"}}}\n')
    await expect(client.rpc('ping', {})).resolves.toEqual({})

    expect(subscription.subscription_id).toBe('s1')
    expect(events).toEqual([{ type: 'first' }, { type: 'second' }])
  })
})
