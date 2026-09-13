"""The write side of ``RpcServer``, and the delta coalescer in front of it.

Nothing caps what the server writes: a frame goes out whole, however large it
is. What is bounded is the wait for the peer to take it. Draining under the
write lock is what makes the subscription queue a real bound rather than a
staging area in front of one, and it is also what can hold every RPC behind
one dead subscription, so both the overflow close and the write itself are
bounded.

The read side keeps its 1 MiB limit: a peer that sends a larger frame closes
the connection.
"""

import asyncio
import json
import os
import socket

import pytest
from loguru import logger

from opendde_harness.tui_rpc import server as server_module
from opendde_harness.tui_rpc import subscriptions
from opendde_harness.tui_rpc.dispatcher import Dispatcher
from opendde_harness.tui_rpc.server import MAX_FRAME_BYTES, RpcServer
from opendde_harness.tui_rpc.subscriptions import SubscriptionEmitter, _merge_consecutive_token_deltas


class _Pipes:
    """A server on one end of two pipes, and the peer's ends of them."""

    def __init__(self, dispatcher: Dispatcher) -> None:
        to_server_r, self.to_server_w = os.pipe()
        self.from_server_r, from_server_w = os.pipe()
        self.server = RpcServer(request_fd=to_server_r, notify_fd=from_server_w, dispatcher=dispatcher)
        self.task: asyncio.Task | None = None
        self._buffer = b""

    async def start(self) -> None:
        self.task = asyncio.create_task(self.server.serve_forever())
        await asyncio.wait_for(self.server.started.wait(), 2)

    def send(self, frame: dict) -> None:
        os.write(self.to_server_w, (json.dumps(frame) + "\n").encode("utf-8"))

    async def read_raw(self, timeout: float = 5.0) -> bytes:
        """One newline-delimited frame's bytes, the newline not included.

        Read in blocks and buffered across calls rather than a byte at a time,
        which would be a syscall each.
        """
        loop = asyncio.get_running_loop()

        def blocking_read() -> bytes:
            while b"\n" not in self._buffer:
                chunk = os.read(self.from_server_r, 65536)
                if not chunk:
                    break
                self._buffer += chunk
            line, _, rest = self._buffer.partition(b"\n")
            self._buffer = rest
            return line

        return await asyncio.wait_for(loop.run_in_executor(None, blocking_read), timeout)

    async def read_frame(self, timeout: float = 5.0) -> dict:
        """One frame off the peer's end of the pipe, decoded."""
        return json.loads((await self.read_raw(timeout)).decode("utf-8"))

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        for fd in (self.to_server_w, self.from_server_r):
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.fixture
async def pipes():
    made: list[_Pipes] = []

    def build(dispatcher: Dispatcher) -> _Pipes:
        harness = _Pipes(dispatcher)
        made.append(harness)
        return harness

    yield build
    for harness in made:
        await harness.close()


async def test_send_frame_awaits_the_transports_drain(pipes):
    """Backpressure: the emitter's bounded queue bounds the socket, not a
    staging area in front of it."""
    dispatcher = Dispatcher()
    harness = pipes(dispatcher)
    await harness.start()

    drains = 0
    protocol = harness.server._write_protocol
    original = protocol._drain_helper

    async def counted() -> None:
        nonlocal drains
        drains += 1
        await original()

    protocol._drain_helper = counted

    await harness.server.send_frame({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})

    assert drains == 1
    assert (await harness.read_frame())["id"] == 1


async def test_the_socket_path_enforces_the_inbound_cap_and_drains_too():
    """The production transport, not just the pipe pair the other tests use.

    ``connect_accepted_socket`` is what ``ddeharness tui`` runs on, and its
    ``StreamWriter`` is built over the reader protocol rather than a protocol
    of its own. This holds that path to the two rules it still has: a write
    awaits the drain, and a frame arriving over the cap closes the connection.
    """
    dispatcher = Dispatcher()

    async def small(params: dict) -> dict:
        return {"ok": True}

    dispatcher.register("test.small", small)

    ours, theirs = socket.socketpair()
    server = RpcServer(sock=ours, dispatcher=dispatcher)
    task = asyncio.create_task(server.serve_forever())
    try:
        await asyncio.wait_for(server.started.wait(), 2)

        drains = 0
        original = server._write_protocol._drain_helper

        async def counted() -> None:
            nonlocal drains
            drains += 1
            await original()

        server._write_protocol._drain_helper = counted

        theirs.sendall(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "test.small", "params": {}}).encode() + b"\n")

        loop = asyncio.get_running_loop()
        theirs.setblocking(True)
        theirs.settimeout(5)
        raw = await loop.run_in_executor(None, lambda: theirs.recv(65536))
        response = json.loads(raw.split(b"\n")[0].decode("utf-8"))

        assert response == {"jsonrpc": "2.0", "id": 3, "result": {"ok": True}}
        assert drains == 1

        # Inbound is where the cap still lives: a frame over it ends the
        # connection rather than being buffered whole.
        oversized = (
            b'{"jsonrpc":"2.0","id":4,"method":"test.small","params":{"blob":"' + b"x" * MAX_FRAME_BYTES + b'"}}\n'
        )

        def push() -> None:
            try:
                theirs.sendall(oversized)
            except OSError:
                # The server closed mid-send, which is the point.
                pass

        await loop.run_in_executor(None, push)
        await asyncio.wait_for(task, 5)

        assert server._stopped.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        theirs.close()


# ---------------------------------------------------------------------------
# The coalescer
# ---------------------------------------------------------------------------


def test_consecutive_deltas_merge_into_one_event():
    """The point of the 16ms window: one frame per run, not one per token.

    A run is per delta type, so a thinking delta never joins a token one.
    """
    batch = [
        {"type": "token.delta", "payload": {"text": "a"}},
        {"type": "token.delta", "payload": {"text": "b"}},
        {"type": "thinking.delta", "payload": {"text": "c"}},
        {"type": "thinking.delta", "payload": {"text": "d"}},
    ]

    assert _merge_consecutive_token_deltas(batch) == [
        {"type": "token.delta", "payload": {"text": "ab"}},
        {"type": "thinking.delta", "payload": {"text": "cd"}},
    ]


def test_a_non_delta_event_still_breaks_the_run_in_order():
    batch = [
        {"type": "token.delta", "payload": {"text": "one"}},
        {"type": "tool.start", "payload": {"name": "read"}},
        {"type": "token.delta", "payload": {"text": "two"}},
    ]

    merged = _merge_consecutive_token_deltas(batch)

    assert [event["type"] for event in merged] == ["token.delta", "tool.start", "token.delta"]
    assert merged[0]["payload"]["text"] == "one"
    assert merged[2]["payload"]["text"] == "two"


# ---------------------------------------------------------------------------
# A peer that has stopped reading
# ---------------------------------------------------------------------------


def _stall_drain(server: RpcServer) -> asyncio.Event:
    """Make every drain on `server` block until the returned event is set."""
    released = asyncio.Event()

    async def blocked() -> None:
        await released.wait()

    server._write_protocol._drain_helper = blocked
    return released


async def test_a_stalled_writer_still_lets_an_overflowing_subscription_close(pipes, monkeypatch):
    """The deadlock: the close needed the lock the stall was holding.

    ``_close_overflow`` used to send the overflow notification *before*
    retiring the subscription. That write queued behind the write lock the
    subscription's own coalescer was holding while its drain hung, so nothing
    closed and the producer stayed inside ``emit`` forever.
    """
    monkeypatch.setattr(subscriptions, "QUEUE_CAPACITY", 2)
    monkeypatch.setattr(subscriptions, "QUEUE_PUT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(subscriptions, "OVERFLOW_NOTIFY_TIMEOUT_S", 0.05)

    harness = pipes(Dispatcher())
    await harness.start()
    released = _stall_drain(harness.server)

    emitter = SubscriptionEmitter(harness.server.send_frame)
    sub_id = await emitter.register("tui:s")
    sub = emitter._by_id[sub_id]

    try:
        # The first event reaches the coalescer, which blocks in the write;
        # the rest fill the queue and then find it full.
        for index in range(20):
            await asyncio.wait_for(emitter.emit("tui:s", {"type": "token.delta", "payload": {"text": str(index)}}), 3)
            if sub.closed:
                break

        assert sub.closed, "a persistently stalled subscription must retire itself"
        assert sub.coalesce_task.done() or sub.coalesce_task.cancelled()
        assert sub_id not in emitter._by_id
        # The lock is free again, so other RPC writes are not stuck behind it.
        assert not harness.server._write_lock.locked()
    finally:
        released.set()


async def test_a_drain_that_never_completes_closes_the_connection(pipes, monkeypatch):
    """A write side stalled forever must not hold every RPC behind one lock."""
    monkeypatch.setattr(server_module, "DRAIN_TIMEOUT_S", 0.05)

    harness = pipes(Dispatcher())
    await harness.start()
    released = _stall_drain(harness.server)

    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message), level="ERROR")
    try:
        await asyncio.wait_for(harness.server.send_frame({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}), 3)

        assert any("peer stopped reading" in line for line in logged)
        assert harness.server._stopped.is_set()
        # The read pump reached its shutdown rather than waiting on a peer
        # that is not there.
        await asyncio.wait_for(harness.task, 3)
        # And a later frame is dropped rather than waiting the same again.
        await asyncio.wait_for(harness.server.send_frame({"jsonrpc": "2.0", "id": 2, "result": {}}), 1)
    finally:
        logger.remove(sink)
        released.set()
