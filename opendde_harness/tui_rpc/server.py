"""asyncio JSON-RPC 2.0 server loop over a full-duplex socket (or, for tests, a
POSIX pipe pair).

Topology: the parent listens on a TCP-loopback socket; the Node child connects,
sends an auth token line, then exchanges newline-JSON frames over the same
connection. ``RpcServer`` is given the accepted socket *object* and wires it via
``loop.connect_accepted_socket`` (cross-platform: selector + proactor loops).
A legacy pipe path (``request_fd``/``notify_fd`` + ``connect_read/write_pipe``)
is retained for the ``--check`` smoke and unit tests.

`RpcServer` owns the read pump (one line-delimited JSON frame per iteration),
dispatches concurrently via `asyncio.create_task` so a long-running streaming
subscription doesn't block other RPC calls, and serializes writes with an
`asyncio.Lock` so concurrent dispatch tasks can't interleave bytes on the wire.

Inbound frame size limit: 1 MiB (specs §2.5). A frame over the cap closes
the connection. Nothing caps what the server writes: a frame goes out whole,
however large it is.

Writes await the transport's drain, so a slow reader backpressures the
subscription emitter's bounded queue instead of piling up unboundedly in the
socket's write buffer. That wait is bounded by ``DRAIN_TIMEOUT_S``: a peer
that has not read a byte for that long is gone, and a write side stalled
behind it would otherwise hold every RPC behind one lock forever, so the
connection is closed and the server stops.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from opendde_harness.tui_rpc.dispatcher import Dispatcher


# Per specs/tui-ipc.md §2.5
MAX_FRAME_BYTES = 1 * 1024 * 1024  # 1 MiB

# How long one write may wait for the peer to make room before the connection
# is declared dead.
#
# Every write holds ``_write_lock`` while it drains, so a peer that stops
# reading holds every other RPC behind it. Thirty seconds is far longer than
# any TUI render takes and far shorter than forever: past it the peer is gone,
# and the honest answer is to close rather than to queue for a reader that
# will never come back.
DRAIN_TIMEOUT_S = 30.0


class RpcServer:
    """Read JSON-RPC frames from `request_fd`, write responses to `notify_fd`.

    Args:
        request_fd: POSIX fd opened for reading (the Node→Python pipe).
        notify_fd:  POSIX fd opened for writing (the Python→Node pipe).
        dispatcher: a `Dispatcher` instance with all handlers registered.

    The server takes ownership of the FDs: they are closed on `stop()`.
    """

    def __init__(
        self,
        request_fd: int = -1,
        notify_fd: int = -1,
        dispatcher: "Dispatcher | None" = None,
        *,
        sock: "socket.socket | None" = None,
        auth_token: str | None = None,
    ) -> None:
        self._request_fd = request_fd
        self._notify_fd = notify_fd
        self._dispatcher = dispatcher
        # Cross-platform production transport: a single connected TCP-loopback
        # socket passed as an object (no os.dup of a socket fd, which is not
        # supported on Windows). When set it takes precedence over the fd pair.
        self._sock = sock
        # Optional shared-secret line the peer MUST send first. TCP loopback is
        # reachable by any local process (unlike an AF_UNIX file guarded by
        # 0600 perms), so the token restores the "only our spawned child can
        # talk to us" trust boundary. None disables the check (pipe/test paths).
        self._auth_token = auth_token

        self._reader: asyncio.StreamReader | None = None
        self._write_transport: asyncio.WriteTransport | None = None
        self._write_protocol: asyncio.BaseProtocol | None = None
        # Wraps the write transport so ``send_frame`` can await real socket
        # backpressure. Both paths below build one.
        self._writer: asyncio.StreamWriter | None = None
        self._write_lock = asyncio.Lock()
        self._pending: set[asyncio.Task] = set()
        self._stopped = asyncio.Event()
        self._started = asyncio.Event()

    @property
    def started(self) -> asyncio.Event:
        """Set once the read pump has attached to the FD; useful for tests."""
        return self._started

    # ----- write side -------------------------------------------------------

    async def send_frame(self, frame: dict) -> None:
        """Serialize and write a single JSON frame + newline to the peer.

        All writes (responses + notifications) MUST go through this method so
        the lock serializes them. The frame goes out whole, however large.
        """
        if self._stopped.is_set():
            # The connection is gone or going -- a peer that stopped reading,
            # or an ordinary shutdown. Nothing to write to, and nothing to
            # wait the drain timeout for a second time.
            return
        if self._write_transport is None:
            raise RuntimeError("RpcServer.send_frame called before serve_forever()")
        data = (json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            self._write_transport.write(data)
            # Connects the emitter's bounded queue to the socket's write
            # buffer: without this the queue bounds a staging area only.
            if self._writer is not None:
                try:
                    await asyncio.wait_for(self._writer.drain(), DRAIN_TIMEOUT_S)
                except TimeoutError:
                    logger.error(
                        "tui_rpc: peer stopped reading for {}s; closing the connection",
                        DRAIN_TIMEOUT_S,
                    )
                    self._abandon_stalled_peer()

    def _abandon_stalled_peer(self) -> None:
        """Give up on a peer that has stopped reading, and stop the server.

        A drain that never completes holds ``_write_lock`` forever, so every
        other RPC queues behind one dead subscription. Closing the transport
        and feeding the reader EOF ends ``serve_forever``'s read loop, which
        runs the ordinary shutdown: in-flight dispatch tasks cancelled, FDs
        closed. ``_stopped`` also makes every later ``send_frame`` a no-op
        rather than another wait of the same length.
        """
        self._stopped.set()
        if self._write_transport is not None:
            try:
                self._write_transport.close()
            except Exception:
                logger.exception("tui_rpc: error closing a stalled write transport")
        if self._reader is not None:
            # Unblocks the ``readuntil`` in ``serve_forever``, whose ``finally``
            # is what actually shuts the server down.
            self._reader.feed_eof()

    # ----- main loop --------------------------------------------------------

    async def serve_forever(self) -> None:
        """Run the read/dispatch/write pump until EOF or `stop()`."""
        loop = asyncio.get_running_loop()

        reader = asyncio.StreamReader(limit=MAX_FRAME_BYTES)
        reader_protocol = asyncio.StreamReaderProtocol(reader)

        # P0 fix (2026-05-15): the production transport in
        # ``tui_commands.run_subprocess_with_rpc`` dups the same accepted unix
        # socket fd into ``request_fd`` and ``notify_fd``. CPython's
        # ``connect_write_pipe`` builds a ``_UnixWritePipeTransport`` whose
        # ``__init__`` registers a reader callback on the WRITE fd to detect
        # "peer closed the pipe" (read-end EOF) — this works for real pipes
        # but is fatal for a bidirectional SOCK_STREAM socket: any inbound
        # byte (e.g. ``system.hello``) makes the reader callback fire,
        # ``_close()`` runs, and every subsequent ``send_frame`` becomes a
        # silent no-op. After 5 such drops asyncio also logs ``"pipe closed
        # by peer or os.write(pipe, data) raised exception."`` — which is the
        # exact symptom seen during ``/cha`` slash autocomplete.
        #
        # Use ``connect_accepted_socket`` (full-duplex selector transport) on
        # the same fd instead. The transport's ``.write()`` works without
        # the spurious peer-close detection. We keep the pipe-based path as a
        # fallback for tests/CI that wire bare ``os.pipe()`` pairs.
        req_is_sock = notif_is_sock = False
        if self._sock is None:
            try:
                req_is_sock = stat.S_ISSOCK(os.fstat(self._request_fd).st_mode)
                notif_is_sock = stat.S_ISSOCK(os.fstat(self._notify_fd).st_mode)
            except OSError:
                req_is_sock = notif_is_sock = False

        if self._sock is not None:
            # Cross-platform production path: full-duplex transport over a single
            # connected socket object (TCP loopback). connect_accepted_socket is
            # implemented on both the selector (POSIX) and proactor (Windows)
            # event loops, and passing the socket object avoids os.dup of a
            # socket fd -- which is unsupported on Windows.
            self._sock.setblocking(False)
            transport, _ = await loop.connect_accepted_socket(lambda: reader_protocol, self._sock)
            self._write_transport = transport
            self._write_protocol = reader_protocol
            self._writer = asyncio.StreamWriter(transport, reader_protocol, reader, loop)
        elif req_is_sock and notif_is_sock:
            # Both fds are dups of the same accepted socket. Close the read
            # dup and reclaim the write dup as a ``socket.socket`` — only one
            # handle is needed for a full-duplex transport.
            try:
                os.close(self._request_fd)
            except OSError:
                pass
            sock = socket.socket(fileno=self._notify_fd)
            sock.setblocking(False)
            transport, _ = await loop.connect_accepted_socket(lambda: reader_protocol, sock)
            self._write_transport = transport
            self._write_protocol = reader_protocol
            self._writer = asyncio.StreamWriter(transport, reader_protocol, reader, loop)
        else:
            # Legacy / test path: bare pipes via ``os.pipe()``.
            # `os.fdopen` so the transport owns a Python file object; loop
            # will close the underlying fd when the transport closes.
            await loop.connect_read_pipe(lambda: reader_protocol, os.fdopen(self._request_fd, "rb", buffering=0))
            # ``FlowControlMixin`` rather than ``BaseProtocol``: it is what
            # gives the pipe transport a ``_drain_helper`` for ``send_frame``
            # to await, so this path backpressures like the socket ones.
            write_transport, write_protocol = await loop.connect_write_pipe(
                lambda: asyncio.streams.FlowControlMixin(loop=loop),
                os.fdopen(self._notify_fd, "wb", buffering=0),
            )
            self._write_transport = write_transport
            self._write_protocol = write_protocol
            self._writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)

        self._reader = reader

        self._started.set()
        logger.info(
            "tui_rpc: RpcServer started (pid={}, request_fd={}, notify_fd={}, mode={})",
            os.getpid(),
            self._request_fd,
            self._notify_fd,
            "socket-obj" if self._sock is not None else ("socket" if req_is_sock else "pipe"),
        )

        # Trust-boundary gate for the TCP-loopback transport: the peer must send
        # the shared secret as the very first newline-terminated line before any
        # JSON-RPC frame. Only our spawned Node child knows it (passed via env),
        # so a rogue local process that connects to the port is rejected here
        # before any dispatch. Disabled (None) for the pipe/test paths.
        if self._auth_token is not None:
            try:
                first = await asyncio.wait_for(reader.readuntil(b"\n"), timeout=10.0)
            except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                logger.error("tui_rpc: auth token not received; closing connection")
                self._stopped.set()
                return
            if first.rstrip(b"\n") != self._auth_token.encode("utf-8"):
                logger.error("tui_rpc: auth token mismatch; closing connection")
                self._stopped.set()
                return

        try:
            while not self._stopped.is_set():
                try:
                    line = await reader.readuntil(b"\n")
                except asyncio.IncompleteReadError as exc:
                    # EOF — peer closed. Drain whatever partial bytes we have.
                    if exc.partial:
                        logger.warning(
                            "tui_rpc: incomplete final frame ({} bytes); dropping",
                            len(exc.partial),
                        )
                    break
                except asyncio.LimitOverrunError:
                    logger.error(
                        "tui_rpc: frame exceeds {} bytes; closing connection",
                        MAX_FRAME_BYTES,
                    )
                    break

                if len(line) > MAX_FRAME_BYTES:
                    logger.error("tui_rpc: frame {} bytes > {} cap; closing", len(line), MAX_FRAME_BYTES)
                    break

                # Spawn the dispatch as an independent task so streaming /
                # slow handlers don't block subsequent reads.
                task = asyncio.create_task(self._handle_frame(line))
                self._pending.add(task)
                task.add_done_callback(self._pending.discard)
        finally:
            await self._shutdown()

    async def _handle_frame(self, raw: bytes) -> None:
        try:
            try:
                frame = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                # JSON-RPC §-32700 parse_error response (id unknown → null).
                resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "parse_error",
                        "data": {"reason": str(exc)},
                    },
                }
                await self.send_frame(resp)
                return

            response = await self._dispatcher.dispatch(frame)

            # If the original frame omitted `id` (a notification per JSON-RPC
            # 2.0), suppress the response — but dispatcher already echoed
            # whatever it received as id, so we only suppress when id was
            # explicitly absent in the inbound frame.
            if isinstance(frame, dict) and "id" not in frame:
                return
            await self.send_frame(response)
        except Exception:
            # Last-resort guard so a single buggy handler can't kill the pump.
            logger.exception("tui_rpc: _handle_frame failed")

    async def _shutdown(self) -> None:
        # Cancel any in-flight dispatch tasks.
        for task in list(self._pending):
            if not task.done():
                task.cancel()
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
        self._pending.clear()

        if self._write_transport is not None:
            try:
                self._write_transport.close()
            except Exception:
                logger.exception("tui_rpc: error closing write transport")
            self._write_transport = None
        self._writer = None

        self._stopped.set()
        logger.info("tui_rpc: RpcServer stopped (pid={})", os.getpid())

    async def stop(self) -> None:
        """Signal the read loop to exit and wait for cleanup."""
        self._stopped.set()
        # We can't easily interrupt `readuntil`, but closing the write side
        # plus setting `_stopped` will cause the next iteration after EOF to
        # bail. Caller typically just cancels the serve_forever task.


__all__ = ["RpcServer", "MAX_FRAME_BYTES", "DRAIN_TIMEOUT_S"]
