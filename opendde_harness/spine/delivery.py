"""Delivery: what a channel can do (Capabilities), the streaming opt-in
(SupportsStreaming), the per-channel send surface (Outlet), and the hub that
routes each deliverable to its outlet (DeliveryHub).

The hub keeps one bounded queue and one serial worker per (channel,
conversation): a deliverable is routed by its source channel and conversation,
and the queue is the backpressure point — a full or stalled queue blocks only
that conversation's sender, never another conversation's or another channel's
(no head-of-line blocking), while per-conversation order is held by its single
worker. This mirrors the lane model on the delivery side.

Transports import the vocabulary defined here; the spine never imports
a transport.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from loguru import logger

from opendde_harness.spine.events import (
    Deliverable,
    StreamDelta,
    TurnEnded,
    TurnEvent,
    TurnFailed,
    TurnRetry,
    TurnStarted,
)


@dataclass(frozen=True)
class Capabilities:
    """What a channel can do, declared explicitly (not inferred from methods).

    Only capabilities with a real consumer live here. ``media``/``reactions``
    are adapter-internal today (nothing routes on them) — add them back with
    their consumer when one exists.
    """

    interactive_login: bool = False  # QR / scan login (weixin, whatsapp); read by CLI `channel login`
    streaming: bool = False  # SupportsStreaming slot; activated in B


@runtime_checkable
class SupportsStreaming(Protocol):
    """Opt-in incremental delivery (edit-in-place). Inert until the agent loop
    is wired to produce stream chunks (scope B)."""

    async def send_stream_chunk(self, chat_id: str, stream_id: str, delta: str, *, done: bool = False) -> None: ...


@runtime_checkable
class Outlet(Protocol):
    """A channel's send surface. ``deliver`` either renders the deliverable or, if
    the channel can't express it, eats it with a normal return (logging its own
    skip). Only a real failure — transport error, bug — raises, which the hub
    retries. Eating is not failure. Lifecycle (connect/teardown) stays on the
    channel; an outlet is just the send seam."""

    name: str
    capabilities: Capabilities

    async def deliver(self, out: Deliverable) -> None: ...


_SEND_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubles each retry (1, 2, 4)
_OUTLET_QUEUE_MAXSIZE = 100  # per-outlet backpressure bound; config knob lands with its consumer
_SEND_TIMEOUT_S = 30.0  # bound on one outlet send; a hung transport must not own the queue
_FENCE_TIMEOUT_S = 30.0  # bound on the render barrier; a wedged outlet must not wedge the caller

#: Queue key: one queue and worker per (channel, conversation). A deliverable
#: with no conversation (``post`` of a standalone menu) rides the channel's
#: shared "" queue.
_Key = tuple[str, str]


@dataclass(frozen=True)
class _StreamClose:
    """A marker the hub puts on a conversation's queue so the stream's done=True
    chunk is sent after the last StreamDelta still in flight (a sourceless
    lifecycle event can't ride the queue itself, but this can)."""

    conversation_id: str


@dataclass
class _Counters:
    """What one queue has done since the last fence read."""

    delivered: int = 0
    failed: int = 0


@dataclass(frozen=True)
class DeliveryReport:
    """The outcome of one turn's delivery, as the fence saw it.

    ``failed`` counts events the outlet never rendered: a send that raised past
    its retries, a delta dropped because its stream was retired, anything still
    queued when the fence's bound expired. ``timed_out`` says the bound expired
    rather than the queue draining.
    """

    delivered: int
    failed: int
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.failed == 0 and not self.timed_out


class DeliveryHub:
    """Routes each deliverable into its (channel, conversation) bounded queue,
    where a serial worker delivers it (retrying a raising send with backoff).
    Holds the outlet registry plus a queue, worker and counters per key; no turn
    state beyond the open/retired stream tables.

    One queue per conversation, not per channel: the TUI runs every conversation
    over a single channel, and a subscription whose reader stalled would
    otherwise hold every other conversation's tokens behind it. Same-conversation
    order is still held by that conversation's single worker.

    Streaming rides the same queue: StreamDelta is sent via send_stream_chunk and
    close_stream enqueues a marker so the closing chunk follows the deltas. The
    open-stream table is the worker's alone (single owner); the channel a stream
    rides is recorded synchronously on enqueue so close_stream can route to it.

    A failing send never kills a worker. A stream whose send raised is *retired*:
    its later deltas are dropped rather than re-sent, because the outlet edits
    one message in place and a blind retry of a partially transmitted delta would
    duplicate text. Everything else in that queue still goes out, and every item
    is accounted for, so the fence below always resolves."""

    def __init__(
        self,
        send_max_retries: int = _SEND_MAX_RETRIES,
        *,
        send_timeout: float = _SEND_TIMEOUT_S,
        fence_timeout: float = _FENCE_TIMEOUT_S,
    ) -> None:
        self._send_max_retries = send_max_retries
        self._send_timeout = send_timeout
        self._fence_timeout = fence_timeout
        self._outlets: dict[str, Outlet] = {}
        self._queues: dict[_Key, asyncio.Queue[Deliverable | _StreamClose]] = {}
        self._workers: dict[_Key, asyncio.Task[None]] = {}
        self._counters: dict[_Key, _Counters] = {}
        # conversation_id -> channel, written on enqueue (sink path), read by
        # close_stream to route its marker; the open-stream table below is the
        # worker's (conversation_id -> chat_id, present iff the stream is open).
        self._stream_channel: dict[str, str] = {}
        self._open_streams: dict[str, str] = {}
        # Streams whose outlet raised: no further delta is sent until they close.
        self._retired_streams: set[str] = set()

    def register(self, outlet: Outlet) -> None:
        # Register-once, at startup: a running worker captures its outlet when it
        # starts, so re-registering a different outlet for a live channel does not
        # hot-swap it.
        self._outlets[outlet.name] = outlet

    async def dispatch(self, out: Deliverable) -> None:
        await self._enqueue(out)

    async def post(self, out: Deliverable) -> None:
        """Send a deliverable that did not come from a turn (e.g. a Sentinel menu).
        Routes like dispatch; the caller stamps source.channel. Returns once the
        event is queued, not once delivered — delivery is the outlet worker's, and
        a full queue backpressures this channel's caller."""
        await self._enqueue(out)

    async def close_stream(self, conversation_id: str) -> None:
        """End a conversation's stream. Routes a close marker through the
        conversation's queue so its done=True chunk follows the last StreamDelta
        still in flight. Driven by a lifecycle event (TurnEnded / TurnFailed); a
        conversation with no open stream is a no-op."""
        channel = self._stream_channel.pop(conversation_id, None)
        if channel is None:
            return
        key = (channel, conversation_id)
        queue = self._queues.get(key)
        if queue is not None:
            self._ensure_worker(key)  # a marker nobody consumes would hang this turn's fence
            await queue.put(_StreamClose(conversation_id))

    async def _enqueue(self, out: Deliverable) -> None:
        if out.source is None:
            raise ValueError(f"cannot route a {type(out).__name__} with no source")
        channel = out.source.channel
        if channel not in self._outlets:
            logger.warning("no outlet for channel {!r}; dropping {}", channel, type(out).__name__)
            return
        if isinstance(out, StreamDelta):
            # Remember the channel this stream rides so a later close_stream (driven
            # by a sourceless lifecycle event) can route its marker here.
            self._stream_channel.setdefault(out.conversation_id, channel)
        elif isinstance(out, TurnRetry) and out.discard:
            # An edit-in-place outlet cannot un-send what it showed; the closed
            # stream stays as a finished message and the re-run opens a new one.
            await self.close_stream(out.conversation_id)
        key = (channel, out.conversation_id or "")
        queue = self._queue_for(key)
        await queue.put(out)  # full queue blocks only this conversation

    def _queue_for(self, key: _Key) -> asyncio.Queue:
        queue = self._queues.get(key)
        if queue is None:
            queue = asyncio.Queue(maxsize=_OUTLET_QUEUE_MAXSIZE)
            self._queues[key] = queue
            self._counters[key] = _Counters()
        self._ensure_worker(key)
        return queue

    def _ensure_worker(self, key: _Key) -> None:
        worker = self._workers.get(key)
        if worker is None or worker.done():
            # Restart on done() too, not just absence: the worker is resident
            # (blocks on get), so a dead one would leave its queue unconsumed and
            # silently deadlock this conversation's senders. Mirrors the lane worker.
            self._workers[key] = asyncio.create_task(self._run_outlet(key))

    async def _run_outlet(self, key: _Key) -> None:
        queue = self._queues[key]
        outlet = self._outlets[key[0]]
        counters = self._counters[key]
        try:
            while True:
                item = await queue.get()
                delivered = False
                try:
                    delivered = await self._handle(outlet, item)
                except asyncio.CancelledError:
                    logger.warning(
                        "delivery cancelled mid-send: channel={!r} event={}",
                        outlet.name,
                        type(item).__name__,
                    )
                    raise
                except Exception as exc:
                    # One event's failure, not the worker's: log it, account for
                    # it, keep consuming. A dead worker would leave everything
                    # behind it undelivered and every fence on this key hanging.
                    if isinstance(item, (StreamDelta, _StreamClose)):
                        self._retired_streams.add(item.conversation_id)
                    logger.error(
                        "delivery failed: channel={!r} event={} reason={}",
                        outlet.name,
                        type(item).__name__,
                        exc,
                    )
                finally:
                    # Always mark done and always account for the item — eaten,
                    # failed, retries exhausted, cancelled mid-send — so a fence's
                    # join() reflects every dequeued item and never hangs, and its
                    # report never calls an undelivered event delivered.
                    queue.task_done()
                    if delivered:
                        counters.delivered += 1
                    else:
                        counters.failed += 1
        finally:
            # Leaving for any reason (cancelled by aclose, or an exception past
            # every guard) means nothing will consume what is left: account for it
            # here so a fence resolves instead of waiting on a queue with no reader.
            self._fail_remaining(queue, counters)

    async def _handle(self, outlet: Outlet, item: Deliverable | _StreamClose) -> bool:
        """Deliver one dequeued item; True if the outlet took it.

        False is a delivery that did not happen and is not going to be retried;
        raising is a delivery that failed here and now. Both count as failures at
        the fence — an outlet that *eats* an event it cannot express returns
        normally and counts as delivered, per the Outlet contract."""
        if isinstance(item, _StreamClose):
            await self._close_stream_chunk(outlet, item.conversation_id)
            return True
        if isinstance(item, StreamDelta):
            if item.conversation_id in self._retired_streams:
                # This stream's outlet already raised. The delta cannot be placed
                # in a message that was never finished, and re-sending it risks
                # duplicating text an edit-in-place outlet already showed.
                return False
            await self._stream_chunk(outlet, item)
            return True
        return await self._deliver_with_retry(outlet, item)

    def _fail_remaining(self, queue: asyncio.Queue, counters: _Counters) -> None:
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()
            counters.failed += 1

    async def _stream_chunk(self, outlet: Outlet, ev: StreamDelta) -> None:
        # A non-streaming outlet eats the delta (the full text reaches it another
        # way); only an outlet that both can and declares streaming gets chunks.
        if not (isinstance(outlet, SupportsStreaming) and outlet.capabilities.streaming):
            return
        chat_id = ev.source.chat_id
        self._open_streams.setdefault(ev.conversation_id, chat_id)  # first delta opens the stream
        await asyncio.wait_for(
            outlet.send_stream_chunk(chat_id, ev.conversation_id, ev.delta, done=False),
            self._send_timeout,
        )

    async def _close_stream_chunk(self, outlet: Outlet, conversation_id: str) -> None:
        retired = conversation_id in self._retired_streams
        self._retired_streams.discard(conversation_id)  # the next turn opens a fresh stream
        chat_id = self._open_streams.pop(conversation_id, None)
        if chat_id is None or retired:
            return  # no open stream (empty turn, non-streaming outlet), or a retired one
        if isinstance(outlet, SupportsStreaming) and outlet.capabilities.streaming:
            await asyncio.wait_for(
                outlet.send_stream_chunk(chat_id, conversation_id, "", done=True),
                self._send_timeout,
            )

    async def _deliver_with_retry(self, outlet: Outlet, out: Deliverable) -> bool:
        delay = _RETRY_BASE_DELAY
        for attempt in range(self._send_max_retries + 1):
            try:
                await asyncio.wait_for(outlet.deliver(out), self._send_timeout)
                return True
            except Exception as exc:
                if attempt == self._send_max_retries:
                    logger.error(
                        "delivery failed after {} retries: channel={!r} event={} reason={}",
                        self._send_max_retries,
                        outlet.name,
                        type(out).__name__,
                        exc,
                    )
                    return False
                await asyncio.sleep(delay)
                delay *= 2
        return False

    def drain(self) -> int:
        """Drop every not-yet-delivered (still-queued) event and return the count.
        Synchronous (no await) so it is atomic against the live workers. This only
        drops queued events; a best-effort flush within a shutdown window is not yet
        implemented."""
        dropped = 0
        for key, queue in self._queues.items():
            counters = self._counters[key]
            before = counters.failed
            self._fail_remaining(queue, counters)
            dropped += counters.failed - before
        if dropped:
            logger.warning("delivery hub drained {} undelivered events on shutdown", dropped)
        return dropped

    async def fence(
        self,
        channel: str,
        conversation_id: str | None = None,
        *,
        timeout: float | None = None,
    ) -> DeliveryReport:
        """The per-turn delivery fence: block until everything this conversation
        queued has been delivered or explicitly failed, then report which.

        Bounded, so a wedged outlet cannot wedge the caller — a timeout returns a
        report with ``timed_out`` set rather than hanging. Reading the fence
        resets the counters, so each turn's report covers that turn alone. With no
        ``conversation_id`` it fences the whole channel."""
        keys = [k for k in self._queues if k[0] == channel and (conversation_id is None or k[1] == conversation_id)]
        if not keys:
            return DeliveryReport(delivered=0, failed=0, timed_out=False)
        joins = [asyncio.create_task(self._queues[k].join()) for k in keys]
        _done, pending = await asyncio.wait(joins, timeout=self._fence_timeout if timeout is None else timeout)
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning(
                "delivery fence timed out: channel={!r} conversation={!r} queues_still_draining={}",
                channel,
                conversation_id,
                len(pending),
            )
        delivered = failed = 0
        for key in keys:
            counters = self._counters[key]
            delivered += counters.delivered
            failed += counters.failed
            counters.delivered = counters.failed = 0
        return DeliveryReport(delivered=delivered, failed=failed, timed_out=bool(pending))

    async def wait_idle(self, channel: str, conversation_id: str | None = None) -> None:
        """Block until this conversation's outlet has delivered everything queued —
        the render barrier a caller awaits after a turn's result() before it treats
        the output as on-screen (result() means 'no more events', not 'delivered').
        A conversation with nothing ever queued is already idle. Callers that need
        to know *whether* it rendered use ``fence``, which this wraps."""
        await self.fence(channel, conversation_id)

    async def aclose(self) -> None:
        """Cancel every outlet worker. Abrupt: in-flight delivery (mid-retry) is
        cancelled, not finished. Finishing the current send within a window is not yet
        implemented."""
        for worker in self._workers.values():
            worker.cancel()
        await asyncio.gather(*self._workers.values(), return_exceptions=True)
        self._workers.clear()


def make_hub_sink(hub: DeliveryHub) -> Callable[[TurnEvent], Awaitable[None]]:
    """Adapt the hub into a scheduler EventSink: deliverables route through the
    hub; lifecycle events carry no source, so they are dropped here and never
    reach the deliverable-only enqueue path (lifecycle -> taps lands later). The
    REPL and the gateway share this sink; the TUI keeps its own (it fires
    message.complete / error after the render barrier)."""

    async def sink(event: TurnEvent) -> None:
        if isinstance(event, (TurnStarted, TurnFailed, TurnEnded)):
            return
        await hub.dispatch(event)

    return sink
