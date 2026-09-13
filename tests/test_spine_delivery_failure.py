"""What the hub does when an outlet fails: the stream is retired, the worker
lives, the rest of the queue still goes out, and the per-turn fence reports the
failure instead of leaving the caller on a barrier that never lifts.

F2: a raising ``send_stream_chunk`` used to kill the outlet worker, leaving its
queue unconsumed and ``wait_idle``'s join() waiting forever — so a turn could be
saved with no ``message.complete`` and the UI wedged.
"""

import asyncio

from opendde_harness.spine.delivery import Capabilities, DeliveryHub
from opendde_harness.spine.events import StreamDelta, Text
from opendde_harness.spine.message import ChatType, Source

SRC = Source(channel="chat", chat_id="c1", sender_id="u", chat_type=ChatType.DM)


class Outlet:
    """Records what it was asked to send; fails the first N stream chunks."""

    name = "chat"
    capabilities = Capabilities(streaming=True)

    def __init__(self, fail_chunks: int = 0, hang: bool = False) -> None:
        self.log: list[tuple] = []
        self._fail_chunks = fail_chunks
        self._hang = hang

    async def deliver(self, out) -> None:
        self.log.append(("deliver", type(out).__name__))

    async def send_stream_chunk(self, chat_id, stream_id, delta, *, done=False) -> None:
        if self._hang:
            await asyncio.Event().wait()
        if self._fail_chunks > 0:
            self._fail_chunks -= 1
            raise RuntimeError("socket gone")
        self.log.append(("chunk", delta, done))


def _hub(outlet: Outlet, **kwargs) -> DeliveryHub:
    hub = DeliveryHub(send_max_retries=0, **kwargs)
    hub.register(outlet)
    return hub


async def test_a_failing_stream_delta_keeps_the_worker_alive_and_still_delivers_the_rest():
    outlet = Outlet(fail_chunks=1)
    hub = _hub(outlet, fence_timeout=1.0)

    await hub.dispatch(StreamDelta(delta="par", source=SRC, conversation_id="conv"))
    await hub.dispatch(Text(content="whole", source=SRC, conversation_id="conv"))

    report = await asyncio.wait_for(hub.fence("chat", "conv"), 2.0)

    assert outlet.log == [("deliver", "Text")]  # the Text still went out
    assert report.delivered == 1
    assert report.failed == 1
    assert report.ok is False
    assert report.timed_out is False
    assert not hub._workers[("chat", "conv")].done()  # worker alive, queue consumed
    await hub.aclose()


async def test_a_retired_streams_later_deltas_are_dropped_rather_than_resent():
    outlet = Outlet(fail_chunks=1)
    hub = _hub(outlet, fence_timeout=1.0)

    await hub.dispatch(StreamDelta(delta="par", source=SRC, conversation_id="conv"))
    await hub.dispatch(StreamDelta(delta="more", source=SRC, conversation_id="conv"))
    report = await asyncio.wait_for(hub.fence("chat", "conv"), 2.0)

    # Nothing was re-sent: an edit-in-place outlet would have duplicated text.
    assert outlet.log == []
    assert report.failed == 2

    # Closing the stream retires the retirement, so the next turn streams again.
    await hub.close_stream("conv")
    await hub.dispatch(StreamDelta(delta="next turn", source=SRC, conversation_id="conv"))
    await asyncio.wait_for(hub.fence("chat", "conv"), 2.0)
    assert outlet.log == [("chunk", "next turn", False)]
    await hub.aclose()


async def test_the_fence_reports_a_timeout_instead_of_waiting_on_a_hung_outlet():
    outlet = Outlet(hang=True)
    hub = _hub(outlet, fence_timeout=0.05)

    await hub.dispatch(StreamDelta(delta="par", source=SRC, conversation_id="conv"))
    report = await asyncio.wait_for(hub.fence("chat", "conv"), 2.0)

    assert report.timed_out is True
    assert report.ok is False
    await hub.aclose()


async def test_wait_idle_returns_even_though_the_stream_failed():
    outlet = Outlet(fail_chunks=1)
    hub = _hub(outlet, fence_timeout=1.0)

    await hub.dispatch(StreamDelta(delta="par", source=SRC, conversation_id="conv"))
    await hub.dispatch(Text(content="whole", source=SRC, conversation_id="conv"))

    await asyncio.wait_for(hub.wait_idle("chat"), 2.0)
    await hub.aclose()


async def test_a_cancelled_worker_accounts_for_what_it_never_delivered():
    outlet = Outlet(hang=True)
    hub = _hub(outlet, fence_timeout=1.0)

    await hub.dispatch(StreamDelta(delta="par", source=SRC, conversation_id="conv"))
    await hub.dispatch(Text(content="whole", source=SRC, conversation_id="conv"))
    await asyncio.sleep(0)  # let the worker pick up the first item and hang
    await hub.aclose()

    report = await asyncio.wait_for(hub.fence("chat", "conv"), 2.0)
    assert report.delivered == 0
    assert report.failed == 2  # the hung send and the event behind it
    assert report.timed_out is False
