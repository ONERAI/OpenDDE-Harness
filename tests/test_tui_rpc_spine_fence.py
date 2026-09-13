"""One outlet queue per conversation, and a bounded per-turn fence in front of
``message.complete``.

F8: the TUI runs every conversation over one channel, so a single per-channel
queue meant a subscription whose reader had stalled held every other
conversation's tokens behind it — and the sink's channel-wide render barrier made
one chat's completion wait on another chat's backlog.
"""

import asyncio

from opendde_harness.spine import (
    ChatType,
    Origin,
    Source,
    StreamDelta,
    TurnOutcome,
    TurnRequest,
    Usage,
)
from opendde_harness.spine.delivery import Capabilities, DeliveryHub
from opendde_harness.tui_rpc.spine import build_tui

_USAGE = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def _src(chat: str) -> Source:
    return Source(channel="tui", chat_id=chat, sender_id="u", chat_type=ChatType.DM)


class StallingOutlet:
    """Streams for every conversation but one, which it never finishes."""

    name = "tui"
    capabilities = Capabilities(streaming=True)

    def __init__(self, stalled: str) -> None:
        self._stalled = stalled
        self.gate = asyncio.Event()
        self.seen: list[tuple[str, str]] = []

    async def deliver(self, out) -> None:
        self.seen.append((out.conversation_id, type(out).__name__))

    async def send_stream_chunk(self, chat_id, stream_id, delta, *, done=False) -> None:
        if stream_id == self._stalled:
            await self.gate.wait()
            return
        self.seen.append((stream_id, delta))


async def test_a_stalled_conversation_does_not_hold_another_conversations_events():
    outlet = StallingOutlet("a")
    hub = DeliveryHub(fence_timeout=0.1)
    hub.register(outlet)

    await hub.dispatch(StreamDelta(delta="x", source=_src("a"), conversation_id="a"))
    await hub.dispatch(StreamDelta(delta="y", source=_src("b"), conversation_id="b"))

    flowing = await asyncio.wait_for(hub.fence("tui", "b"), 2.0)
    assert flowing.ok is True
    assert outlet.seen == [("b", "y")]

    # The stall is confined to the conversation that caused it.
    stalled = await asyncio.wait_for(hub.fence("tui", "a"), 2.0)
    assert stalled.timed_out is True

    outlet.gate.set()
    await hub.aclose()


class StallingEmitter:
    """A SubscriptionEmitter stand-in: one session's subscription never drains."""

    def __init__(self, stalled: str) -> None:
        self._stalled = stalled
        self.gate = asyncio.Event()
        self.events: list[tuple[str, str]] = []

    async def emit(self, session_key: str, event: dict) -> None:
        if session_key == self._stalled and event["type"] == "token.delta":
            await self.gate.wait()
        self.events.append((session_key, event["type"]))


class StreamingLoop:
    """An AgentLoop stand-in: every turn streams one delta and ends."""

    async def run_turn(self, req, emit, drain, *, stream=True):
        await emit(StreamDelta(delta=f"hi {req.text}"))
        return TurnOutcome(usage=_USAGE, explicit_reply=True, usage_detail={})


def _request(conversation: str) -> TurnRequest:
    return TurnRequest(origin=Origin.USER, source=_src(conversation), text=conversation, conversation=conversation)


async def test_one_chats_stalled_subscription_does_not_hold_another_chats_completion():
    emitter = StallingEmitter("a")
    # Two user slots: the pool, not the queue, is what serialises two chats when
    # it is sized at one, and this test is about the queue.
    scheduler, _hub, turn_ids, teardown = build_tui(StreamingLoop(), emitter, user_pool=2)
    turn_ids["a"] = "turn-a"
    turn_ids["b"] = "turn-b"

    stalled = scheduler.submit(_request("a"))
    flowing = scheduler.submit(_request("b"))

    await asyncio.wait_for(flowing.result(), 2.0)
    assert ("b", "token.delta") in emitter.events
    assert ("b", "message.complete") in emitter.events
    assert ("a", "message.complete") not in emitter.events

    emitter.gate.set()
    await asyncio.wait_for(stalled.result(), 2.0)
    assert ("a", "message.complete") in emitter.events

    await teardown()
