"""The turn's edges: what is durable before what, and what reaches the outlet once.

Four orderings that nothing inside the loop can check. A streamed reply must not
be repeated as a closing Text. A no-model delivery is recorded before it is
emitted. The commit happens before the optional work that follows it -- and what
happens to a committed turn whose hand-off to extraction is cut off is stated here
rather than assumed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore
from opendde_harness.memory_engine.outbox import MemoryOutbox
from opendde_harness.providers.base import LLMProvider, LLMResponse, StreamDelta
from opendde_harness.spine import ChatType, Origin, Source, TurnRequest
from opendde_harness.spine.events import StreamDelta as StreamEvent
from opendde_harness.spine.events import Text

MODEL = "primary/model"
KEY = "cli:t"
USAGE = {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}


class Answering(LLMProvider):
    """Streams one reply and stops."""

    def __init__(self, reply: str = "the answer") -> None:
        super().__init__()
        self.reply = reply

    def get_default_model(self) -> str:
        return MODEL

    async def chat(self, messages, tools=None, model=None, **kwargs):
        return LLMResponse(content=self.reply, finish_reason="stop", usage=dict(USAGE))

    async def chat_stream(self, messages, tools=None, model=None, **kwargs):
        yield StreamDelta(content=self.reply)
        yield StreamDelta(content=None, finish_reason="stop", usage=dict(USAGE))


class Backend:
    """Indexes nothing; records what it was offered."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def recall(self, query, *, user_id=None, agent_id=None, top_k):
        return []

    async def store(self, session_id: str, messages: list[dict[str, Any]], *, metadata=None) -> bool:
        self.calls.append(session_id)
        return True

    async def feedback(self, signals: dict[str, Any]) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _loop(tmp_path, backend: Any = None) -> AgentLoop:
    return AgentLoop(Answering(), tmp_path, AgentLoopSettings(model=MODEL), backend=backend)


def _request(text: str = "hello", *, deliver: str | None = None) -> TurnRequest:
    channel, _, chat = KEY.partition(":")
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel=channel, chat_id=chat, sender_id="user", chat_type=ChatType.DM),
        text=text,
        conversation=KEY,
        deliver_text=deliver,
        media=(),
    )


def _no_injections() -> list[Any]:
    return []


async def test_a_streamed_reply_is_not_repeated_as_a_closing_text(tmp_path) -> None:
    """The reply travels one way. Streaming it and then sending it whole would show
    a TUI the answer twice, which is what the single return-to-emit boundary
    exists to prevent."""
    loop = _loop(tmp_path)
    seen: list[Any] = []

    async def emit(event: Any) -> None:
        seen.append(event)

    outcome = await loop.run_turn(_request(), emit, _no_injections, stream=True)

    assert [type(e).__name__ for e in seen if isinstance(e, (StreamEvent, Text))] == ["StreamDelta"]
    assert outcome.text == "the answer", "the outcome still carries an observation copy"


async def test_a_non_streaming_outlet_gets_the_reply_as_one_text(tmp_path) -> None:
    """The other half of the same rule: nothing streamed, so the Text is the reply."""
    loop = _loop(tmp_path)
    seen: list[Any] = []

    async def emit(event: Any) -> None:
        seen.append(event)

    await loop.run_turn(_request(), emit, _no_injections, stream=False)

    texts = [e for e in seen if isinstance(e, Text)]
    assert [t.content for t in texts] == ["the answer"]
    assert not [e for e in seen if isinstance(e, StreamEvent)]


async def test_a_no_model_delivery_is_on_disk_before_it_is_emitted(tmp_path) -> None:
    """A forwarded report is one message and no turn, and it is still saved first.

    A save that fails after the emit leaves the user a delivered message no turn
    recorded, which is the one ordering this path has to keep.
    """
    loop = _loop(tmp_path)
    order: list[str] = []
    real_save = loop.sessions.save

    def save(session):
        order.append("saved")
        return real_save(session)

    loop.sessions.save = save  # type: ignore[method-assign]

    async def emit(event: Any) -> None:
        order.append(f"emitted {type(event).__name__}")

    outcome = await loop.run_turn(_request(deliver="the report"), emit, _no_injections, stream=True)

    assert order == ["saved", "emitted Text"]
    assert outcome.explicit_reply is True
    recorded = loop.sessions.get_or_create(KEY).messages
    assert len(recorded) == 1 and "the report" in str(recorded[0])


async def test_the_turn_is_committed_before_the_work_that_follows_it(tmp_path) -> None:
    """Compaction, extraction and feedback all run after ``turn.completed``.

    Each of them is optional and one of them is a second network call; a turn the
    user has already read must not be lost to any of them.
    """
    loop = _loop(tmp_path, backend=Backend())
    order: list[str] = []
    original = loop._maybe_compact_remote

    async def watched(session, outcome, *, session_key=""):
        order.append("after-turn work")
        return await original(session, outcome, session_key=session_key)

    loop._maybe_compact_remote = watched  # type: ignore[method-assign]

    async def emit(_event: Any) -> None:
        return None

    await loop.run_turn(_request(), emit, _no_injections, stream=True)

    records = loop.sessions.get_or_create(KEY).messages
    assert order == ["after-turn work"]
    assert any(m.get("role") == "assistant" for m in records), "the answer is on disk"


async def test_a_turn_cut_off_before_the_hand_off_keeps_the_conversation(tmp_path) -> None:
    """Committed, then cancelled before the extraction queue was written.

    What survives is the conversation: the journal closed first, so the session log
    holds the turn. What does not survive is the hand-off -- the queue is empty and
    nothing rescans the log for a committed turn that never reached it, so this
    turn will not be indexed. Stated here as the current behaviour, because the
    alternative (a scan on startup) is a decision, not a detail.
    """
    backend = Backend()
    loop = _loop(tmp_path, backend=backend)

    async def cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError

    loop._maybe_compact_remote = cancelled  # type: ignore[method-assign]

    async def emit(_event: Any) -> None:
        return None

    with pytest.raises(asyncio.CancelledError):
        await loop.run_turn(_request(), emit, _no_injections, stream=True)

    reloaded = AgentLoop(Answering(), tmp_path, AgentLoopSettings(model=MODEL), backend=Backend())
    records = reloaded.sessions.get_or_create(KEY).messages
    assert any(m.get("role") == "assistant" for m in records), "the answer is on disk"
    assert MemoryOutbox(tmp_path, MemoryStore(tmp_path)).pending() == [], "and nothing was queued for extraction"
