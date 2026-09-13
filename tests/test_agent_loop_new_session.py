"""``/new`` closes a conversation, preserves it, and does not wait to do it.

The plugin backend is the single owner of automatic durable extraction, so
``/new`` hands it the turns the closed session completed and returns. It used to
call the host's own summarizer first -- one unsized LLM call over the whole
unconsolidated tail -- and refuse to reset when that call failed, which put the
user's ability to start a new conversation behind a network round trip. Nothing
about the reset depends on extraction now, and nothing waits for it.
"""

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.providers.base import ErrorClassification, LLMProvider, LLMResponse
from tests import _messages as build

MODEL = "primary/model"
KEY = "tui:chat-1"

# What a caller is allowed to feel. The backend stub below takes 30s, so
# anything that awaits it lands two orders of magnitude above this.
_NEW_BOUND_S = 2.0


class _Provider(LLMProvider):
    """Answers nothing; ``/new`` must not reach a model at all."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls += 1
        return LLMResponse(content="unexpected", finish_reason="stop")

    async def chat_with_retry(self, **kwargs):
        self.calls += 1
        return LLMResponse(content="unexpected", finish_reason="stop")

    def classify_error(self, exc):
        return ErrorClassification("no_model")

    def get_default_model(self):
        return MODEL


class SlowBackend:
    """A wedged memory service: ``store`` never finishes inside a test's life."""

    def __init__(self, delay: float = 30.0) -> None:
        self.delay = delay
        self.calls: list[tuple[str, int]] = []
        self.entered = asyncio.Event()

    async def recall(self, query, *, user_id=None, agent_id=None, top_k):
        return []

    async def store(self, session_id: str, messages: list[dict[str, Any]], *, metadata=None) -> bool:
        self.calls.append((session_id, len(messages)))
        self.entered.set()
        await asyncio.sleep(self.delay)
        return True

    async def feedback(self, signals: dict[str, Any]) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _loop(tmp_path: Path, backend: Any = None) -> AgentLoop:
    return AgentLoop(
        _Provider(),
        tmp_path,
        AgentLoopSettings(model=MODEL),
        backend=backend,
    )


def _talk(loop: AgentLoop, key: str = KEY) -> Any:
    """Give the session two turns' worth of messages and persist them."""
    session = loop.sessions.get_or_create(key)
    session.record(build.user("run the assay"))
    session.record(build.assistant("done"))
    session.record(build.user("and again"))
    session.record(build.assistant("done again"))
    loop.sessions.save(session)
    return session


def _archives(loop: AgentLoop) -> list[Path]:
    return sorted((loop.sessions.sessions_dir / "_closed").rglob("*.jsonl"))


async def _quiesce(loop: AgentLoop) -> None:
    """Stop the background drain without letting a wedged backend hold the test."""
    await loop.drain_backend_stores(timeout=0.05)
    if loop._outbox_task is not None:
        loop._outbox_task.cancel()


async def test_new_preserves_the_closed_session_and_starts_an_empty_one(tmp_path):
    loop = _loop(tmp_path)
    session = _talk(loop)
    live_path = loop.sessions._get_session_path(KEY)
    said = live_path.read_text(encoding="utf-8")
    assert "run the assay" in said

    reply = await loop._handle_slash_command(session, "/new")

    assert reply == "New session started."
    archived = _archives(loop)
    assert len(archived) == 1, "the closed conversation is preserved, not cleared in place"
    assert archived[0].read_text(encoding="utf-8") == said

    fresh = loop.sessions.get_or_create(KEY)
    assert fresh.messages == []
    assert fresh is not session, "the cache no longer hands back the closed session"
    assert live_path.exists(), "and the key has a session file of its own again"


async def test_closing_the_same_key_twice_keeps_both_conversations(tmp_path):
    loop = _loop(tmp_path)
    await loop._handle_slash_command(_talk(loop), "/new")
    await loop._handle_slash_command(_talk(loop), "/new")

    assert len(_archives(loop)) == 2


async def test_new_does_not_wait_for_the_backend_to_index(tmp_path):
    """A slow or wedged memory service must not hold the user inside the
    conversation they asked to leave."""
    backend = SlowBackend()
    loop = _loop(tmp_path, backend)
    session = _talk(loop)

    started = time.monotonic()
    reply = await loop._handle_slash_command(session, "/new")
    elapsed = time.monotonic() - started

    assert reply == "New session started."
    assert elapsed < _NEW_BOUND_S, f"/new waited {elapsed:.1f}s on the backend"
    await asyncio.wait_for(backend.entered.wait(), timeout=_NEW_BOUND_S)
    assert backend.calls == [(KEY, 4)], "the closed session's turns were handed over all the same"
    assert loop.sessions.get_or_create(KEY).messages == []

    # The hand-off is queued, not abandoned: the outbox still owes it.
    assert [e.session for e in loop.outbox.pending()] == [KEY]
    await _quiesce(loop)


async def test_new_queues_the_hand_off_in_the_extraction_outbox(tmp_path):
    backend = SlowBackend()
    loop = _loop(tmp_path, backend)

    await loop._handle_slash_command(_talk(loop), "/new")

    queued = loop.outbox.pending()
    assert len(queued) == 1
    assert queued[0].session == KEY
    assert len(queued[0].messages) == 4
    assert queued[0].queued_at
    assert loop.outbox.path.exists(), "the queue is on disk before /new returns"

    await _quiesce(loop)


async def test_new_starts_the_next_generation_on_the_same_key(tmp_path):
    """The key survives the reset, so the generation is the only thing that tells
    a record written after it from one written before."""
    loop = _loop(tmp_path)
    _talk(loop)

    await loop._handle_slash_command(loop.sessions.get_or_create(KEY), "/new")

    assert loop.sessions.get_or_create(KEY).generation == 1
    loop.sessions.invalidate(KEY)
    assert loop.sessions.get_or_create(KEY).generation == 1, "and it survives a reload"


async def test_with_no_backend_nothing_is_extracted_and_nothing_is_recorded(tmp_path):
    """The host has no writer of its own any more. Without a plugin the closed
    session is preserved and that is all that happens -- no LLM call, no
    profile rewrite, no pending record to replay."""
    loop = _loop(tmp_path)
    provider = loop.provider

    reply = await loop._handle_slash_command(_talk(loop), "/new")

    assert reply == "New session started."
    assert provider.calls == 0, "/new reaches no model"
    assert loop.context.memory.read_extraction_cursor() == {}
    assert not loop.context.memory.state_file.exists()
    assert not (tmp_path / "user_memory" / "outbox.jsonl").exists(), "nothing is queued"
    assert loop.context.memory.read_long_term() == "", "user.md is never written automatically"
    assert len(_archives(loop)) == 1


async def test_new_on_a_key_that_never_spoke_is_a_no_op_that_still_resets(tmp_path):
    loop = _loop(tmp_path)
    session = loop.sessions.get_or_create("tui:never-used")

    assert await loop._handle_slash_command(session, "/new") == "New session started."
    assert _archives(loop) == [], "there was nothing to preserve"
    assert loop.sessions.get_or_create("tui:never-used").messages == []


async def test_a_backend_that_raises_does_not_stop_the_reset(tmp_path):
    class Exploding(SlowBackend):
        async def store(self, session_id, messages, *, metadata=None):
            raise RuntimeError("memory service is down")

    loop = _loop(tmp_path, Exploding())

    assert await loop._handle_slash_command(_talk(loop), "/new") == "New session started."
    assert loop.sessions.get_or_create(KEY).messages == []
    assert len(_archives(loop)) == 1


async def test_a_session_that_cannot_be_moved_aside_is_not_cleared(tmp_path, monkeypatch):
    """The preserved copy is the point. If the move fails there is no reset:
    clearing in place would destroy exactly what /new is supposed to keep."""
    loop = _loop(tmp_path)
    session = _talk(loop)
    said = loop.sessions._get_session_path(KEY).read_text(encoding="utf-8")

    def _refuse(_key):
        raise OSError("read-only file system")

    monkeypatch.setattr(loop.sessions, "archive", _refuse)

    reply = await loop._handle_slash_command(session, "/new")

    assert "nothing was cleared" in reply
    assert loop.sessions._get_session_path(KEY).read_text(encoding="utf-8") == said


@pytest.mark.parametrize("cmd", ["/help", "/stop", "not a command"])
async def test_other_input_does_not_close_the_session(tmp_path, cmd):
    loop = _loop(tmp_path)
    session = _talk(loop)

    await loop._handle_slash_command(session, cmd)

    assert _archives(loop) == []
    assert loop.sessions.get_or_create(KEY).messages != []
