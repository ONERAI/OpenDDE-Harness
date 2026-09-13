"""The turn journal: what survives a turn that never finished.

A turn used to reach disk once, after it returned. Everything it did before
that -- a launched job, a written file, a created artifact -- lived in a list in
memory, and Esc or a lost process took the record of it with them. A filesystem
checkpoint cannot reconstruct a submitted remote job or the id it was told once.

So these tests are all about the gap: a tool that completed, a model call that
never came back, and what a fresh SessionManager can see afterwards.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.agent.tools.base import Tool
from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import (
    ErrorClassification,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from opendde_harness.providers.messages import tool_calls_of
from opendde_harness.session.manager import (
    LIFECYCLE_TYPE,
    TOOL_STARTED,
    TURN_COMPLETED,
    TURN_INTERRUPTED,
    TURN_STARTED,
    Session,
    SessionManager,
    new_record_id,
)
from opendde_harness.spine import ChatType, Origin, Source, TurnRequest
from tests import _messages as build
from tests._messages import assistant, text_of, user

MODEL = "primary/model"
KEY = "cli:t"


class LaunchTool(Tool):
    """A tool that changes something this process cannot undo.

    Stands in for the real cases -- a shell command, a spawned subagent, a
    submitted design job -- and reports an id only the journal will ever hold.
    """

    external_effects = True

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "launch_job"

    @property
    def description(self) -> str:
        return "Launch a job on a remote worker."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"spec": {"type": "string"}}, "required": ["spec"]}

    async def execute(self, spec: str) -> str:
        self.calls += 1
        return f"job-7 submitted for {spec}"


class LookupTool(LaunchTool):
    """The same tool, declaring that it changes nothing outside this process."""

    external_effects = False

    @property
    def name(self) -> str:
        return "lookup"

    async def execute(self, spec: str) -> str:
        self.calls += 1
        return f"looked up {spec}"


class _Provider(LLMProvider):
    """Asks for one tool call, then blocks forever on the next call.

    The shape of the failure this is all about: the tool has already run and
    changed the world, and the model call that would have used its result never
    comes back.
    """

    def __init__(self, tool: str = "launch_job") -> None:
        super().__init__()
        self.tool = tool
        self.calls = 0
        self.blocked = asyncio.Event()

    async def chat(self, messages, tools=None, model=None, **kwargs):
        return await self.chat_with_retry(messages=messages, tools=tools, model=model, **kwargs)

    async def chat_with_retry(self, messages=None, tools=None, model=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="launching",
                tool_calls=[ToolCallRequest(id="call-1", name=self.tool, arguments={"spec": "assay"})],
                finish_reason="tool_calls",
            )
        self.blocked.set()
        await asyncio.Event().wait()  # never returns
        raise AssertionError("unreachable")

    def classify_error(self, exc):
        return ErrorClassification("no_model")

    def get_default_model(self):
        return MODEL


class _Answering(_Provider):
    """Answers the second call instead of blocking, so the turn completes."""

    async def chat_with_retry(self, messages=None, tools=None, model=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="launching",
                tool_calls=[ToolCallRequest(id="call-1", name=self.tool, arguments={"spec": "assay"})],
                finish_reason="tool_calls",
            )
        return LLMResponse(content="the job is running", finish_reason="stop")


def _request(key: str = KEY) -> TurnRequest:
    channel, _, chat = key.partition(":")
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel=channel, chat_id=chat, sender_id="user", chat_type=ChatType.DM),
        text="launch the assay",
        conversation=key,
    )


def _loop(tmp_path: Path, provider: LLMProvider, tool: Tool | None = None, backend: Any = None) -> AgentLoop:
    loop = AgentLoop(provider, tmp_path, AgentLoopSettings(model=MODEL), backend=backend)
    if tool is not None:
        loop.tools.register(tool)
    return loop


def _reloaded(tmp_path: Path, key: str = KEY) -> Session:
    """The session as a process that has never seen it reads it back."""
    session = SessionManager(tmp_path).peek(key)
    assert session is not None
    return session


def _kinds(session: Session) -> list[str]:
    return [r["kind"] for r in session.lifecycle]


# ── Cancellation mid-turn ──────────────────────────────────────────────


async def test_a_cancelled_turn_keeps_the_tool_it_completed_exactly_once(tmp_path):
    """The whole point. The tool ran, the next model call never returned, and a
    fresh reader must find the call and its result -- once -- plus the fact that
    the turn did not finish."""
    provider = _Provider()
    tool = LaunchTool()
    loop = _loop(tmp_path, provider, tool)

    task = asyncio.create_task(loop._process_message(_request()))
    await asyncio.wait_for(provider.blocked.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    session = _reloaded(tmp_path)
    results = [m for m in session.messages if m.get("role") == "toolResult"]
    assert len(results) == 1, "the completed tool result survives exactly once"
    assert "job-7" in text_of(results[0])
    assert results[0]["toolCallId"] == "call-1"
    assert tool.calls == 1

    assert _kinds(session) == [TURN_STARTED, TOOL_STARTED, TURN_INTERRUPTED]
    assert set(session.turn_status().values()) == {"interrupted"}
    assert session.uncertain_tool_calls() == [], "the tool did report back"


async def test_the_accepted_user_message_is_on_disk_before_the_first_model_call(tmp_path):
    """A turn whose very first call never returns still leaves the request that
    started it."""

    class _Silent(_Provider):
        async def chat_with_retry(self, messages=None, tools=None, model=None, **kwargs):
            self.calls += 1
            self.blocked.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    provider = _Silent()
    loop = _loop(tmp_path, provider, LaunchTool())

    task = asyncio.create_task(loop._process_message(_request()))
    await asyncio.wait_for(provider.blocked.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    session = _reloaded(tmp_path)
    assert [m["content"] for m in session.messages if m["role"] == "user"] == ["launch the assay"]
    assert _kinds(session) == [TURN_STARTED, TURN_INTERRUPTED]


async def test_a_state_changing_tool_cancelled_mid_call_is_recorded_as_uncertain(tmp_path):
    """Cancelled inside the tool, so nothing ever reported back. The journal says
    which tool was given what, and the harness will not run it again."""
    entered = asyncio.Event()

    class Wedged(LaunchTool):
        async def execute(self, spec: str) -> str:
            self.calls += 1
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    tool = Wedged()
    loop = _loop(tmp_path, _Provider(), tool)

    task = asyncio.create_task(loop._process_message(_request()))
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    session = _reloaded(tmp_path)
    uncertain = session.uncertain_tool_calls()
    assert len(uncertain) == 1
    assert uncertain[0]["name"] == "launch_job"
    assert uncertain[0]["arguments"] == {"spec": "assay"}
    assert [m.get("role") for m in session.messages if m.get("role") == "toolResult"] == []
    assert tool.calls == 1

    # And the unanswered call never reaches the next request, which would be a
    # 400 on every turn after it -- and an invitation to run the job twice.
    assert all(not tool_calls_of(m) for m in session.get_history())


async def test_a_read_only_tool_is_not_journaled_as_started(tmp_path):
    """Nothing outside this process changed, so there is nothing to reconstruct."""
    provider = _Provider(tool="lookup")
    loop = _loop(tmp_path, provider, LookupTool())

    task = asyncio.create_task(loop._process_message(_request()))
    await asyncio.wait_for(provider.blocked.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _kinds(_reloaded(tmp_path)) == [TURN_STARTED, TURN_INTERRUPTED]


async def test_a_turn_that_finished_is_completed_not_interrupted(tmp_path):
    loop = _loop(tmp_path, _Answering(), LaunchTool())

    reply = await loop._process_message(_request())

    assert reply.content == "the job is running"
    session = _reloaded(tmp_path)
    assert _kinds(session) == [TURN_STARTED, TOOL_STARTED, TURN_COMPLETED]
    assert set(session.turn_status().values()) == {"completed"}
    assert [m["role"] for m in session.messages] == ["user", "assistant", "toolResult", "assistant"]
    turn_ids = {m["turn_id"] for m in session.messages}
    assert len(turn_ids) == 1, "one turn, one turn_id on every message it recorded"


async def test_the_turn_is_committed_before_the_caller_is_told_it_finished(tmp_path):
    """The final assistant message and ``turn.completed`` land before anything
    optional runs. Cancelled during remote compaction, a turn the user has
    already read used to be lost with it."""
    seen: list[list[str]] = []
    loop = _loop(tmp_path, _Answering(), LaunchTool())
    original = loop._maybe_compact_remote

    async def _watch(session, outcome, *, session_key=""):
        seen.append(_kinds(_reloaded(tmp_path)))
        return await original(session, outcome, session_key=session_key)

    loop._maybe_compact_remote = _watch

    await loop._process_message(_request())

    assert seen == [[TURN_STARTED, TOOL_STARTED, TURN_COMPLETED]]


# ── Reload and ids ─────────────────────────────────────────────────────


def test_a_session_file_written_before_ids_existed_loads_and_gets_them(tmp_path):
    """Adding a field is not a reason to rewrite a conversation, and an
    append-only log has nowhere to put one anyway."""
    manager = SessionManager(tmp_path)
    path = manager._get_session_path(KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                json.dumps({"_type": "metadata", "key": KEY, "metadata": {}, "last_consolidated": 0}),
                json.dumps({"role": "user", "content": "old news"}),
                json.dumps({"role": "assistant", "content": "indeed"}),  # the pre-pi shape
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    before = path.read_text(encoding="utf-8")

    session = manager.peek(KEY)
    assert session is not None
    assert [text_of(m) for m in session.messages] == ["old news", "indeed"]
    ids = [m["id"] for m in session.messages]
    assert all(isinstance(i, str) and i for i in ids)
    assert len(set(ids)) == 2
    assert session.generation == 0
    assert path.read_text(encoding="utf-8") == before, "reading a session does not rewrite it"


def test_the_journal_writes_nothing_twice_when_the_loop_rebuilds_its_tail(tmp_path):
    """``_journal_id`` is the journal's idempotence key, host-only and stamped on
    the live message dict. The loop does not mutate a message it shrinks: it
    elides an image and excerpts a tool body into *copies*, and the list it then
    keeps working from is the one holding them. So the stamp has to survive the
    copy, or every flush after the first shrink writes the whole tail again.
    """
    from opendde_harness.agent.loop.main import AgentLoop
    from opendde_harness.session.journal import JOURNAL_KEY, TurnJournal

    manager = SessionManager(tmp_path)
    session = manager.get_or_create(KEY)
    messages = [user("look at these")]
    for i in range(4):
        messages.append(assistant(calls=[(f"c{i}", "read", {})]))
        messages.append(
            build.tool_result(
                f"c{i}",
                "read",
                [msg.text_block(f"body {i} " * 200), msg.image_block("AAAA", "image/png")],
            )
        )
    journal = TurnJournal(
        manager,
        session,
        sanitize=lambda live: {k: v for k, v in live.items() if k != JOURNAL_KEY},
        start_index=0,
    )

    journal.open(messages)
    recorded = len(session.messages)
    assert recorded == len(messages), "the whole tail is on record once"
    assert all(JOURNAL_KEY in m for m in messages), "and every live message carries the stamp"

    shrunk, elided = AgentLoop._emergency_shrink(messages)
    assert elided, "there was something to shrink"
    assert shrunk is not messages and any(a is not b for a, b in zip(shrunk, messages)), "copies, not mutation"
    journal.flush(shrunk)

    assert len(session.messages) == recorded, "the rebuilt tail is already on record"
    # And what is on record is what was said, not the elided stand-in.
    bodies = [text_of(m) for m in session.messages if m["role"] == "toolResult"]
    assert all("body" in body for body in bodies)


def test_the_durable_record_of_an_assistant_turn_keeps_every_field_pi_sent(tmp_path):
    """The record on disk *is* pi's message. A field dropped here is a thinking
    signature or a native tool-call id that cannot be replayed, and the next
    request re-renders a turn the model would otherwise recognise as its own.
    """
    loop = _loop(tmp_path, _Answering(), LaunchTool())
    answered = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "weighing it", "thinkingSignature": "sig-1"},
            {"type": "text", "text": "on it", "textSignature": "tsig"},
            {"type": "toolCall", "id": "fc_0|1", "name": "launch_job", "arguments": {"spec": "assay"}},
        ],
        "api": "openai-responses",
        "provider": "openai",
        "model": "gpt-5",
        "responseModel": "gpt-5-2026-01-01",
        "responseId": "resp_1",
        "providerThinkingLevel": "medium",
        "rawStopReason": "tool_use",
        "stopReason": "toolUse",
        "endTurn": False,
        "timestamp": 17,
    }

    entry = loop._sanitized_record(dict(answered))

    assert entry == {**answered, "timestamp": 17}, "every field pi sent, and its own timestamp"


def test_record_ids_sort_in_the_order_they_were_minted():
    ids = [new_record_id() for _ in range(500)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 500


def test_lifecycle_records_are_not_messages(tmp_path):
    """Every reader of the message projection -- the provider history, the resume
    wire, the message count a session list shows -- is unaffected by the journal."""
    manager = SessionManager(tmp_path)
    session = manager.get_or_create(KEY)
    session.record_lifecycle(TURN_STARTED, turn_id="t1")
    session.record(user("hello"))
    session.record_lifecycle(TURN_COMPLETED, turn_id="t1")
    manager.save(session)

    manager.invalidate(KEY)
    reloaded = manager.peek(KEY)
    assert reloaded is not None
    assert [m["role"] for m in reloaded.messages] == ["user"]
    assert _kinds(reloaded) == [TURN_STARTED, TURN_COMPLETED]
    assert [r.get("kind") or r["role"] for r in reloaded.records()] == [TURN_STARTED, "user", TURN_COMPLETED]
    assert manager.list_sessions()[0]["message_count"] == 1
    assert manager.records(KEY) == reloaded.records()


def test_the_journal_survives_a_save_that_appends_after_it(tmp_path):
    """Interleaving is the file's own order, appended in the same rule it is read
    back with -- so a second save does not reshuffle the first."""
    manager = SessionManager(tmp_path)
    session = manager.get_or_create(KEY)
    session.record_lifecycle(TURN_STARTED, turn_id="t1")
    session.record(user("one"))
    manager.save(session)
    session.record_lifecycle(TURN_COMPLETED, turn_id="t1")
    session.record_lifecycle(TURN_STARTED, turn_id="t2")
    session.record(user("two"))
    manager.save(session)

    manager.invalidate(KEY)
    reloaded = manager.peek(KEY)
    assert reloaded is not None
    assert [r.get("kind") or r["content"] for r in reloaded.records()] == [
        TURN_STARTED,
        "one",
        TURN_COMPLETED,
        TURN_STARTED,
        "two",
    ]


async def test_new_bumps_the_generation_and_the_closed_file_keeps_its_records(tmp_path):
    loop = _loop(tmp_path, _Answering(), LaunchTool())
    await loop._process_message(_request())
    live = loop.sessions._get_session_path(KEY)
    said = live.read_text(encoding="utf-8")

    reply = await loop._handle_slash_command(loop.sessions.get_or_create(KEY), "/new")

    assert reply == "New session started."
    closed = sorted((loop.sessions.sessions_dir / "_closed").rglob("*.jsonl"))
    assert len(closed) == 1
    assert closed[0].read_text(encoding="utf-8") == said
    kept = [json.loads(line) for line in said.splitlines() if line.strip()]
    assert [r["kind"] for r in kept if r.get("_type") == LIFECYCLE_TYPE] == [
        TURN_STARTED,
        TOOL_STARTED,
        TURN_COMPLETED,
    ]

    fresh = _reloaded(tmp_path)
    assert fresh.messages == []
    assert fresh.lifecycle == []
    assert fresh.generation == 1


async def test_a_turn_after_a_long_history_still_records_its_own_tail(tmp_path):
    """The journal holds the offset its turn starts at and slices for itself. Given
    an already-sliced tail it sliced twice, and every message of a turn on a long
    session -- the answer included -- fell past the end of the list."""
    loop = _loop(tmp_path, _Answering(), LaunchTool())
    session = loop.sessions.get_or_create(KEY)
    for n in range(12):
        session.record(user(f"old question {n}"))
        session.record(assistant(f"old answer {n}"))
    loop.sessions.save(session)
    before = len(session.messages)

    await loop._process_message(_request())

    reloaded = _reloaded(tmp_path)
    assert [m["role"] for m in reloaded.messages[before:]] == ["user", "assistant", "toolResult", "assistant"]
    assert text_of(reloaded.messages[-1]) == "the job is running"
    assert _kinds(reloaded) == [TURN_STARTED, TOOL_STARTED, TURN_COMPLETED]


def test_undoing_a_turn_takes_its_lifecycle_with_it(tmp_path):
    """A ``turn.started`` whose messages no longer exist reads as an interrupted
    turn, and the next resume would say so."""
    manager = SessionManager(tmp_path)
    session = manager.get_or_create(KEY)
    session.record_lifecycle(TURN_STARTED, turn_id="t1")
    session.record({**user("one"), "turn_id": "t1"})
    session.record({**assistant("first"), "turn_id": "t1"})
    session.record_lifecycle(TURN_COMPLETED, turn_id="t1")
    session.record_lifecycle(TURN_STARTED, turn_id="t2")
    session.record({**user("two"), "turn_id": "t2"})
    manager.save(session)

    assert session.undo_last_turn() == 1
    manager.save(session)
    manager.invalidate(KEY)

    reloaded = manager.peek(KEY)
    assert reloaded is not None
    assert [text_of(m) for m in reloaded.messages] == ["one", "first"]
    assert reloaded.turn_status() == {"t1": "completed"}
