"""G3 — two conversations, two lanes, two of everything the turn reads.

The loop is one process-wide object and the things a turn decides with used to
live on it: the model, the window, the rendered prefix. Two conversations
running at once on different models is exactly where a single copy of any of
them is wrong, and wrong silently -- one session's instructions sent to the
other's model, or a history trimmed to a window that belongs to somebody else.

So both turns are held at the service until both are outstanding, and then every
request is checked against the conversation that made it: its own system prefix,
its own model, its own window's verdict about how much history fits, and its own
journal.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from opendde_harness.context_engine import project_instructions as instructions
from opendde_harness.session.manager import SessionManager
from opendde_harness.spine import OriginPools, Scheduler
from tests import _messages as build
from tests._gate import (
    CODEX,
    MODEL,
    Echo,
    Events,
    bind,
    declare,
    gate_config,
    loop_for,
    request,
    requires_service,
    service,  # noqa: F401  -- the fixture the gate shares
    system_prompt,
)

pytestmark = requires_service

SMALL = "tui:small"
LARGE = "tui:large"

#: Small enough that the prefix plus the seeded history do not fit, large
#: enough that the mandatory set still does.
SMALL_WINDOW = 24_000
LARGE_WINDOW = 400_000

#: One seeded message, sized so that forty exchanges are well past the small
#: window and well inside the large one.
FILLER = "filler " * 700

#: How many exchanges each conversation is seeded with.
EXCHANGES = 20


class LoopRunner:
    def __init__(self, loop) -> None:
        self.loop = loop

    async def run(self, req, emit, drain):
        return await self.loop.run_turn(req, emit, drain, stream=True)


@pytest.fixture
def forget_instruction_state():
    """The per-conversation instruction switches live in a module global."""
    yield
    instructions.SESSIONS.clear()


def _seed(workspace: Path, key: str, label: str, exchanges: int = EXCHANGES) -> int:
    """A history long enough that a small window cannot hold all of it."""
    manager = SessionManager(workspace)
    session = manager.get_or_create(key)
    for n in range(exchanges):
        turn = f"{label}-t{n}"
        session.record({**build.user(f"{label} question {n}. " + FILLER), "turn_id": turn})
        session.record({**build.assistant(f"{label} answer {n}. " + FILLER), "turn_id": turn})
    manager.save(session)
    return len(session.messages)


async def test_two_conversations_keep_their_own_prefix_window_and_journal(
    tmp_path,
    service,  # noqa: F811
    monkeypatch,
    forget_instruction_state,
):
    # Standing instructions the launch directory carries, switched off for one
    # conversation only -- which is the one thing that makes the rendered
    # project-instructions block differ per conversation.
    monkeypatch.chdir(tmp_path)
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# House rules\n\nAlways name the assay plate.\n", encoding="utf-8")
    instructions.SESSIONS.set(LARGE, agents.resolve().as_posix(), enabled=False)

    stored_small = _seed(tmp_path, SMALL, "small")
    stored_large = _seed(tmp_path, LARGE, "large")

    providers = {
        **declare("faux", window=SMALL_WINDOW),
        **declare("faux-codex", window=LARGE_WINDOW),
    }
    config = gate_config(tmp_path, model=MODEL, providers=providers, defaults={"maxToolIterations": 1})
    loop = loop_for(service, config, tools=[Echo()])
    loop.set_session_binding(SMALL, bind(service, MODEL, config))
    loop.set_session_binding(LARGE, bind(service, CODEX, config))
    scheduler = Scheduler(LoopRunner(loop), OriginPools(user=2, system=1), Events())

    # Neither turn's first call is answered until both have been made.
    service.barrier = 2
    try:
        first = scheduler.submit(request("what did the small one say", SMALL))
        second = scheduler.submit(request("what did the large one say", LARGE))
        outcomes = await asyncio.wait_for(asyncio.gather(first.result(), second.result()), timeout=120)
        assert all(outcome is not None for outcome in outcomes), "both turns answered"
    finally:
        await scheduler.shutdown(grace=0.0)
        await loop.close_mcp()

    by_model: dict[str, list[dict]] = {"faux": [], "faux-codex": []}
    for asked, context in zip(service.models_asked, service.contexts, strict=True):
        by_model[asked[0]].append(context)
    assert by_model["faux"] and by_model["faux-codex"], "both lanes reached the service"

    # 1. Each request carries its own conversation's prefix.
    for context in by_model["faux"]:
        prompt = system_prompt(context)
        assert MODEL in prompt, "the identity block names the model this request is going to"
        assert CODEX not in prompt
        assert "Always name the assay plate" in prompt, "this conversation still follows the file"
    for context in by_model["faux-codex"]:
        prompt = system_prompt(context)
        assert CODEX in prompt
        assert MODEL not in prompt
        assert "Always name the assay plate" not in prompt, "this conversation switched the file off"

    # 2. Each request's history was trimmed against its own declared window.
    small_sent = len(by_model["faux"][0]["messages"])
    large_sent = len(by_model["faux-codex"][0]["messages"])
    assert small_sent < stored_small, f"the small window dropped history ({small_sent} of {stored_small})"
    assert large_sent >= stored_large, f"the large window dropped none ({large_sent} of {stored_large})"
    # The selection policy protects the first three user messages when no
    # compaction marker is in force, and requires the newest exchange; what a
    # small window drops is between them.
    small_history = str(by_model["faux"][0]["messages"])
    large_history = str(by_model["faux-codex"][0]["messages"])
    dropped = [n for n in range(EXCHANGES) if f"small question {n}." not in small_history]
    assert dropped, "the small window dropped exchanges"
    assert 0 not in dropped, "the protected head stayed"
    assert EXCHANGES - 1 not in dropped, "and the newest exchange is mandatory"
    assert [n for n in range(EXCHANGES) if f"large question {n}." not in large_history] == [], (
        "the large window dropped nothing"
    )

    # 3. Neither request carries a word of the other conversation.
    assert "large question" not in str(by_model["faux"])
    assert "small question" not in str(by_model["faux-codex"])

    # 4. And neither journal holds the other's records.
    reader = SessionManager(tmp_path)
    small_session = reader.peek(SMALL)
    large_session = reader.peek(LARGE)
    assert small_session is not None and large_session is not None
    assert "large" not in str(small_session.messages)
    assert "small" not in str(large_session.messages)
    small_turns = {record.get("turn_id") for record in small_session.lifecycle}
    large_turns = {record.get("turn_id") for record in large_session.lifecycle}
    assert small_turns and large_turns and not (small_turns & large_turns), "no turn id in both journals"
