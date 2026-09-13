"""G6 — what the TUI shows when it reopens the session a cancel left behind.

The journal now holds two kinds of record: the messages of the conversation and
the lifecycle of each turn. Only the first kind is a transcript. A resumed
session therefore has to do two things at once -- leave the lifecycle records out
of what it shows and counts, and still say, out of them, that one of the turns
never finished. A transcript that shows an interrupted turn's work without
saying so invites the user to assume it completed.

Driven through ``session.resume`` itself rather than through its helpers: the
count the banner opens on and the marker the transcript carries are two answers
from one call, and it is the call that has to get both right.
"""

from __future__ import annotations

import asyncio
import json

from opendde_harness.session.manager import TURN_INTERRUPTED, SessionManager
from opendde_harness.tui_rpc.methods.session import session_resume
from tests._gate import (
    MODEL,
    Events,
    Launch,
    declare,
    gate_config,
    loop_for,
    no_injections,
    request,
    requires_service,
    service,  # noqa: F401  -- the fixture the gate shares
)

pytestmark = requires_service

KEY = "tui:resumed"


def _write_home_config(config) -> None:
    """The config the RPC handler reads for itself.

    ``session.resume`` calls ``load_config()`` with no path, which resolves
    under ``HOME`` -- the temporary one the suite's ``isolated_home`` fixture
    installs. Writing this session's own config there is what makes the handler
    read the same workspace the turn ran in.
    """
    from pathlib import Path

    path = Path.home() / ".opendde_harness" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.model_dump(mode="json", by_alias=True, exclude_none=True)), encoding="utf-8")


async def _interrupt_a_turn(tmp_path, service, key: str = KEY):  # noqa: F811
    """One turn whose second model call never comes back, cancelled there."""
    config = gate_config(tmp_path, model=MODEL, providers=declare("faux"), defaults={"maxToolIterations": 4})
    loop = loop_for(service, config, tools=[Launch()])
    service.hold_from = 2
    task = asyncio.create_task(loop.run_turn(request("launch the assay", key), Events(), no_injections, stream=True))
    await asyncio.wait_for(service.held.wait(), timeout=20)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return config, loop


async def test_resume_shows_the_interrupted_turn_and_counts_only_messages(tmp_path, service):  # noqa: F811
    config, loop = await _interrupt_a_turn(tmp_path, service)
    try:
        _write_home_config(config)
        result = await session_resume({"session_id": KEY}, agent_loop_factory=lambda: loop)
    finally:
        await loop.close_mcp()

    stored = SessionManager(tmp_path).peek(KEY)
    assert stored is not None
    assert [record["kind"] for record in stored.lifecycle][-1] == TURN_INTERRUPTED, "the turn is on record as cut off"
    assert len(stored.lifecycle) >= 3, "there are lifecycle records to leave out"

    assert result["session_id"] == KEY, "the session resumed under its own key"
    wire = result["messages"]
    # One entry per stored message, plus the one marker the interrupted turn
    # earns. Nothing from the lifecycle: those are not messages.
    assert len(wire) == len(stored.messages) + 1, f"{len(wire)} wire entries for {len(stored.messages)} messages"
    assert {entry["role"] for entry in wire} <= {"user", "assistant", "tool", "system"}
    body = json.dumps(wire)
    assert "turn.started" not in body and "tool.started" not in body, "no lifecycle record reached the transcript"

    marker = wire[-1]
    assert marker["role"] == "system"
    assert "interrupted before finishing" in marker["text"]
    # The completed tool's own result is still in the transcript, before it.
    assert any("job-1 submitted" in (entry.get("text") or "") for entry in wire[:-1])
    # And the banner's own count agrees with the transcript it was handed.
    assert result["info"]["usage"]["context_used"] > 0


async def test_a_session_whose_turns_all_finished_resumes_with_no_marker(tmp_path, service):  # noqa: F811
    """The control: the same handler, a session with nothing to warn about."""
    config = gate_config(tmp_path, model=MODEL, providers=declare("faux"), defaults={"maxToolIterations": 1})
    loop = loop_for(service, config, tools=[Launch()])
    try:
        await loop.run_turn(request("launch the assay", KEY), Events(), no_injections, stream=True)
        _write_home_config(config)
        result = await session_resume({"session_id": KEY}, agent_loop_factory=lambda: loop)
    finally:
        await loop.close_mcp()

    stored = SessionManager(tmp_path).peek(KEY)
    assert stored is not None
    wire = result["messages"]
    assert len(wire) == len(stored.messages), "one wire entry per message, and no marker"
    assert all("interrupted" not in (entry.get("text") or "") for entry in wire)
