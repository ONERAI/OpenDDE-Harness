"""G1 — Esc in the middle of a tool loop, end to end through a real lane.

The shape of the failure: a tool that changed something outside this process has
already run, and the model call that would have used its result never comes
back. What has to survive is the completed exchange -- once, not twice -- plus
the fact that the turn did not finish; and the conversation's lane has to be
usable afterwards, because a wedged lane means every later turn on that
conversation hangs with no error to show for it.

Driven through the spine :class:`~opendde_harness.spine.Scheduler`, because the
lane is the thing that would wedge: ``handle.cancel()`` is what the TUI's Esc
reaches, and the next ``submit`` on the same conversation is what proves the
lane came back.
"""

from __future__ import annotations

import asyncio

import pytest

from opendde_harness.session.manager import (
    TOOL_STARTED,
    TURN_COMPLETED,
    TURN_INTERRUPTED,
    TURN_STARTED,
    SessionManager,
)
from opendde_harness.spine import OriginPools, Scheduler, TurnFailed
from tests._gate import (
    MODEL,
    Events,
    Launch,
    declare,
    gate_config,
    loop_for,
    request,
    requires_service,
    service,  # noqa: F401  -- the fixture the gate shares
)

pytestmark = requires_service

KEY = "tui:cancel"


class LoopRunner:
    """The lane's runner: one real :class:`AgentLoop`, nothing in front of it."""

    def __init__(self, loop) -> None:
        self.loop = loop

    async def run(self, req, emit, drain):
        return await self.loop.run_turn(req, emit, drain, stream=True)


def _kinds(tmp_path, key: str = KEY) -> list[str]:
    """The journal as a process that has never seen this session reads it."""
    session = SessionManager(tmp_path).peek(key)
    assert session is not None, "the turn left a session file"
    return [record["kind"] for record in session.lifecycle]


async def test_cancelling_mid_tool_loop_keeps_the_completed_exchange_and_frees_the_lane(tmp_path, service):  # noqa: F811
    config = gate_config(tmp_path, model=MODEL, providers=declare("faux"), defaults={"maxToolIterations": 4})
    tool = Launch()
    loop = loop_for(service, config, tools=[tool])
    sink = Events()
    scheduler = Scheduler(LoopRunner(loop), OriginPools(user=2, system=1), sink)

    # The second model call of the turn never comes back: by then the tool has
    # run and reported an id only the journal will ever hold.
    service.hold_from = 2
    handle = scheduler.submit(request("launch the assay", KEY))
    await asyncio.wait_for(service.held.wait(), timeout=20)

    handle.cancel()
    assert await handle.result() is None, "a cancelled turn has no outcome"
    failures = sink.of(TurnFailed)
    assert len(failures) == 1 and failures[0].cancelled is True, "the lane reported the cancel once"

    session = SessionManager(tmp_path).peek(KEY)
    assert session is not None
    results = [m for m in session.messages if m.get("role") == "toolResult"]
    assert len(results) == 1, "the completed tool result survives exactly once"
    assert "job-1 submitted" in str(results[0]["content"]), "and it is the one the tool actually reported"
    assert tool.seen and len(tool.seen) == 1, "the tool ran once"
    assert _kinds(tmp_path) == [TURN_STARTED, TOOL_STARTED, TURN_INTERRUPTED]
    assert set(session.turn_status().values()) == {"interrupted"}
    assert session.uncertain_tool_calls() == [], "the tool did report back before the cancel"

    # The lane: a next turn on the same conversation runs to an answer.
    service.hold_from = None
    service.release.set()
    again = scheduler.submit(request("and what did it say", KEY))
    outcome = await asyncio.wait_for(again.result(), timeout=60)
    try:
        assert outcome is not None, "the lane was not wedged by the cancelled turn"
        assert outcome.text, "the second turn answered"
    finally:
        await scheduler.shutdown(grace=0.0)
        await loop.close_mcp()

    reloaded = SessionManager(tmp_path).peek(KEY)
    assert reloaded is not None
    statuses = sorted(reloaded.turn_status().values())
    assert statuses == ["completed", "interrupted"], "two turns, each with its own verdict"
    assert _kinds(tmp_path).count(TURN_INTERRUPTED) == 1, "the interrupted turn is recorded once"
    assert TURN_COMPLETED in _kinds(tmp_path)
    # And the exchange the cancelled turn completed is still there exactly once,
    # beside the second turn's own. The job id is per call, so the cancelled
    # turn's own result is the only one that can be counted.
    bodies = [str(m.get("content")) for m in reloaded.messages if m.get("role") == "toolResult"]
    assert sum("job-1 submitted" in body for body in bodies) == 1, "not replayed, not written twice"


@pytest.mark.usefixtures("service")
async def test_a_turn_nobody_cancelled_is_completed_not_interrupted(tmp_path, service):  # noqa: F811
    """The control: the same rig, no cancel, and the journal says so."""
    config = gate_config(tmp_path, model=MODEL, providers=declare("faux"), defaults={"maxToolIterations": 1})
    loop = loop_for(service, config, tools=[Launch()])
    scheduler = Scheduler(LoopRunner(loop), OriginPools(user=2, system=1), Events())

    handle = scheduler.submit(request("launch the assay", KEY))
    try:
        assert await asyncio.wait_for(handle.result(), timeout=60) is not None
    finally:
        await scheduler.shutdown(grace=0.0)
        await loop.close_mcp()

    assert _kinds(tmp_path) == [TURN_STARTED, TOOL_STARTED, TURN_COMPLETED]
