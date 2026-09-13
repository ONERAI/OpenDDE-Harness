"""G7 — indexing a turn is not the turn's business, and a refused write is retried.

Two promises used to be one: a turn committed its conversation and handed it to
the memory backend in the same breath, so a slow index was a slow turn. Here the
turn appends one line to a durable queue and is done, and a background drain owes
the rest -- which means two things have to hold at once. A backend that refuses a
write once and takes it the second time must end with the turn stored, once. And
a backend that never answers at all must not delay the turn the user is waiting
for.

Driven through a real turn on the scripted service rather than through the queue
alone: the thing being checked is where the work sits relative to the answer.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from opendde_harness.memory_engine import dispatch as dispatch_module
from opendde_harness.session.manager import TURN_COMPLETED, SessionManager
from tests._gate import (
    MODEL,
    Echo,
    Events,
    declare,
    gate_config,
    loop_for,
    no_injections,
    request,
    requires_service,
    service,  # noqa: F401  -- the fixture the gate shares
)

pytestmark = requires_service

KEY = "tui:outbox"

#: What a user is allowed to feel while a wedged backend is still thinking.
TURN_BOUND_S = 2.0


class Backend:
    """A memory backend stub: records every write and answers to a script."""

    def __init__(self, *, refusals: int = 0, delay: float = 0.0) -> None:
        self.refusals = refusals
        self.delay = delay
        self.calls: list[tuple[str, int]] = []

    async def recall(self, query, *, user_id=None, agent_id=None, top_k):
        return []

    async def store(self, session_id: str, messages: list[dict[str, Any]], *, metadata=None) -> bool:
        self.calls.append((session_id, len(messages)))
        if self.delay:
            import asyncio

            await asyncio.sleep(self.delay)
        if len(self.calls) <= self.refusals:
            return False
        return True

    async def feedback(self, signals: dict[str, Any]) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _config(tmp_path):
    return gate_config(tmp_path, model=MODEL, providers=declare("faux"), defaults={"maxToolIterations": 1})


async def test_a_write_refused_once_is_retried_and_the_turn_is_stored_once(tmp_path, service, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dispatch_module, "STORE_RETRY_BACKOFF_S", 0.0)
    backend = Backend(refusals=1)
    loop = loop_for(service, _config(tmp_path), tools=[Echo()], backend=backend)

    try:
        await loop.run_turn(request("index this", KEY), Events(), no_injections, stream=True)
        await loop.drain_backend_stores(timeout=10)
    finally:
        await loop.close_mcp()

    assert len(backend.calls) == 2, "refused once, taken the second time"
    assert len({call for call in backend.calls}) == 1, "the same turn, not a second one synthesised for the retry"
    assert loop.outbox.pending() == [], "and the queue is empty rather than holding a turn it already stored"
    assert loop.outbox.deferred == 0, "nothing was abandoned"


async def test_a_wedged_backend_does_not_delay_the_turn_the_user_is_waiting_for(tmp_path, service):  # noqa: F811
    """The whole point of the queue. The answer is committed; indexing it is a
    separate promise, and one that cannot be kept here."""
    backend = Backend(delay=30.0)
    loop = loop_for(service, _config(tmp_path), tools=[Echo()], backend=backend)

    started = time.monotonic()
    try:
        outcome = await loop.run_turn(request("index this", KEY), Events(), no_injections, stream=True)
        elapsed = time.monotonic() - started
    finally:
        task = loop._outbox_task
        if task is not None:
            task.cancel()
        await loop.close_mcp()

    assert outcome.text, "the turn answered"
    assert elapsed < TURN_BOUND_S, f"the turn waited {elapsed:.1f}s on a backend that never answered"

    session = SessionManager(tmp_path).peek(KEY)
    assert session is not None
    assert [record["kind"] for record in session.lifecycle][-1] == TURN_COMPLETED, "and it was committed as complete"
    assert loop.outbox.pending(), "the turn is queued, waiting for a backend that will answer"


@pytest.mark.usefixtures("service")
async def test_with_no_backend_configured_nothing_is_queued(tmp_path, service):  # noqa: F811
    """The control: the queue exists for a backend, and there is none here."""
    loop = loop_for(service, _config(tmp_path), tools=[Echo()])

    try:
        await loop.run_turn(request("index this", KEY), Events(), no_injections, stream=True)
    finally:
        await loop.close_mcp()

    assert loop.backend is None
    assert loop.outbox.pending() == [], "no backend, nothing owed"
