"""Lane, worker and scheduler invariants — the ones the module's docstrings
promise, one test each.

The concurrency core has no other guard: a regression here is a hang or a lost
message, not a wrong string. Every test drives a fake runner (no model, no
filesystem) and asserts on the sink's lifecycle events and the handles' futures.
"""

import asyncio
import time

import pytest

from opendde_harness.spine import (
    BusyPolicy,
    ChatType,
    Origin,
    OriginPools,
    Scheduler,
    Source,
    TurnEnded,
    TurnFailed,
    TurnOutcome,
    TurnRequest,
    TurnStarted,
    Usage,
)
from opendde_harness.spine import scheduler as scheduler_mod
from opendde_harness.spine.scheduler import _DEFAULT_IDLE_TTL, SchedulerDrainingError

_USAGE = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def _src(chat: str = "c1") -> Source:
    return Source(channel="chan", chat_id=chat, sender_id="u", chat_type=ChatType.DM)


def _req(
    text: str,
    *,
    chat: str = "c1",
    origin: Origin = Origin.USER,
    busy: BusyPolicy = BusyPolicy.APPEND,
) -> TurnRequest:
    return TurnRequest(origin=origin, source=_src(chat), text=text, busy=busy)


def _outcome(text: str | None = None) -> TurnOutcome:
    return TurnOutcome(usage=_USAGE, explicit_reply=True, text=text)


class Sink:
    """Collects the lifecycle/deliverable events the lane emits."""

    def __init__(self, raise_on: type | None = None) -> None:
        self.events: list = []
        self._raise_on = raise_on

    async def __call__(self, event) -> None:
        self.events.append(event)
        if self._raise_on is not None and isinstance(event, self._raise_on):
            raise RuntimeError("sink is broken")

    def kinds(self) -> list[str]:
        return [type(e).__name__ for e in self.events]


class FnRunner:
    """A TurnRunner whose body is a callable; records the texts it ran."""

    def __init__(self, fn=None) -> None:
        self._fn = fn
        self.ran: list[str] = []

    async def run(self, req, emit, drain):
        self.ran.append(req.text)
        if self._fn is None:
            return _outcome(req.text)
        return await self._fn(req, emit, drain)


def _scheduler(runner, *, sink=None, user: int = 2, system: int = 1):
    return Scheduler(runner, OriginPools(user=user, system=system), sink or Sink())


async def test_append_runs_a_lanes_turns_serially_in_submit_order():
    order: list[tuple[str, str]] = []

    async def fn(req, emit, drain):
        order.append(("start", req.text))
        await asyncio.sleep(0)
        order.append(("end", req.text))
        return _outcome(req.text)

    sched = _scheduler(FnRunner(fn))
    first = sched.submit(_req("one"))
    second = sched.submit(_req("two"))

    assert (await first.result()).text == "one"
    assert (await second.result()).text == "two"
    assert order == [("start", "one"), ("end", "one"), ("start", "two"), ("end", "two")]


async def test_a_waiters_timeout_neither_cancels_the_turn_nor_strands_the_lane():
    """F1: abandoning a wait is not cancelling a turn.

    An RPC disconnect or a client timeout cancels the *waiter*. Unshielded, that
    cancelled the lane's own future, the worker's set_result raised
    InvalidStateError, and every turn queued behind it was stranded unresolved.
    """
    gate = asyncio.Event()
    started = asyncio.Event()

    async def fn(req, emit, drain):
        if req.text == "one":
            started.set()
            await gate.wait()
        return _outcome(req.text)

    runner = FnRunner(fn)
    sched = _scheduler(runner)
    first = sched.submit(_req("one"))
    await started.wait()
    second = sched.submit(_req("two"))

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(first.result(), 0.05)

    gate.set()
    assert (await first.result()).text == "one"
    assert (await second.result()).text == "two"
    assert runner.ran == ["one", "two"]


async def test_interrupt_preempts_the_running_turn_and_jumps_the_append_backlog():
    started = asyncio.Event()

    async def fn(req, emit, drain):
        if req.text == "slow":
            started.set()
            await asyncio.Event().wait()  # cancelled by the interrupter
        return _outcome(req.text)

    runner = FnRunner(fn)
    sink = Sink()
    sched = _scheduler(runner, sink=sink, user=1)
    slow = sched.submit(_req("slow"))
    await started.wait()
    queued = sched.submit(_req("queued"))
    urgent = sched.submit(_req("urgent", busy=BusyPolicy.INTERRUPT))

    assert await slow.result() is None  # preempted, but resolved — not stranded
    assert (await urgent.result()).text == "urgent"
    assert (await queued.result()).text == "queued"
    assert runner.ran == ["slow", "urgent", "queued"]
    assert isinstance(sink.events[1], TurnFailed)
    assert sink.events[1].cancelled is True


async def test_a_drained_inject_merges_into_the_running_turn_and_shares_its_outcome():
    gate = asyncio.Event()
    started = asyncio.Event()
    merged: list[str] = []

    async def fn(req, emit, drain):
        started.set()
        await gate.wait()
        merged.extend(r.text for r in drain())
        return _outcome(req.text)

    runner = FnRunner(fn)
    sched = _scheduler(runner)
    host = sched.submit(_req("host"))
    await started.wait()
    assert sched.has_inflight("chan:c1") is True
    inject = sched.submit(_req("mid", busy=BusyPolicy.INJECT))
    gate.set()

    outcome = await host.result()
    assert merged == ["mid"]
    assert await inject.result() is outcome  # the merged inject shares the host's fate
    assert runner.ran == ["host"]  # it never ran as a turn of its own


async def test_an_inject_the_turn_never_drained_falls_back_to_a_fresh_turn():
    gate = asyncio.Event()
    started = asyncio.Event()

    async def fn(req, emit, drain):
        if req.text == "host":
            started.set()
            await gate.wait()  # returns without ever draining
        return _outcome(req.text)

    runner = FnRunner(fn)
    sched = _scheduler(runner)
    host = sched.submit(_req("host"))
    await started.wait()
    inject = sched.submit(_req("mid", busy=BusyPolicy.INJECT))
    gate.set()

    assert (await host.result()).text == "host"
    assert (await inject.result()).text == "mid"  # no message lost
    assert runner.ran == ["host", "mid"]


async def test_cancelling_a_queued_turn_drops_it_without_running_it():
    gate = asyncio.Event()
    started = asyncio.Event()

    async def fn(req, emit, drain):
        started.set()
        await gate.wait()
        return _outcome(req.text)

    runner = FnRunner(fn)
    sched = _scheduler(runner)
    host = sched.submit(_req("host"))
    await started.wait()
    queued = sched.submit(_req("queued"))
    queued.cancel()
    gate.set()

    assert await queued.result() is None
    assert (await host.result()).text == "host"
    assert runner.ran == ["host"]


async def test_cancelling_the_running_turn_resolves_it_and_emits_one_cancelled_failure():
    started = asyncio.Event()

    async def fn(req, emit, drain):
        started.set()
        await asyncio.Event().wait()
        return _outcome(req.text)

    sink = Sink()
    sched = _scheduler(FnRunner(fn), sink=sink)
    handle = sched.submit(_req("slow"))
    await started.wait()
    handle.cancel()

    assert await handle.result() is None
    assert sink.kinds() == ["TurnStarted", "TurnFailed"]  # paired, exactly once
    assert sink.events[-1].cancelled is True


async def test_stop_cancels_the_running_turn_and_resolves_everything_waiting():
    started = asyncio.Event()

    async def fn(req, emit, drain):
        started.set()
        await asyncio.Event().wait()
        return _outcome(req.text)

    runner = FnRunner(fn)
    sched = _scheduler(runner)
    running = sched.submit(_req("running"))
    await started.wait()
    queued_one = sched.submit(_req("q1"))
    queued_two = sched.submit(_req("q2"))
    mailboxed = sched.submit(_req("mid", busy=BusyPolicy.INJECT))

    assert sched.cancel_conversation("chan:c1") == 4
    for handle in (running, queued_one, queued_two, mailboxed):
        assert await handle.result() is None
    assert runner.ran == ["running"]
    assert sched.cancel_conversation("chan:nothing-here") == 0


async def test_shutdown_seals_the_scheduler_and_resolves_every_unfinished_turn():
    started = asyncio.Event()

    async def fn(req, emit, drain):
        started.set()
        await asyncio.Event().wait()
        return _outcome(req.text)

    sched = _scheduler(FnRunner(fn))
    running = sched.submit(_req("running"))
    await started.wait()
    queued = sched.submit(_req("queued"))

    await asyncio.wait_for(sched.shutdown(grace=0.01), 2.0)

    assert await running.result() is None
    assert await queued.result() is None
    with pytest.raises(SchedulerDrainingError):
        sched.submit(_req("late"))


async def test_the_reaper_reclaims_an_idle_lane_and_leaves_a_busy_one():
    started = asyncio.Event()

    async def fn(req, emit, drain):
        if req.text == "busy":
            started.set()
            await asyncio.Event().wait()
        return _outcome(req.text)

    sched = _scheduler(FnRunner(fn))
    done = sched.submit(_req("done", chat="c1"))
    await done.result()
    busy = sched.submit(_req("busy", chat="c2"))
    await started.wait()

    assert sched._sweep(time.monotonic() + _DEFAULT_IDLE_TTL + 1) == 1
    assert set(sched._lanes) == {"chan:c2"}

    busy.cancel()
    await busy.result()


async def test_submit_from_a_foreign_loop_is_refused():
    sched = _scheduler(FnRunner())

    def in_another_loop() -> None:
        loop = asyncio.new_event_loop()

        async def go() -> None:
            with pytest.raises(RuntimeError, match="event loop"):
                sched.submit(_req("x"))

        try:
            loop.run_until_complete(go())
        finally:
            loop.close()

    await asyncio.to_thread(in_another_loop)


async def test_a_non_user_interrupt_is_demoted_to_append_and_cannot_preempt():
    gate = asyncio.Event()
    started = asyncio.Event()

    async def fn(req, emit, drain):
        if req.text == "user-turn":
            started.set()
            await gate.wait()
        return _outcome(req.text)

    runner = FnRunner(fn)
    sched = _scheduler(runner)
    user_turn = sched.submit(_req("user-turn"))
    await started.wait()
    system_turn = sched.submit(_req("sys", origin=Origin.SUBAGENT, busy=BusyPolicy.INTERRUPT))
    await asyncio.sleep(0)
    gate.set()

    assert (await user_turn.result()).text == "user-turn"  # not preempted
    assert (await system_turn.result()).text == "sys"
    assert runner.ran == ["user-turn", "sys"]


async def test_a_runner_that_emits_a_lifecycle_event_fails_that_turn_only():
    async def fn(req, emit, drain):
        if req.text == "bad":
            await emit(TurnStarted())  # a runner may not emit lifecycle events
        return _outcome(req.text)

    runner = FnRunner(fn)
    sink = Sink()
    sched = _scheduler(runner, sink=sink)
    bad = sched.submit(_req("bad"))
    good = sched.submit(_req("good"))

    assert await bad.result() is None
    assert (await good.result()).text == "good"  # the lane survives
    failures = [e for e in sink.events if isinstance(e, TurnFailed)]
    assert len(failures) == 1
    assert failures[0].cancelled is False
    assert "lifecycle" in failures[0].error


async def test_a_sink_that_raises_while_reporting_a_failure_does_not_strand_the_queue():
    async def fn(req, emit, drain):
        if req.text == "bad":
            raise RuntimeError("turn blew up")
        return _outcome(req.text)

    runner = FnRunner(fn)
    sched = _scheduler(runner, sink=Sink(raise_on=TurnFailed))
    bad = sched.submit(_req("bad"))
    good = sched.submit(_req("good"))

    assert await asyncio.wait_for(bad.result(), 1.0) is None
    assert (await asyncio.wait_for(good.result(), 1.0)).text == "good"


async def test_a_system_turn_does_not_hold_the_user_pools_slot():
    started = asyncio.Event()
    gate = asyncio.Event()

    async def fn(req, emit, drain):
        if req.origin is Origin.SUBAGENT:
            started.set()
            await gate.wait()
        return _outcome(req.text)

    sched = _scheduler(FnRunner(fn), user=1, system=1)
    system_turn = sched.submit(_req("sys", chat="c1", origin=Origin.SUBAGENT))
    await started.wait()
    user_turn = sched.submit(_req("user-turn", chat="c2"))

    # The user turn runs while the system turn still holds the system slot.
    assert (await asyncio.wait_for(user_turn.result(), 1.0)).text == "user-turn"
    gate.set()
    assert (await system_turn.result()).text == "sys"


async def test_a_finished_turn_emits_started_then_ended_with_the_runners_usage():
    sink = Sink()
    sched = _scheduler(FnRunner(), sink=sink)

    outcome = await sched.submit(_req("hi")).result()

    assert outcome is not None
    assert sink.kinds() == ["TurnStarted", "TurnEnded"]
    assert isinstance(sink.events[0], TurnStarted)
    ended = sink.events[1]
    assert isinstance(ended, TurnEnded)
    assert ended.usage is _USAGE
    assert ended.conversation_id == "chan:c1"
    assert ended.latency_ms >= 0


async def test_shutdown_does_not_hang_on_a_turn_that_swallows_its_cancellation(monkeypatch):
    """Phase 4 cancels the stragglers; the wait that follows is bounded.

    A payload that catches CancelledError and keeps going used to hold shutdown
    open forever, and with it every caller still on result().
    """
    monkeypatch.setattr(scheduler_mod, "_SURVIVOR_WAIT_S", 0.05)
    started = asyncio.Event()

    async def fn(req, emit, drain):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.Event().wait()  # refuses to die
        return _outcome(req.text)

    sched = _scheduler(FnRunner(fn))
    handle = sched.submit(_req("stubborn"))
    await started.wait()

    await asyncio.wait_for(sched.shutdown(grace=0.01), 2.0)

    assert await asyncio.wait_for(handle.result(), 1.0) is None
    sched._lanes["chan:c1"].cancel_running()  # let the straggler unwind before the loop closes
    await asyncio.sleep(0)
