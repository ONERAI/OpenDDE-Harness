"""One cost per session: the footer and ``/status`` answer with the same number.

The owner opened a session that had made forty-five model calls over four turns
and found the footer and ``/status`` disagreeing about what it had cost. They
were reporting different things without saying so: the footer showed the *last
turn's* running cost, ``/status`` the *session's*, and a resumed session's
footer started at ``$0.000`` however long its conversation was because the
process had counted none of the turns that produced it.

Three things are pinned here, one per half of that:

* a two-turn session's ``message.complete`` carries the session's totals from
  the same tracker ``/status`` reads, and the two render one figure;
* a resumed session opens on what its stored records say it has already spent,
  adopted into that tracker so the turns that follow add to it;
* a resumed session's window is known before its first call, because the bound
  model's row is read while the banner is being built.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from opendde_harness.agent.loop.factory import build_agent_loop
from opendde_harness.providers import messages as msg
from opendde_harness.session.manager import SessionManager
from opendde_harness.token_wise.usage_tracker import UsageTracker
from opendde_harness.tui_rpc.methods import slash_routing
from opendde_harness.tui_rpc.methods.session import session_resume
from opendde_harness.tui_rpc.spine import build_tui
from tests._gate import (
    MODEL,
    Echo,
    bind,
    declare,
    gate_config,
    request,
    requires_service,
    service,  # noqa: F401  -- the fixture the gate shares
)

KEY = "tui:costed"

#: A published price for the scripted model, which pi carries none for: its
#: catalogue row states zero on every rate, and zero is "nobody knows" rather
#: than free (``PiModelProvider.list_rates``). Stated per million, as pi states
#: it, so the provider's own arithmetic is the one under test.
FAUX_ROW: dict[str, Any] = {
    "contextWindow": 128_000,
    "maxTokens": 16_384,
    "cost": {"input": 3.0, "output": 15.0, "cacheRead": 0.3, "cacheWrite": 3.75},
}


class Recorder:
    """A ``SubscriptionEmitter`` stand-in: every event, in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, session_key: str, event: dict[str, Any]) -> None:
        self.events.append((session_key, event))

    def completions(self, session_key: str) -> list[dict[str, Any]]:
        return [
            event["payload"]["usage"]
            for key, event in self.events
            if key == session_key and event["type"] == "message.complete"
        ]


def _write_home_config(config) -> None:
    """The config the RPC handler reads for itself.

    ``session.resume`` calls ``load_config()`` with no path, which resolves
    under the temporary ``HOME`` the suite installs. Writing this session's own
    config there is what makes the handler read the workspace the session is in.
    """
    path = Path.home() / ".opendde_harness" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.model_dump(mode="json", by_alias=True, exclude_none=True)), encoding="utf-8")


def _priced_loop(service, config, *, priced: bool = True):  # noqa: F811
    """A gate loop whose model has a published price, and no unread row left.

    ``_row_asked`` is set with the row: the provider reads the service's listing
    once and keeps it, so a row installed here is the row every later ask
    answers from -- including the window, which is what the resume test wants
    left *unread*.
    """
    binding = bind(service, config.agents.defaults.model, config)
    if priced:
        binding.provider._row = dict(FAUX_ROW)
        binding.provider._row_asked = True
    loop = build_agent_loop(config, provider=binding.provider)
    loop.tools.register(Echo())
    return binding, loop


def _dollars(report: str) -> str:
    """The figure on ``/status``'s cost line, as it is printed."""
    for line in report.splitlines():
        if line.startswith("Estimated cost:"):
            return line.split()[2]
    raise AssertionError(f"no cost line in:\n{report}")


def _status_report(loop) -> str:
    return slash_routing.session_usage_report(KEY, MODEL, loop.usage_tracker.session_usage(KEY))


@requires_service
async def test_the_footer_and_status_report_one_cost_for_one_session(tmp_path, service):  # noqa: F811
    """Two turns, one figure. The footer's number *is* ``/status``'s number.

    Driven through the TUI's own spine so the assertion is on the event the
    front-end actually reads: ``message.complete``'s usage, whose ``session_*``
    fields the footer shows. Before this the footer showed the turn's cost --
    correct for one turn and wrong for every session of more than one.
    """
    config = gate_config(tmp_path, model=MODEL, providers=declare("faux"), defaults={"maxToolIterations": 3})
    _binding, loop = _priced_loop(service, config)
    emitter = Recorder()
    scheduler, _hub, turn_ids, teardown = build_tui(loop, emitter)

    try:
        for index, prompt in enumerate(("first question", "second question"), start=1):
            turn_ids[KEY] = f"t{index}"
            await asyncio.wait_for(scheduler.submit(request(prompt, KEY)).result(), timeout=60)
    finally:
        await teardown()
        await loop.close_mcp()

    usages = emitter.completions(KEY)
    assert len(usages) == 2, f"one completion per turn, got {len(usages)}"
    first, second = usages
    session_cost = second["session_list_cost_usd"]

    assert session_cost is not None and session_cost > 0, "the model was given a published price"
    # The turn's own cost is the smaller figure, and showing it as the session's
    # is the bug: the first turn's spend is missing from it.
    assert session_cost > second["list_cost_usd"] > 0
    # And the session is exactly its turns, which is also the parity check on the
    # exhaustion wrap-up: both turns here run out of tool iterations and answer
    # with one more call, and that call used to reach the turn's own total and
    # not the session tracker -- so the footer and ``/status`` were a whole call
    # apart on every exhausted turn.
    assert session_cost == pytest.approx(first["list_cost_usd"] + second["list_cost_usd"])
    assert second["session_calls"] >= 2
    assert second["session_input_tokens"] > second["prompt_tokens"], "the session's prompt, not the last call's"

    # And the line the user compares it against says the same thing.
    # Three decimals on both sides: the same figure, rounded the same way.
    assert _dollars(_status_report(loop)) == f"${session_cost:,.3f}"


@requires_service
async def test_resume_reads_the_bound_models_window_before_the_first_call(tmp_path, service):  # noqa: F811
    """``session.resume`` primes the model's row, so the footer opens with a window.

    The window comes from the row the model service reports, and before the
    first request of a process there is no row: a resumed session's footer had
    an occupancy and nothing to divide it by, so it showed no percentage at all
    until a turn landed. The row is not declared here, so the service tier is
    the only one that can answer.
    """
    config = gate_config(tmp_path, model=MODEL, providers=declare("faux"))
    _write_home_config(config)
    binding, loop = _priced_loop(service, config, priced=False)

    manager = SessionManager(tmp_path)
    session = manager.get_or_create(KEY)
    session.record(msg.user_message("what did we decide"))
    manager.save(session)

    assert binding.provider.context_window() is None, "nothing has asked the service about this model yet"

    try:
        result = await session_resume({"session_id": KEY}, agent_loop_factory=lambda: loop)
    finally:
        await loop.close_mcp()

    assert result["session_id"] == KEY
    assert result["info"]["context_window"] == 128_000, "the faux row's own figure, read while answering"
    assert result["info"]["usage"]["context_source"] == "model-service"
    assert binding.provider.context_window() == 128_000, "and it stays read, so the turn asks for nothing extra"


# ---------------------------------------------------------------------------
# The stored history, with no service involved: the records carry the figures.
# ---------------------------------------------------------------------------


def _assistant(*, input_tokens: int, output_tokens: int, cost: float) -> dict[str, Any]:
    """One assistant record as the model service wrote it: a pi message whose
    usage carries the call's tokens and what pi priced it at."""
    record = msg.assistant_message("an answer", model="faux/echo", provider="faux")
    record["usage"] = {
        "input": input_tokens,
        "output": output_tokens,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": input_tokens + output_tokens,
        "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": cost},
    }
    return record


class _StoredLoop:
    """The loop ``session.resume`` needs to answer about a stored session.

    A real one would need a provider built against a service; what the handler
    asks of it is the session manager, the window and the usage tracker, and the
    tracker is the thing under test -- it is the same class the loop installs.
    """

    def __init__(self, workspace: Path) -> None:
        self.sessions = SessionManager(workspace)
        self.usage_tracker = UsageTracker(persist=False)
        self.context_engine = None

    def session_model(self, _session_key: str) -> str:
        return MODEL

    def binding_for_session(self, _session_key: str):
        return None

    def resolve_window(self, _model: str | None = None, _binding=None):
        from opendde_harness.providers.rates import SOURCE_DECLARED, Resolved

        return Resolved(128_000, SOURCE_DECLARED)

    # The rest of what the bundle asks a loop for; none of it is under test
    # here, and a real loop would need a provider built to answer.
    tools = SimpleNamespace(tool_names=[])
    context = SimpleNamespace(skills=SimpleNamespace(list_skills=lambda **_kwargs: []))


async def test_a_resumed_session_opens_on_what_its_records_say_it_has_spent(tmp_path):
    """Three stored calls, and the session opens on their sum rather than on zero.

    The banner's figures and ``/status``'s are one accounting: the handler seeds
    the tracker with the history, so the report ``/status`` builds from that
    tracker states the same total the banner just returned -- and a turn run
    afterwards adds to it instead of starting a second sum.
    """
    loop = _StoredLoop(tmp_path)
    session = loop.sessions.get_or_create(KEY)
    session.record(msg.user_message("the question"))
    for tokens in (1_000, 2_000, 3_000):
        session.record(_assistant(input_tokens=tokens, output_tokens=100, cost=0.01 * (tokens / 1_000)))
    loop.sessions.save(session)

    result = await session_resume({"session_id": KEY}, agent_loop_factory=lambda: loop)
    usage = result["info"]["usage"]

    assert usage["calls"] == 3, "one per stored record that carries a usage figure"
    assert usage["input"] == 6_000
    assert usage["output"] == 300
    # 0.01 + 0.02 + 0.03, each call priced as it was stored.
    assert usage["cost_usd"] == pytest.approx(0.06)
    assert usage["list_cost_usd"] == pytest.approx(0.06)

    # The tracker now holds the same history, so the line the user compares the
    # footer against opens on the same number.
    assert _dollars(_status_report(loop)) == "$0.060"
    assert "summed over 3 calls" in _status_report(loop)

    # Asked twice -- a ``/model`` switch refreshes the banner -- the history is
    # adopted once. Counting the file a second time doubled the session.
    again = await session_resume({"session_id": KEY}, agent_loop_factory=lambda: loop)
    assert again["info"]["usage"]["cost_usd"] == pytest.approx(0.06)
    assert again["info"]["usage"]["calls"] == 3


async def test_a_session_with_no_stored_usage_still_opens_at_zero(tmp_path):
    """A record that carries no usage figure is not a call, and adds nothing.

    A session written before the model service, or one whose only assistant
    message this harness wrote itself (a recovery prefill), carries zeroes; a
    sum over them would report a session of several calls that moved no tokens.
    """
    loop = _StoredLoop(tmp_path)
    session = loop.sessions.get_or_create(KEY)
    session.record(msg.user_message("the question"))
    session.record(msg.assistant_message("a prefill nobody was billed for"))
    loop.sessions.save(session)

    usage = (await session_resume({"session_id": KEY}, agent_loop_factory=lambda: loop))["info"]["usage"]

    assert (usage["calls"], usage["input"], usage["output"]) == (0, 0, 0)
    assert usage["cost_usd"] == 0.0, "metered and nothing spent, which is not the same as unknown"
    assert usage["list_cost_usd"] is None, "no call was priced, so no published figure is claimed"
