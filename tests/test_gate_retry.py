"""G5 — a failed model call is retried in the user's sight, or not retried at all.

The budget is spent inside the model service by pi's own retry loop, and what
reaches the turn is one ``TurnRetry`` per new attempt -- ``turn.retry`` on the
wire. Two things have to hold at the turn level rather than at the call level: a
transient failure is announced and then answered, and a failure pi calls
deterministic is the turn's answer at once with nothing announced.

``faux-flaky`` reads its prompt as the script: ``fail:<n>`` fails the first n
attempts of every request with wording pi's classifier calls transient, and
``fatal`` fails every attempt with wording it does not.
"""

from __future__ import annotations

from opendde_harness.spine.events import Notice, TurnRetry
from tests._gate import (
    FLAKY,
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

KEY = "tui:retry"


def _config(tmp_path, *, retries: int = 3):
    """The gap budgets are generous on purpose: pi waits between attempts, and a
    short idle budget would end the stream instead of letting it retry."""
    return gate_config(
        tmp_path,
        model=FLAKY,
        providers=declare("faux-flaky"),
        defaults={
            "maxToolIterations": 1,
            "llmRetries": retries,
            "llmFirstTokenTimeout": 60,
            "llmIdleTimeout": 60,
        },
    )


async def test_a_transient_failure_is_announced_once_per_call_and_then_answered(tmp_path, service):  # noqa: F811
    """One announcement per model call, and the answer is the attempt that worked.

    The script fails the first attempt of every request, and a turn that calls a
    tool makes more than one request -- so the count that has to hold is one
    ``turn.retry`` per request, never two for one and never none.
    """
    echo = Echo()
    loop = loop_for(service, _config(tmp_path), tools=[echo])
    events = Events()

    try:
        outcome = await loop.run_turn(request("fail:1 please", KEY), events, no_injections, stream=True)
    finally:
        await loop.close_mcp()

    retries = events.of(TurnRetry)
    assert service.streams > 0
    assert len(retries) == service.streams, "one announcement per model call: not swallowed, not doubled"
    assert {(event.attempt, event.total) for event in retries} == {(2, 4)}, "attempt two of a budget of three retries"
    assert all(event.reason == "503 service unavailable" for event in retries), "pi's own wording travels"
    assert all(event.discard is False for event in retries), "nothing had been shown when the attempt failed"
    assert all("503" not in (notice.detail or "") for notice in events.of(Notice)), (
        "a retried failure is announced as a retry, not also reported as a problem"
    )

    # The turn's answer is the successful attempt's: the tool the model asked
    # for ran on the user's own words, and the reply is the model's text rather
    # than a report about a failure.
    assert echo.seen and echo.seen[0].endswith("fail:1 please")
    assert outcome.text and outcome.text.startswith("Echo: ")
    assert "503" not in outcome.text, "the discarded attempt's failure is not the answer"


async def test_a_deterministic_failure_is_not_retried_and_is_the_turns_one_error(tmp_path, service):  # noqa: F811
    """What separates this from the retried case is pi's verdict, not the budget:
    the same budget is configured and none of it is spent."""
    loop = loop_for(service, _config(tmp_path), tools=[Echo()])
    events = Events()

    try:
        outcome = await loop.run_turn(request("fatal, do not repeat this", KEY), events, no_injections, stream=True)
    finally:
        await loop.close_mcp()

    assert events.of(TurnRetry) == [], "a deterministic failure is nobody's second attempt"
    assert service.streams == 1, "and the turn made one call"
    assert outcome.text and "malformed" in outcome.text, "the failure is what the turn answers"
    assert outcome.text.count("malformed") == 1, "said once"


async def test_a_budget_of_zero_is_one_attempt_and_no_announcement(tmp_path, service):  # noqa: F811
    """The same transient failure; only the budget differs."""
    loop = loop_for(service, _config(tmp_path, retries=0), tools=[Echo()])
    events = Events()

    try:
        outcome = await loop.run_turn(request("fail:1 please", KEY), events, no_injections, stream=True)
    finally:
        await loop.close_mcp()

    assert events.of(TurnRetry) == [], "nothing was left to spend, so nothing was announced"
    assert service.streams == 1
    assert outcome.text and "503 service unavailable" in outcome.text
