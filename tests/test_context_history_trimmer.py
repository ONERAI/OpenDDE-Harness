"""The deterministic history-selection policy, clause by clause.

Each test here names one sentence of the policy in
``context_engine/history_trimmer.py``. The token estimator is replaced by a
message counter so the arithmetic is exact and the assertions are about the
policy rather than about a tokenizer.
"""

from __future__ import annotations

import pytest

from opendde_harness.context_engine.excerpt import TOOL_BODY_PLACEHOLDER
from opendde_harness.context_engine.history_trimmer import ContextBudgetError, HistoryTrimmer
from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import COMPACTION_KEY, LLMProvider
from tests import _messages as build

PER_MESSAGE = 10
#: system + user, at PER_MESSAGE each: the fixed cost every budget starts from.
FIXED = 2 * PER_MESSAGE

MESSAGES = [
    build.user("start"),
    build.assistant(calls=[("call_1", "read", {})]),
    build.tool_result("call_1", "read", "file contents"),
    build.assistant("done"),
    build.user("next"),
    build.assistant("ok"),
]


class _Replays(LLMProvider):
    """A provider whose own backend made the marker it is shown."""

    supports_compaction = True

    def replays_compaction(self, marker, model):
        return marker.get("provider") == "fake" and marker.get("model") == model

    async def chat(self, messages, tools=None, model=None, **kwargs):
        raise AssertionError("selection makes no model call")

    def get_default_model(self) -> str:
        return "fake/model"


def _marker(model: str = "fake/model", provider: str = "fake") -> dict:
    return build.marker({"provider": provider, "model": model, "items": [{"type": "opaque"}]})


def _turns(n: int, *, words: int = 1) -> list[dict]:
    """``n`` plain user/assistant pairs."""
    out: list[dict] = []
    for i in range(n):
        out.append(build.user(f"question {i} " * words))
        out.append(build.assistant(f"answer {i} " * words))
    return out


def _tool_turn(i: int) -> list[dict]:
    """One user turn whose assistant calls a tool and reads the result."""
    return [
        build.user(f"do {i}"),
        build.assistant(calls=[(f"c{i}", "read", {})]),
        build.tool_result(f"c{i}", "read", f"body {i}"),
        build.assistant(f"done {i}"),
    ]


class _Counter:
    def __init__(self) -> None:
        self.calls = 0
        self.visits = 0


@pytest.fixture
def counts(monkeypatch) -> _Counter:
    """Price every message at PER_MESSAGE and count what the estimator sees."""
    import opendde_harness.context_engine.history_trimmer as module

    counter = _Counter()

    def chain(provider, model, messages, tools):
        counter.calls += 1
        counter.visits += len(messages)
        return len(messages) * PER_MESSAGE, "count"

    def one(message):
        counter.visits += 1
        return PER_MESSAGE

    monkeypatch.setattr(module, "estimate_prompt_tokens_chain", chain)
    monkeypatch.setattr(module, "estimate_message_tokens", one)
    return counter


def _selector(window: int | None, *, protect: int = 3, provider=None) -> HistoryTrimmer:
    return HistoryTrimmer(provider or _Replays(), "fake/model", list, window, protect_first_n=protect)


def _build(history: list[dict]) -> list[dict]:
    return [build.system("sys"), *history, build.user("now")]


def _select(trimmer: HistoryTrimmer, messages: list[dict], reserved_output: int = 0):
    return trimmer.select(
        session_messages=messages,
        reserved_output=reserved_output,
        build_messages=_build,
    )


# ---------------------------------------------------------------------------
# Clause 1 — budget against the rendered prefix, the tools, the user message
# and the output reservation
# ---------------------------------------------------------------------------


def test_the_budget_is_the_window_less_the_reservation_and_the_fixed_prompt(counts) -> None:
    messages = _turns(10)
    trimmer = _selector(300)

    _, outcome = _select(trimmer, messages, reserved_output=100)

    # 300 window - 100 reply - 20 system+user = 180, at 10 a message.
    assert outcome.max_prompt_tokens == 200
    assert len(outcome.included_ids) == (200 - FIXED) // PER_MESSAGE
    assert outcome.ok and outcome.estimated_tokens <= outcome.max_prompt_tokens


def test_a_bigger_reservation_leaves_less_history(counts) -> None:
    messages = _turns(10)

    _, small = _select(_selector(300), messages, reserved_output=100)
    _, large = _select(_selector(300), messages, reserved_output=160)

    assert len(large.included_ids) < len(small.included_ids)
    assert (len(small.included_ids) - len(large.included_ids)) * PER_MESSAGE == 60


# ---------------------------------------------------------------------------
# Clause 2 — the protected head is the first N user messages
# ---------------------------------------------------------------------------


def test_the_first_three_user_messages_are_never_dropped(counts) -> None:
    messages = _turns(12)
    trimmer = _selector(120)  # room for 10 messages of history

    _, outcome = _select(trimmer, messages)

    heads = [mid for mid in range(len(messages)) if messages[mid].get("role") == "user"][:3]
    assert set(heads) <= set(outcome.included_ids)
    assert len(outcome.included_ids) == 10


def test_protect_first_n_zero_protects_nothing(counts) -> None:
    messages = _turns(12)

    _, outcome = _select(_selector(120, protect=0), messages)

    assert 0 not in outcome.included_ids, "with no protected head the oldest turn is the first to go"
    assert len(outcome.included_ids) == 10


def test_a_tool_result_answering_a_protected_call_is_protected_with_it(counts) -> None:
    """Protecting the assistant call but not its result used to drop both: the
    result was droppable and its exchange took the call along."""
    messages = [*_tool_turn(0), *_turns(8)]
    trimmer = _selector(90, protect=1)

    _, outcome = _select(trimmer, messages)

    assert HistoryTrimmer.structural_errors(outcome.history) == []
    assert 0 in outcome.included_ids


# ---------------------------------------------------------------------------
# Clause 3 — the newest completed exchange and its enclosing user message
# ---------------------------------------------------------------------------


def test_the_newest_exchange_and_the_user_message_it_answers_are_required(counts) -> None:
    messages = [*_turns(9), *_tool_turn(9)]
    # Room for the protected head and little else.
    trimmer = _selector(FIXED + 8 * PER_MESSAGE)

    _, outcome = _select(trimmer, messages)

    newest = list(range(len(messages) - 4, len(messages)))
    assert set(newest) <= set(outcome.included_ids), "the newest exchange rode out whole"
    assert HistoryTrimmer.structural_errors(outcome.history) == []


def test_an_unfinished_newest_exchange_is_not_the_required_one(counts) -> None:
    """An interrupted turn leaves a call with no result. Sending it is refused
    outright, so the requirement falls back to the newest complete exchange."""
    orphaned = build.assistant(calls=[("gone", "read", {})])
    messages = [*_turns(3), *_tool_turn(3), build.user("again"), orphaned]
    trimmer = _selector(200)

    _, outcome = _select(trimmer, messages)

    assert HistoryTrimmer.structural_errors(outcome.history) == []
    assert len(messages) - 1 not in outcome.included_ids


# ---------------------------------------------------------------------------
# Clause 4 — fill with the newest complete exchanges, chronologically
# ---------------------------------------------------------------------------


def test_the_fill_takes_the_newest_and_hands_them_back_in_order(counts) -> None:
    messages = _turns(10)
    trimmer = _selector(FIXED + 6 * PER_MESSAGE, protect=0)

    _, outcome = _select(trimmer, messages)

    assert outcome.included_ids == sorted(outcome.included_ids), "chronological out"
    assert outcome.included_ids == list(range(len(messages) - 6, len(messages))), "newest in"


# ---------------------------------------------------------------------------
# Clause 5 — a compatible marker is the frame; clause 6 — an incompatible one
# is no boundary
# ---------------------------------------------------------------------------


def test_a_compatible_marker_is_kept_and_what_it_stands_for_is_not_costed(counts) -> None:
    before = _turns(20)
    messages = [*before, _marker(), *_turns(2)]
    trimmer = _selector(FIXED + 6 * PER_MESSAGE)

    _, outcome = _select(trimmer, messages)

    assert outcome.included_ids == list(range(len(before), len(messages))), "the marker and its tail, nothing before"
    assert any(COMPACTION_KEY in m for m in outcome.history)
    assert outcome.ok
    # Nothing before the marker was priced: 5 candidates, not 25.
    assert counts.visits < len(before)


def test_an_incompatible_marker_establishes_no_boundary(counts) -> None:
    before = _turns(3)
    messages = [*before, _marker(model="other/model"), *_turns(1)]
    trimmer = _selector(500)

    _, outcome = _select(trimmer, messages)

    assert outcome.included_ids == list(range(len(messages))), "a marker this model does not replay is just a message"


def test_the_marker_survives_pressure_it_cannot_relieve(counts) -> None:
    """Deleting the marker saves its own placeholder and puts the whole history
    it stands for back on the wire."""
    messages = [_marker(), *_turns(6)]
    trimmer = _selector(FIXED + 4 * PER_MESSAGE)

    _, outcome = _select(trimmer, messages)

    assert 0 in outcome.included_ids
    assert any(COMPACTION_KEY in m for m in outcome.history)


# ---------------------------------------------------------------------------
# Clause 7 — excerpt the oldest tool bodies before dropping a turn
# ---------------------------------------------------------------------------


def test_the_oldest_tool_bodies_are_excerpted_before_a_turn_is_dropped() -> None:
    """Real token counts here: the point is that a long body is what gives, and
    a body is only worth excerpting because it is long."""
    messages: list[dict] = []
    for i in range(6):
        turn = _tool_turn(i)
        turn[2]["content"] = [msg.text_block(f"body {i} " * 400)]
        messages += turn
    trimmer = _selector(4_000)

    _, outcome = _select(trimmer, messages)

    assert outcome.excerpted_ids, "the oldest bodies were excerpted"
    excerpted = [m for m in outcome.history if msg.text_of(m) == TOOL_BODY_PLACEHOLDER]
    assert excerpted, "and the placeholder is what the model is shown"
    # The newest three results keep their bodies; the exchange is still whole.
    assert sum(1 for m in outcome.history if msg.is_tool_result(m)) > len(excerpted)
    assert HistoryTrimmer.structural_errors(outcome.history) == []


def test_excerpting_never_touches_the_session_or_a_call() -> None:
    messages: list[dict] = []
    for i in range(6):
        turn = _tool_turn(i)
        turn[2]["content"] = [msg.text_block(f"body {i} " * 400)]
        messages += turn
    originals = [dict(m) for m in messages]
    trimmer = _selector(4_000)

    _, outcome = _select(trimmer, messages)

    assert outcome.excerpted_ids, "there was something to excerpt"
    assert messages == originals, "and the log still holds every original body"
    for message in outcome.history:
        calls = msg.tool_calls_of(message)
        if calls:
            assert TOOL_BODY_PLACEHOLDER not in str(calls), "a call's arguments are never excerpted"
        if not msg.is_tool_result(message):
            assert msg.text_of(message) != TOOL_BODY_PLACEHOLDER


def test_an_excerpt_keeps_the_reference_back_to_its_call() -> None:
    messages: list[dict] = []
    for i in range(6):
        turn = _tool_turn(i)
        turn[2]["content"] = [msg.text_block(f"body {i} " * 400)]
        messages += turn
    trimmer = _selector(4_000)

    _, outcome = _select(trimmer, messages)

    for message in outcome.history:
        if msg.text_of(message) == TOOL_BODY_PLACEHOLDER:
            assert message.get("toolCallId"), "an excerpt still names the call it answers"
            assert message.get("toolName"), "and the tool that produced it"


# ---------------------------------------------------------------------------
# Clause 8 — then drop the oldest unprotected exchanges
# ---------------------------------------------------------------------------


def test_dropping_takes_the_whole_exchange(counts) -> None:
    messages = [*_tool_turn(0), *_tool_turn(1)]
    trimmer = _selector(FIXED + 5 * PER_MESSAGE, protect=0)

    _, outcome = _select(trimmer, messages)

    assert HistoryTrimmer.structural_errors(outcome.history) == []
    # The call and its result leave together, rather than the call being
    # stranded with no result.
    assert 1 not in outcome.included_ids and 2 not in outcome.included_ids


def test_history_still_starts_at_a_user_message(counts) -> None:
    trimmer = _selector(FIXED + 3 * PER_MESSAGE, protect=0)

    _, outcome = _select(trimmer, MESSAGES)

    assert HistoryTrimmer.structural_errors(outcome.history) == []
    assert not outcome.history or outcome.history[0]["role"] == "user"


# ---------------------------------------------------------------------------
# Clause 9 — a mandatory set that cannot fit is an error, not a big request
# ---------------------------------------------------------------------------


def test_a_mandatory_set_that_cannot_fit_raises_instead_of_sending(counts) -> None:
    messages = _turns(8)
    # The protected head (3) plus the newest exchange (2) cannot fit in 2.
    trimmer = _selector(FIXED + 2 * PER_MESSAGE)

    with pytest.raises(ContextBudgetError) as excinfo:
        _select(trimmer, messages)

    message = str(excinfo.value)
    assert "fake/model" in message and "/new" in message
    assert "question" not in message, "an error about size quotes no conversation"


def test_the_protected_head_is_never_dropped_to_fit(counts) -> None:
    """The head is refused-with-an-error territory, not droppable."""
    messages = _turns(8)

    with pytest.raises(ContextBudgetError):
        _select(_selector(FIXED + 4 * PER_MESSAGE), messages)

    # The same budget with no protected head is simply a small window.
    _, outcome = _select(_selector(FIXED + 4 * PER_MESSAGE, protect=0), messages)
    assert len(outcome.included_ids) == 4


# ---------------------------------------------------------------------------
# Clause 10 — an unknown window is not a number
# ---------------------------------------------------------------------------


def test_unknown_window_ships_everything_and_reports_nothing_enforced(monkeypatch) -> None:
    import opendde_harness.context_engine.history_trimmer as module

    monkeypatch.setattr(module, "estimate_prompt_tokens_chain", lambda *a: (10**9, "count"))
    trimmer = _selector(None)

    _, outcome = _select(trimmer, MESSAGES)

    # Nothing to trim against, so nothing is dropped: an invented window used
    # to cut real history here.
    assert outcome.included_ids == list(range(len(MESSAGES)))
    assert outcome.max_prompt_tokens is None
    assert outcome.ok and outcome.over_by == 0 and outcome.warnings == []


# ---------------------------------------------------------------------------
# One pass: bounded estimator work, and one corrective round when the exact
# projection disagrees with the sum of per-message costs
# ---------------------------------------------------------------------------


def test_doubling_the_history_does_not_more_than_double_the_estimator_work(monkeypatch) -> None:
    """The bound, not the clock: selection used to re-estimate once per dropped
    exchange, so 100 -> 200 -> 400 candidates cost 5k -> 20k -> 80k message
    visits by the estimator. Precomputing costs once makes the growth linear."""
    import opendde_harness.context_engine.history_trimmer as module

    def work(exchanges: int) -> tuple[int, int]:
        counter = _Counter()

        def chain(provider, model, messages, tools):
            counter.calls += 1
            counter.visits += len(messages)
            return len(messages) * PER_MESSAGE, "count"

        monkeypatch.setattr(module, "estimate_prompt_tokens_chain", chain)
        monkeypatch.setattr(module, "estimate_message_tokens", lambda m: PER_MESSAGE)
        _select(_selector(FIXED + 20 * PER_MESSAGE, protect=1), _turns(exchanges))
        return counter.visits, counter.calls

    small_visits, small_calls = work(200)
    large_visits, large_calls = work(400)

    assert large_visits <= 2.5 * small_visits, f"{small_visits} -> {large_visits} visits for a doubled history"
    assert large_calls <= small_calls + 1, f"{small_calls} -> {large_calls} estimator calls"


def test_the_exact_projection_corrects_an_underestimate(monkeypatch) -> None:
    """Summing per-message costs is not the same arithmetic as pricing the
    chain. The gap is settled by a bounded number of corrective removals, and
    the request that goes out is under the budget."""
    import opendde_harness.context_engine.history_trimmer as module

    calls = _Counter()

    def chain(provider, model, messages, tools):
        calls.calls += 1
        # 3 tokens per message more than the per-message cost says.
        return len(messages) * (PER_MESSAGE + 3), "count"

    monkeypatch.setattr(module, "estimate_prompt_tokens_chain", chain)
    monkeypatch.setattr(module, "estimate_message_tokens", lambda m: PER_MESSAGE)
    trimmer = _selector(FIXED + 30 * PER_MESSAGE, protect=1)

    _, outcome = _select(trimmer, _turns(12))

    assert outcome.ok, "the request that goes out fits"
    assert calls.calls <= 2 + HistoryTrimmer._MAX_CORRECTIONS, "and the correction is bounded"


# ---------------------------------------------------------------------------
# The oracle: every selection is structurally sound at every budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", [60, 80, 100, 140, 200, 400, 1_000])
def test_every_selection_passes_the_structural_oracle(counts, window) -> None:
    messages = [*_tool_turn(0), *_turns(2), *_tool_turn(1), *_turns(1), *_tool_turn(2)]

    try:
        built, outcome = _select(_selector(window, protect=1), messages)
    except ContextBudgetError:
        return  # refused rather than sent: the other correct answer

    assert HistoryTrimmer.structural_errors(built) == []
    validation = _selector(window, protect=1).validate_candidate(built, 0)
    assert validation["errors"] == []
    assert validation["ok"], validation
    assert outcome.included_ids == sorted(set(outcome.included_ids))
