"""A compaction marker is a boundary, and every hand it passes through knows it.

The provider replaces the whole history before the marker with one opaque item
when it talks to the model that made it. That only works if the marker reaches
the request: the session projection and the selector's both used to drop the
key, and the assistant coalescer merged the marker into the reply before it.
Each of those alone loses the boundary, and the backend then compacts the same
conversation again, every turn.
"""

from __future__ import annotations

import pytest

from opendde_harness.context_engine.assembler import _coalesce_assistant
from opendde_harness.context_engine.history_trimmer import HistoryTrimmer
from opendde_harness.providers.base import COMPACTION_KEY, LLMProvider, compaction_boundary, wire_history
from opendde_harness.providers.messages import text_of
from opendde_harness.session.manager import Session
from opendde_harness.utils.helpers import count_text_tokens, estimate_message_tokens
from tests import _messages as build
from tests._config import config as build_config
from tests._config import declared

MODEL = "fake/model"


def _projected(messages: list[dict]) -> list[dict]:
    """The one projection between the log and the request."""
    return HistoryTrimmer.history_from_ids(messages, HistoryTrimmer.canonical_ids(messages, list(range(len(messages)))))


def _marker(model: str = MODEL) -> dict:
    return build.marker({"provider": "fake", "model": model, "items": [{"type": "opaque"}]})


MESSAGES = [
    build.user("an early question"),
    build.assistant("an early answer"),
    _marker(),
    build.user("and now?"),
]


class _Replays(LLMProvider):
    """A provider whose backend made the marker."""

    supports_compaction = True

    def replays_compaction(self, marker, model):
        return marker.get("provider") == "fake" and marker.get("model") == model

    async def chat(self, messages, tools=None, model=None, **kwargs):
        raise AssertionError("no call is made here")

    def get_default_model(self) -> str:
        return MODEL


def test_every_projection_between_the_log_and_the_request_keeps_the_marker() -> None:
    session = Session(key="cli:t")
    for message in MESSAGES:
        session.record(dict(message))

    projections = {
        "session": session.get_history(),
        "selector": _projected(MESSAGES),
    }
    for name, history in projections.items():
        marker = next((m for m in history if COMPACTION_KEY in m), None)
        assert marker is not None, f"the {name} projection dropped the boundary"
        assert marker[COMPACTION_KEY]["items"] == [{"type": "opaque"}], f"the {name} projection emptied it"


def test_a_marker_is_never_merged_into_an_adjacent_assistant_message() -> None:
    """It is a plain assistant message with str content and no tool calls, so it
    matched the merge on both sides; merging kept the other message's fields and
    the boundary was gone before the request was built."""
    around = [build.user("q"), build.assistant("an answer"), _marker(), build.assistant("a later answer")]

    coalesced = _coalesce_assistant(around)

    assert [text_of(m) for m in coalesced] == ["q", "an answer", "[compacted]", "a later answer"]
    assert COMPACTION_KEY in coalesced[2]


def test_two_ordinary_adjacent_assistants_still_merge() -> None:
    merged = _coalesce_assistant([build.assistant("on it"), build.assistant("done")])

    assert [text_of(m) for m in merged] == ["on it\n\ndone"]


def test_the_budget_counts_the_marker_and_what_follows_it() -> None:
    """Everything before the marker is replaced on the wire, so counting it
    trimmed a session to fit a budget it was nowhere near."""
    provider = _Replays()
    prompt = [build.system("rules"), *MESSAGES]

    sent = wire_history(prompt, provider, MODEL)

    assert sent[0]["content"] == "rules", "the instructions are not part of what a marker replaces"
    assert [text_of(m) for m in sent[1:]] == ["", "and now?"]
    assert sent[1]["role"] == "user", "the marker is priced as the clear text it stands in for"


def test_the_budget_counts_the_clear_user_text_the_marker_carries() -> None:
    """Measured before this: the marker was priced at its placeholder, about
    nineteen tokens, while the request sent the twenty thousand tokens of the
    user's own words that ride in clear beside the opaque summary."""
    provider = _Replays()
    words = "the interface residue numbering " * 400
    carried = _marker()
    carried[COMPACTION_KEY]["items"] = [
        {"role": "user", "content": [{"type": "input_text", "text": words}]},
        {"type": "compaction", "encrypted_content": "x" * 100_000},
    ]

    priced = wire_history([*MESSAGES[:2], carried, MESSAGES[-1]], provider, MODEL)

    assert estimate_message_tokens(priced[0]) == count_text_tokens(words)
    assert estimate_message_tokens(priced[0]) < 10_000, "and the opaque bytes are still not priced"


@pytest.mark.parametrize("where", ["before", "after"])
def test_only_the_last_system_message_is_priced(where) -> None:
    """The converter has one ``instructions`` field and the last writer wins
    it, wherever it sits. An obsolete system message after the marker was
    priced in full and sent not at all."""
    provider = _Replays()
    systems = [build.system("obsolete " * 500), build.system("the rules in force")]
    messages = [*systems, *MESSAGES] if where == "before" else [*MESSAGES[:3], *systems, MESSAGES[-1]]

    priced = wire_history(messages, provider, MODEL)

    assert [m for m in priced if m.get("role") == "system"] == [systems[-1]]
    assert estimate_message_tokens(priced[0]) < 100, "the obsolete one is not counted"


def test_another_model_reads_the_whole_local_history() -> None:
    provider = _Replays()

    assert wire_history(list(MESSAGES), provider, "fake/other") == MESSAGES


def test_a_provider_that_made_no_marker_reads_the_whole_local_history() -> None:
    assert wire_history(list(MESSAGES), None, MODEL) == MESSAGES


def test_the_latest_marker_is_the_boundary() -> None:
    """Two compactions of the same session: the older summary is inside the
    newer one, so the request starts at the newer."""
    provider = _Replays()
    later = _marker()
    later[COMPACTION_KEY]["items"] = [{"type": "opaque", "id": "second"}]
    messages = [*MESSAGES, build.assistant("more"), later, build.user("again")]

    sent = wire_history(messages, provider, MODEL)

    assert [text_of(m) for m in sent] == ["", "again"]
    assert compaction_boundary(messages, provider, MODEL) == len(messages) - 2


def test_budget_pressure_never_deletes_the_boundary() -> None:
    """Measured before this: with a protected head and pressure after the
    marker, the marker was simply the first droppable message -- deleting it
    saved its own handful of tokens and put the whole history it stood for back
    on the wire."""
    head = [
        builder("PREBOUNDARY " + " old" * 1000)
        for builder in (build.user, build.assistant, build.user, build.assistant)
    ]
    call = build.assistant(calls=[("c1", "blob", {})])
    result = build.tool_result("c1", "blob", " tool" * 2000)
    later = [build.user("later"), call, result, build.assistant("done")]
    history = [*head, _marker(), *later, build.user("latest"), build.assistant("ok")]
    trimmer = HistoryTrimmer(_Replays(), MODEL, list, 1000)

    built, outcome = trimmer.select(session_messages=history, reserved_output=0, build_messages=list)

    assert any(COMPACTION_KEY in m for m in built), "the boundary survived the pressure"
    assert outcome.dropped_ids, "and the pressure was real -- something was dropped"
    assert not any("PREBOUNDARY" in str(m.get("content")) for m in wire_history(built, _Replays(), MODEL))
    assert trimmer.structural_errors(built) == []


def test_the_marker_is_the_frame_and_never_a_candidate() -> None:
    """A selection without it sends the history it replaced instead, at full
    size. It is kept whatever the budget says, and the messages it stands for
    are not candidates at all."""
    trimmer = HistoryTrimmer(_Replays(), MODEL, list, 1000)

    built, outcome = trimmer.select(session_messages=MESSAGES, reserved_output=0, build_messages=list)

    assert any(COMPACTION_KEY in m for m in built)
    assert outcome.included_ids == [2, 3], "the marker and what follows it, nothing before"


def test_a_selection_starting_at_the_marker_keeps_it() -> None:
    """Alignment used to skip forward to the first user message, which deleted
    a boundary the selection had kept."""
    assert HistoryTrimmer.canonical_ids([_marker(), build.user("after")], [0, 1]) == [0, 1]


def test_a_capped_history_never_starts_past_the_marker() -> None:
    """A cap that lands after the marker ships the tail of a conversation with
    neither the summary nor the history it stood in for."""
    session = Session(key="cli:t")
    for message in [*MESSAGES, build.assistant("later"), build.user("latest")]:
        session.record(dict(message))

    capped = session.get_history(max_messages=2)

    assert any(COMPACTION_KEY in m for m in capped)
    assert [text_of(m) for m in capped][:1] == ["[compacted]"]
    assert len(session.get_history(max_messages=0)) == len(session.messages), "0 is no cap, not an empty history"


def test_the_projection_keeps_a_leading_marker() -> None:
    assert _projected(MESSAGES[2:])[0][COMPACTION_KEY]["items"] == [{"type": "opaque"}]


# ---------------------------------------------------------------------------
# A marker at index zero. The capped session slice and the curator projection
# both hand back a list that starts at the marker, so zero is an answer, not
# the absence of one -- a sentinel that conflated the two left the boundary
# unprotected, unreinserted and uncosted in exactly those lists.
# ---------------------------------------------------------------------------

LEADING = [_marker(), build.user("next")]


def test_no_boundary_and_a_boundary_at_zero_are_different_answers() -> None:
    provider = _Replays()

    assert compaction_boundary(LEADING, provider, MODEL) == 0
    assert compaction_boundary([build.user("hi")], provider, MODEL) is None


def test_a_leading_marker_is_costed_as_the_text_it_carries() -> None:
    provider = _Replays()
    words = "the interface residue numbering " * 400
    leading = _marker()
    leading[COMPACTION_KEY]["items"] = [{"role": "user", "content": [{"type": "input_text", "text": words}]}]

    priced = wire_history([leading, build.user("next")], provider, MODEL)

    assert estimate_message_tokens(priced[0]) == count_text_tokens(words)


def test_a_leading_marker_is_kept_with_the_tail_after_it() -> None:
    trimmer = HistoryTrimmer(_Replays(), MODEL, list, 1000)

    built, _ = trimmer.select(session_messages=LEADING, reserved_output=0, build_messages=list)

    assert [text_of(m) for m in built] == ["[compacted]", "next"]


def test_a_leading_marker_survives_pressure_it_cannot_relieve() -> None:
    """Deleting it saved a placeholder and put the whole prefix back."""
    trimmer = HistoryTrimmer(_Replays(), MODEL, list, 1000)
    pressure = [_marker(), build.user(" long" * 2000)]

    built, outcome = trimmer.select(session_messages=pressure, reserved_output=0, build_messages=list)

    assert any(COMPACTION_KEY in m for m in built)
    assert outcome.included_ids == [0], "the boundary stayed; the message it could not fit is what went"


# ---------------------------------------------------------------------------
# A tool call before the marker answered after it
# ---------------------------------------------------------------------------

CALL = build.assistant(calls=[("cross", "blob", {})])
RESULT = build.tool_result("cross", "blob", "RESULT")
CROSSING = [build.user("before"), CALL, _marker(), RESULT, build.user("after")]


def test_aligning_to_a_marker_never_leaves_an_orphan_tool_result() -> None:
    """The call is closed over first and then cut away by the alignment; the
    result left behind is refused outright ("must be a response to a preceding
    message with tool_calls"). Past the marker the call is inside the summary,
    so the result goes with it."""
    assert HistoryTrimmer.structural_errors(CROSSING) == [], "the local history is sound; the cut is what breaks it"

    ids = HistoryTrimmer.canonical_ids(CROSSING, [2, 3, 4])

    assert HistoryTrimmer.structural_errors(HistoryTrimmer.history_from_ids(CROSSING, ids)) == []
    assert [CROSSING[i]["role"] for i in ids] == ["assistant", "user"]


def test_a_selection_that_keeps_the_call_still_sends_no_orphan() -> None:
    """The other shape. When the cut keeps the earlier user message the call
    survives selection, locally paired and valid, and the marker's replay
    removes it at conversion instead. Every projection reaches the same wire,
    because the wire is where the marker's reach is known."""
    from opendde_harness.providers.pi_context import to_context
    from opendde_harness.providers.pi_provider import PiModelProvider

    marker = _marker()
    marker[COMPACTION_KEY]["provider"] = "openai-codex"
    marker[COMPACTION_KEY]["model"] = MODEL
    crossing = [build.user("before"), CALL, marker, RESULT, build.user("after")]
    session = Session(key="cli:t")
    for message in crossing:
        session.record(dict(message))

    # No service: the split and the conversion are pure, and they are the whole
    # of what the marker's reach decides.
    provider = PiModelProvider(None, "openai-codex", MODEL.split("/", 1)[1], stored_model=MODEL)
    projections = {
        "selector": _projected(crossing),
        "session": session.get_history(),
    }
    for name, projection in projections.items():
        assert HistoryTrimmer.structural_errors(projection) == [], f"{name} is sound locally"
        replay, sendable = provider._split_at_marker(projection, MODEL)
        assert replay is not None, f"{name} still carries the marker the request replays"
        wire = to_context(sendable, None, provider="openai-codex", model=MODEL.split("/", 1)[1])
        assert "toolCall" not in str(wire) and "toolResult" not in str(wire), name
        assert "RESULT" not in str(wire), name


def test_the_session_and_selector_projections_drop_the_same_orphan() -> None:
    session = Session(key="cli:t")
    for message in CROSSING:
        session.record(dict(message))

    for projection in (session.get_history(max_messages=2), _projected(CROSSING[2:])):
        assert HistoryTrimmer.structural_errors(projection) == []
        assert [m["role"] for m in projection] == ["assistant", "user"]


@pytest.mark.parametrize(
    ("model", "providers", "backend"),
    [
        ("openai-codex/gpt-5.6-luna", None, "openai-codex"),
        ("deepseek/deepseek-chat", None, ""),
        # A provider this config declares: whatever it fronts, pi has no
        # Responses route for it, so nothing there compacts server-side.
        ("custom/review-model", declared("custom", models=["review-model"]), ""),
    ],
)
def test_the_compaction_backend_is_named_before_any_provider_is_built(model, providers, backend) -> None:
    """The history selector asks whether a stored marker will be replayed while
    it budgets the first prompt, before anything has been built, and both
    answers are destructive when guessed: a no deletes a boundary the backend
    would have replayed, a yes lets a stale marker zero the budget of a backend
    that will send the whole history.

    The name is pi's own id, which is the only one a marker is ever written or
    read under, and it is read off the model id's own prefix -- nothing else
    names the provider.
    """
    from opendde_harness.cli._helpers import declared_compaction_provider

    config = build_config(providers, model=model)

    assert declared_compaction_provider(config) == backend


def test_the_selector_leaves_a_compacted_session_alone() -> None:
    """A window the local history is far over, and a marker that means the
    request carries almost none of it."""
    long_history = [
        build.user("question " * 400),
        build.assistant("answer " * 400),
        _marker(),
        build.user("and now?"),
    ]
    trimmer = HistoryTrimmer(_Replays(), MODEL, list, 500)

    _, outcome = trimmer.select(session_messages=long_history, reserved_output=0, build_messages=list)

    assert outcome.dropped_ids == [], "nothing after the marker was dropped to fit"
    assert outcome.included_ids == [2, 3], "and nothing before it was ever a candidate"


def test_the_turns_accounting_says_the_window_was_compacted() -> None:
    """The last call's reported context describes the prompt the compaction
    just replaced. A status line that keeps showing it states a percentage of
    a conversation that no longer exists, so the turn's own accounting carries
    the boundary out to whoever is displaying it."""
    import asyncio
    from datetime import datetime
    from types import SimpleNamespace

    from opendde_harness.agent.loop.main import AgentLoop, FinalCall, LoopOutcome
    from opendde_harness.config.schema import ProvidersConfig

    session = Session(key="tui:t")
    marker = _marker()
    compacted = SimpleNamespace(
        context_config=SimpleNamespace(server_compact_ratio=0.8),
        provider=SimpleNamespace(
            supports_compaction=True,
            compact=lambda *a, **kw: _answer((dict(marker), {})),
            list_rates=lambda: None,
        ),
        usage_tracker=SimpleNamespace(record_snapshot=lambda _snapshot: None),
        sessions=SimpleNamespace(save=lambda _session: None),
        _now_fn=lambda: datetime(2026, 9, 11),
        _build_usage_snapshot=AgentLoop._build_usage_snapshot,
        # The snapshot asks the providers map whether this model is billed by
        # subscription, because a plan is the entry's `login` and nothing about
        # the model says so. An empty map is "no plan", which is what a metered
        # key is, and is the case these tests are about.
        settings=SimpleNamespace(providers=ProvidersConfig()),
    )
    outcome = LoopOutcome(
        usage={"prompt_tokens": 190_000, "context_used": 190_000, "context_max": 200_000},
        final_call=FinalCall(model=MODEL, prompt_tokens=190_000, window=200_000),
    )

    asyncio.run(AgentLoop._maybe_compact_remote(compacted, session, outcome))

    assert outcome.usage["context_compacted"] is True
    # And an ordinary turn, whose prompt is nowhere near the trigger, says
    # nothing at all: the key is the boundary, not a field every turn carries.
    quiet = LoopOutcome(
        usage={"prompt_tokens": 10, "context_used": 10},
        final_call=FinalCall(model=MODEL, prompt_tokens=10, window=200_000),
    )

    asyncio.run(AgentLoop._maybe_compact_remote(compacted, Session(key="tui:q"), quiet))

    assert "context_compacted" not in quiet.usage


def test_the_compaction_call_is_charged_to_the_turn_that_caused_it() -> None:
    """Compaction is a billed call on the full prompt, made because of this
    turn and before its accounting reaches the wire.

    The session tracker behind ``/status`` has always counted it. The turn's
    own total did not, so the footer and ``/status`` disagreed about the same
    call, and the disagreement grew with every compaction.
    """
    import asyncio
    from datetime import datetime
    from types import SimpleNamespace

    from opendde_harness.agent.loop.main import AgentLoop, FinalCall, LoopOutcome
    from opendde_harness.config.schema import ProvidersConfig
    from opendde_harness.providers.rates import ListRates

    rates = ListRates(input=1e-5, output=1e-4, cache_read=0.0, cache_write=0.0)

    marker = _marker()
    spent = {"prompt_tokens": 190_000, "completion_tokens": 2_000, "total_tokens": 192_000}
    compacted = SimpleNamespace(
        context_config=SimpleNamespace(server_compact_ratio=0.8),
        provider=SimpleNamespace(
            supports_compaction=True,
            compact=lambda *a, **kw: _answer((dict(marker), dict(spent))),
            # The rates the model layer reports for the model that answered:
            # the compaction is priced by the same provider its call went to.
            list_rates=lambda: rates,
        ),
        usage_tracker=SimpleNamespace(record_snapshot=lambda _snapshot: None),
        sessions=SimpleNamespace(save=lambda _session: None),
        _now_fn=lambda: datetime(2026, 9, 11),
        _build_usage_snapshot=AgentLoop._build_usage_snapshot,
        # The snapshot asks the providers map whether this model is billed by
        # subscription, because a plan is the entry's `login` and nothing about
        # the model says so. An empty map is "no plan", which is what a metered
        # key is, and is the case these tests are about.
        settings=SimpleNamespace(providers=ProvidersConfig()),
    )
    outcome = LoopOutcome(
        usage={"prompt_tokens": 190_000, "context_used": 190_000, "context_max": 200_000, "list_cost_usd": 0.5},
        final_call=FinalCall(model=MODEL, prompt_tokens=190_000, window=200_000),
    )

    asyncio.run(AgentLoop._maybe_compact_remote(compacted, Session(key="tui:t"), outcome, session_key="tui:t"))

    assert outcome.usage["list_cost_usd"] == pytest.approx(0.5 + rates.cost(190_000, 2_000, 0, 0))


def test_a_compaction_the_vendor_reported_nothing_for_adds_nothing() -> None:
    """Zero would put a price on the wire where the turn had stated none,
    which beside a plan reads as free rather than as covered."""
    import asyncio
    from datetime import datetime
    from types import SimpleNamespace

    from opendde_harness.agent.loop.main import AgentLoop, FinalCall, LoopOutcome
    from opendde_harness.config.schema import ProvidersConfig

    compacted = SimpleNamespace(
        context_config=SimpleNamespace(server_compact_ratio=0.8),
        provider=SimpleNamespace(
            supports_compaction=True,
            compact=lambda *a, **kw: _answer((dict(_marker()), {})),
            list_rates=lambda: None,
        ),
        usage_tracker=SimpleNamespace(record_snapshot=lambda _snapshot: None),
        sessions=SimpleNamespace(save=lambda _session: None),
        _now_fn=lambda: datetime(2026, 9, 11),
        _build_usage_snapshot=AgentLoop._build_usage_snapshot,
        # The snapshot asks the providers map whether this model is billed by
        # subscription, because a plan is the entry's `login` and nothing about
        # the model says so. An empty map is "no plan", which is what a metered
        # key is, and is the case these tests are about.
        settings=SimpleNamespace(providers=ProvidersConfig()),
    )
    outcome = LoopOutcome(
        usage={"prompt_tokens": 190_000, "context_used": 190_000, "context_max": 200_000},
        final_call=FinalCall(model=MODEL, prompt_tokens=190_000, window=200_000),
    )

    asyncio.run(AgentLoop._maybe_compact_remote(compacted, Session(key="tui:t"), outcome, session_key="tui:t"))

    assert "list_cost_usd" not in outcome.usage
    assert "cost_usd" not in outcome.usage


async def _answer(value):
    return value
