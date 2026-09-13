"""``/status`` reports what this session has spent, at list price."""

from __future__ import annotations

import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.providers.base import LLMResponse
from opendde_harness.providers.rates import ListRates, ListTier
from opendde_harness.token_wise.base import CURRENT_SESSION_KEY, UsageSnapshot
from opendde_harness.token_wise.usage_tracker import UsageTracker
from opendde_harness.tui_rpc.methods import slash_routing

SESSION = "tui:session-1"
SONNET = "anthropic/claude-sonnet-4-5"
LONG_CONTEXT = "openai/gpt-5.4"
LIST = {
    SONNET: ListRates(input=3e-6, output=15e-6, cache_read=0.3e-6, cache_write=3.75e-6),
    "openai-codex/gpt-5.1": ListRates(input=1.25e-6, output=10e-6, cache_read=0.125e-6, cache_write=1.25e-6),
    LONG_CONTEXT: ListRates(
        input=2.5e-6,
        output=15e-6,
        cache_read=0.25e-6,
        cache_write=2.5e-6,
        tiers=(ListTier(5e-6, 22.5e-6, 0.5e-6, 5e-6, input_tokens_above=272_000),),
    ),
}


class Rated:
    """A provider that answers with one model's published rates.

    What ``PiModelProvider.list_rates`` does with the service's own row; the
    report is about the totals, so the rates are stated here rather than read
    from a catalogue.
    """

    def __init__(self, model: str) -> None:
        self._model = model

    def list_rates(self) -> ListRates | None:
        return LIST.get(self._model)


class FakeLoop:
    def __init__(self, model: str = SONNET):
        self.model = model
        self.usage_tracker = UsageTracker(persist=False)

    def session_model(self, session_key: str) -> str:
        return self.model

    async def record(self, session_key: str, model: str | None = None, cache_reported: bool = True, **fields):
        name = model or self.model
        snapshot = UsageSnapshot(model=name, session_key=session_key, cache_reported=cache_reported, **fields)
        # Priced as the loop prices it: the provider's rates for the model that
        # answered, at the tier this call's own input reaches.
        rates = LIST.get(name)
        if rates is not None:
            snapshot.list_cost_usd = rates.cost(
                snapshot.input_tokens, snapshot.output_tokens, snapshot.cache_read_tokens, snapshot.cache_write_tokens
            )
        await self.usage_tracker.after_llm_call({}, snapshot)

    def side_call(self, response: LLMResponse) -> None:
        """A call made on the turn's behalf, as ``AgentLoop._record_side_call``
        records it: the same snapshot builder, the same rates."""
        if response.finish_reason == "error":
            return
        model = response.model or self.model
        self.usage_tracker.record_snapshot(
            AgentLoop._build_usage_snapshot(response, model, CURRENT_SESSION_KEY.get() or "", Rated(model))
        )

    def report(self, session_key: str = SESSION) -> str:
        return slash_routing.session_usage_report(
            session_key, self.model, self.usage_tracker.session_usage(session_key)
        )


@pytest.fixture(autouse=True)
def local_list_prices(monkeypatch):
    monkeypatch.setattr(slash_routing, "cli_dispatch", _cli_status)
    yield
    # The mark a test sets on the session ContextVar must not outlive it: a
    # later test asserting the turn boundary leaves no mark would read this one.
    CURRENT_SESSION_KEY.set(None)


async def _cli_status(params, **kwargs):
    assert params["argv"] == ["status"]
    return {"stdout": "OpenDDE Harness Status\nConfig: ok\n", "stderr": "", "exit_code": 0}


async def test_status_appends_session_totals_cache_rate_and_list_price_cost():
    loop = FakeLoop()
    await loop.record(SESSION, input_tokens=400, cache_read_tokens=1_500, cache_write_tokens=100, output_tokens=250)
    await loop.record(SESSION, input_tokens=100, cache_read_tokens=1_900, output_tokens=50)
    await loop.record("tui:other", input_tokens=9_999, output_tokens=9_999)

    result = await slash_routing.session_status({"session_id": SESSION}, agent_loop_factory=lambda: loop)

    lines = result["output"].splitlines()
    assert lines[0] == "OpenDDE Harness Status"
    assert f"Session: {SESSION}" in lines
    assert f"Model: {SONNET}" in lines
    assert (
        "Tokens: 4,300 total · 4,000 input · 300 output (summed over 2 calls; the context bar shows the last call's input plus output)"
        in lines
    )
    assert "Cache hit rate: 85.0% (3,400 of 4,000 input tokens read from cache · 100 written)" in lines
    # 500 fresh * 3 + 300 out * 15 + 3400 cache read * 0.3 + 100 cache write * 3.75, per million.
    # Three decimals, as the footer prints it: the two are read side by side and
    # must round one figure the same way.
    assert "Estimated cost: $0.007 at list price" in lines


async def test_status_without_a_session_is_the_cli_output_alone():
    result = await slash_routing.session_status({}, agent_loop_factory=lambda: FakeLoop())

    assert result["output"] == "OpenDDE Harness Status\nConfig: ok\n"


def test_a_fresh_session_reads_as_zero_not_unknown():
    report = FakeLoop().report()

    assert "Tokens: 0 total · 0 input · 0 output" in report
    assert "Cache hit rate: n/a (no calls yet)" in report
    assert "Estimated cost: $0.000 at list price" in report


async def test_cache_rate_counts_only_the_calls_the_provider_reported():
    """A relay that forwards cache fields on a few calls: the rate is over
    those, and the line says how many were counted, instead of a near-zero
    figure over every token that nobody reported on."""
    loop = FakeLoop()
    await loop.record(SESSION, input_tokens=340_000, cache_reported=False)
    await loop.record(SESSION, input_tokens=340_000, cache_reported=False)
    await loop.record(SESSION, input_tokens=100, cache_read_tokens=3_900)

    report = loop.report()

    assert "Tokens: 684,000 total · 684,000 input · 0 output" in report
    assert (
        "Cache hit rate: 97.5% (3,900 of 4,000 input tokens read from cache on the 1 of 3 calls that reported one"
        " · 0 written)" in report
    )


async def test_cache_rate_is_not_reported_when_no_call_carried_a_figure():
    loop = FakeLoop()
    await loop.record(SESSION, input_tokens=5_000, cache_reported=False)

    report = loop.report()

    assert "Cache hit rate: not reported (none of the 1 calls carried a cache figure)" in report
    assert "0.0%" not in report


async def test_a_session_that_switched_models_is_priced_model_by_model():
    loop = FakeLoop("openai-codex/gpt-5.1")
    await loop.record(SESSION, model=SONNET, input_tokens=1_000_000)
    await loop.record(SESSION, output_tokens=100_000)

    report = loop.report()

    assert "Model: openai-codex/gpt-5.1" in report
    assert "Estimated cost: $4.000 at list price across 2 models" in report


async def test_each_call_pays_its_own_long_context_tier():
    """Three 100k calls and one 300k call carry the same total input; only
    the 300k one crosses the tier, and only it is priced there."""
    loop = FakeLoop(LONG_CONTEXT)
    for _ in range(3):
        await loop.record(SESSION, input_tokens=100_000)
    await loop.record(SESSION, input_tokens=300_000, output_tokens=1_000)

    report = loop.report()

    # 3 * 100k * 2.5 + 300k * 5 + 1k * 22.5, per million.
    assert "Estimated cost: $2.272 at list price" in report


async def test_an_unpriced_model_is_named_rather_than_counted_as_free():
    loop = FakeLoop("my-vllm/my-finetune")
    await loop.record(SESSION, input_tokens=1_000, output_tokens=100)

    alone = loop.report()
    await loop.record(SESSION, model=SONNET, output_tokens=100_000)
    mixed = loop.report()

    assert "Estimated cost: unknown (no list price for my-vllm/my-finetune)" in alone
    assert "Estimated cost: $1.500 at list price · no list price for my-vllm/my-finetune" in mixed


def test_calls_made_on_the_turns_behalf_land_in_the_turns_session():
    """A subsystem that calls the provider outside the loop hands its reply to
    the tracker, which files it under the session the loop marked current and
    the model the retry ladder says answered."""
    loop = FakeLoop()
    CURRENT_SESSION_KEY.set(SESSION)
    loop.side_call(LLMResponse(content="{}", usage={"prompt_tokens": 1_000, "completion_tokens": 100}, model=SONNET))
    loop.side_call(LLMResponse(content="x", finish_reason="error", usage={"prompt_tokens": 9_999}))

    report = loop.report()

    assert "Tokens: 1,100 total · 1,000 input · 100 output" in report
    assert "Estimated cost: $0.005 at list price" in report


def test_the_status_reader_gets_a_copy_it_can_hold_while_the_loop_keeps_recording():
    loop = FakeLoop()
    CURRENT_SESSION_KEY.set(SESSION)
    loop.side_call(LLMResponse(content="", usage={"prompt_tokens": 10}, model=SONNET))

    usage = loop.usage_tracker.session_usage(SESSION)
    loop.side_call(LLMResponse(content="", usage={"prompt_tokens": 10}, model="openai-codex/gpt-5.1"))
    usage.by_model.pop(SONNET)

    assert list(loop.usage_tracker.by_model(SESSION)) == [SONNET, "openai-codex/gpt-5.1"]
    assert usage.totals.input_tokens == 10 and usage.counts.calls == 1


def test_status_before_the_agent_starts_names_the_session_only():
    report = slash_routing.session_usage_report(SESSION, "", None)

    assert report.splitlines() == [f"Session: {SESSION}", "Usage: unavailable until the agent has started"]
