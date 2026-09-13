"""The agent loop driven by pi-ai, offline, against the service's faux models.

Needs the built bundle: ``cd ui-tui && npm run build`` produces
``ui-tui/dist/model-service.js``. Without it the module skips.

Nothing here reaches a vendor and nothing reads a real credential directory:
the ``faux`` and ``faux-slow`` providers are scripted inside the service, and
the one test that builds a provider from configuration builds a synthetic one.
"""

from __future__ import annotations

import copy
import inspect
import os
import shutil
import uuid
from contextlib import aclosing
from pathlib import Path
from typing import Any

import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.agent.tools.base import Tool
from opendde_harness.config.schema import Config, ModelEntry, ProvidersConfig
from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import GenerationSettings, LLMProvider
from opendde_harness.providers.messages import wire_projection
from opendde_harness.providers.model_service import ModelService
from opendde_harness.providers.pi_provider import PiModelProvider
from opendde_harness.spine import ChatType, Origin, Source, TurnRequest
from opendde_harness.token_wise.base import CURRENT_SESSION_KEY
from tests._config import config as build_config
from tests._config import declared
from tests._messages import user

BUNDLE = Path(__file__).resolve().parent.parent / "ui-tui" / "dist" / "model-service.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    not BUNDLE.exists() or NODE is None,
    reason="ui-tui/dist/model-service.js is not built (cd ui-tui && npm run build) or node is missing",
)

MODEL = "faux/echo"
SLOW = "faux-slow/echo"


class Recording(ModelService):
    """The service, plus a copy of every context, options and replay it was sent."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.contexts: list[dict] = []
        self.options: list[dict | None] = []
        self.replays: list[dict | None] = []
        self.retries: list[dict | None] = []
        self.timeouts: list[dict | None] = []

    async def stream_with_id(self, provider, model, context, options=None, *, replay=None, retry=None, timeouts=None):
        self.contexts.append(copy.deepcopy(context))
        self.options.append(copy.deepcopy(options))
        self.replays.append(copy.deepcopy(replay))
        self.retries.append(copy.deepcopy(retry))
        self.timeouts.append(copy.deepcopy(timeouts))
        return await super().stream_with_id(
            provider, model, context, options, replay=replay, retry=retry, timeouts=timeouts
        )


@pytest.fixture
async def service():
    svc = Recording(node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1"})
    await svc.start()
    try:
        yield svc
    finally:
        await svc.close()


class Echo(Tool):
    """The tool the faux model always calls, recording what it was handed."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "echo the text back"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, text: str = "", **_kwargs) -> str:
        self.seen.append(text)
        return f"echoed {text}"


def provider(service: ModelService, model: str = MODEL, **generation) -> PiModelProvider:
    provider_id, _, model_id = model.partition("/")
    built = PiModelProvider(service, provider_id, model_id)
    if generation:
        built.generation = GenerationSettings(**generation)
    return built


def loop_for(tmp_path, built: PiModelProvider, **settings) -> AgentLoop:
    return AgentLoop(built, tmp_path, AgentLoopSettings(model=built.stored_model, **settings))


def request(text: str = "ping") -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel="cli", chat_id="t", sender_id="user", chat_type=ChatType.DM),
        text=text,
        conversation="cli:t",
    )


async def _emit(_event) -> None:
    pass


def assistants(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") == "assistant"]


# ---------------------------------------------------------------------------
# One turn, end to end
# ---------------------------------------------------------------------------


async def test_a_turn_calls_the_tool_and_replays_pis_own_message_on_the_next_call(tmp_path, service):
    """The model layer's message is stored whole and sent back unchanged.

    The faux model answers every request with a tool call, so the turn runs to
    its iteration budget; what matters is the first call's answer reaching the
    second call's context as the object pi produced, not a re-rendering of it.
    """
    built = provider(service)
    loop = loop_for(tmp_path, built, max_iterations=2)
    echo = Echo()
    loop.tools.register(echo)

    await loop.run_turn(request("ping"), _emit, lambda: [], stream=True)
    await loop.close_mcp()

    # The loop wraps the user's text in a runtime-context header before the
    # model sees it, so what the faux model echoes back ends with the message.
    assert echo.seen and echo.seen[0].endswith("ping"), "the tool was called with what the model echoed"

    stored = assistants(loop.sessions.get_or_create("cli:t").messages)
    assert stored, "the turn recorded an assistant message"
    first = stored[0]
    # The record *is* pi's message: its own api / provider / model triple, which
    # is what makes pi replay the turn as its own rather than re-render it.
    assert [block["type"] for block in first["content"]] == ["thinking", "text", "toolCall"]
    assert first["api"] and first["provider"] == "faux" and first["model"] == "echo"

    assert len(service.contexts) >= 2, "the turn made more than one call"
    replayed = [m for m in service.contexts[1]["messages"] if m.get("role") == "assistant"]
    assert replayed == [wire_projection(first)], "the second call replays it verbatim"


async def test_a_stored_turn_reloaded_from_disk_is_replayed_to_the_service_verbatim(tmp_path, service):
    """The whole point of one shape: what the file holds is what the wire gets.

    A turn with tool calls is run, the file is read back by a fresh
    ``SessionManager``, and that history is sent. Every message the service
    receives is the stored record, less the harness's own bookkeeping -- no
    rebuilding on the way out, and no second representation to disagree with the
    first.
    """
    from opendde_harness.session.manager import SessionManager

    built = provider(service)
    loop = loop_for(tmp_path, built, max_iterations=2)
    loop.tools.register(Echo())

    await loop.run_turn(request("ping"), _emit, lambda: [], stream=True)
    loop.sessions.flush("cli:t")
    await loop.close_mcp()

    # A fresh reader, so nothing in memory can answer for the file.
    stored = SessionManager(tmp_path).get_or_create("cli:t").messages
    assert [m["role"] for m in stored] == [
        "user",
        "assistant",
        "toolResult",
        "assistant",
        "toolResult",
        "assistant",
    ]
    history = SessionManager(tmp_path).get_or_create("cli:t").get_history()

    async with aclosing(built.chat_stream(history)) as deltas:
        async for _delta in deltas:
            pass

    sent = service.contexts[-1]["messages"]
    assert sent[:-1] == [wire_projection(m) for m in stored[:-1]], "stored record in, same object out"
    # pi reads ``usage.totalTokens`` off every message it replays and does not
    # check the field is there: a record stored without one failed every request
    # after the first with "Cannot read properties of undefined". The service
    # answering at all is the assertion; the field being present is why.
    for replayed in (m for m in sent if m["role"] == "assistant"):
        assert "totalTokens" in replayed["usage"]
    # The wrap-up is the one exception, and it is the reconciling rule rather
    # than a second shape: the faux model asked for a tool on a call that
    # withheld them, nothing answered it, and an unanswered call must not be
    # replayed as an instruction the model would answer twice. The journal keeps
    # what it intended; the request does not carry it.
    assert msg.tool_calls_of(stored[-1]) and not msg.tool_calls_of(sent[-1])
    assert msg.text_of(sent[-1]) == msg.text_of(stored[-1])


async def test_the_stream_delivers_text_and_reasoning_to_the_loops_callbacks(tmp_path, service):
    loop = loop_for(tmp_path, provider(service))
    text: list[str] = []
    reasoning: list[str] = []

    async def on_token(chunk: str) -> None:
        text.append(chunk)

    async def on_reasoning(chunk: str) -> None:
        reasoning.append(chunk)

    response = await loop._llm_call_stream(
        [user("hello")],
        None,
        MODEL,
        on_token_delta=on_token,
        on_reasoning_delta=on_reasoning,
    )

    assert len(text) > 1, "text arrived as deltas, not one block"
    assert "".join(text).startswith("Echo: hello.")
    assert "hello" in "".join(reasoning)
    assert response.finish_reason == "tool_calls"
    assert response.pi_message is not None
    assert response.tool_calls[0].name == "echo"


# ---------------------------------------------------------------------------
# The two deadlines
# ---------------------------------------------------------------------------


async def test_an_idle_stream_is_retried_and_then_ends_the_turn(tmp_path, service):
    """A silence past the idle budget is a transient failure like any other.

    The deadline is the service's, so the attempt it ends is one pi's retry
    loop sees: the first silence is announced and the call runs again, and only
    when the second attempt goes quiet too is the turn's answer an error.
    ``faux-slow`` delivers twenty tokens a second, so every gap after the first
    event is fifty times the budget set here.
    """
    built = provider(service, SLOW, first_token_timeout=10.0, idle_timeout=0.01, retries=1)
    loop = loop_for(tmp_path, built)
    retries: list[tuple] = []

    async def on_retry(attempt, total, reason, delivered):
        retries.append((attempt, total, reason, delivered))

    response = await loop._llm_call_stream([user("slow")], None, SLOW, on_retry=on_retry)

    assert len(service.contexts) == 1, "the attempts are the service's, not two requests"
    assert [(a, t) for a, t, _r, _d in retries] == [(2, 2)], "one silence announced, one budget spent"
    assert "timed out" in retries[0][2]
    assert response.finish_reason == "error"
    assert response.error_classification is not None
    assert response.error_classification.category == "timeout"
    assert response.error_classification.retryable is True
    assert "0.01s of silence" in (response.content or "")


async def test_the_gap_budgets_travel_with_the_request_in_milliseconds(tmp_path, service):
    """Enforced at the service, so they are sent rather than applied here."""
    built = provider(service, first_token_timeout=12.5, idle_timeout=3.0)
    loop = loop_for(tmp_path, built)

    await loop._llm_call_stream([user("hi")], None, MODEL)

    assert service.timeouts == [{"firstTokenMs": 12500, "idleMs": 3000}]


async def test_every_request_carries_the_configured_retry_budget(tmp_path, service):
    """The budget is the service's to spend, so it travels with each request."""
    built = provider(service, retries=2)
    loop = loop_for(tmp_path, built)

    await loop._llm_call_stream([user("hi")], None, MODEL)

    assert service.retries == [{"maxRetries": 2}]

    # And a provider configured for none still names the budget, rather than
    # leaving the service to pick one.
    none = Recording(node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1"})
    await none.start()
    try:
        await loop_for(tmp_path, provider(none))._llm_call_stream([user("hi")], None, MODEL)
    finally:
        await none.close()
    assert none.retries == [{"maxRetries": 0}]


# ---------------------------------------------------------------------------
# The process-wide service
# ---------------------------------------------------------------------------


@pytest.fixture
async def isolated_service(tmp_path, monkeypatch):
    """``get_service``, with its credential file in ``tmp_path`` and faux providers.

    ``CHATGPT_TOKEN_DIR`` is the documented override for this project's token
    directory, and ``credential_store_path`` resolves through it, so nothing
    here reads or writes the real one. Ambient vendor keys are cleared for the
    same reason ``test_pi_auth`` clears them: a developer's exported key would
    otherwise be seeded into the store this test asserts on.
    """
    from opendde_harness.providers import pi_service

    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_AUTH_TOKEN")) or name.startswith("CHATGPT_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "tokens"))
    monkeypatch.setenv("OPENDDE_MODEL_SERVICE_FAUX", "1")
    try:
        yield pi_service
    finally:
        await pi_service.shutdown_service()


async def test_one_child_serves_the_process_and_is_reconfigured_when_the_config_changes(isolated_service, tmp_path):
    pi_service = isolated_service
    configured: list[dict] = []
    original = ModelService.configure

    async def recording(self, payload):
        configured.append(payload)
        return await original(self, payload)

    ModelService.configure = recording
    try:
        first = await pi_service.get_service(relay_config())
        again = await pi_service.get_service(relay_config())
        assert again is first, "the same child answers the whole process"
        assert len(configured) == 1, "an unchanged configuration is not re-sent"

        edited = relay_config()
        edited.providers.get("custom").api_key = "sk-other-synthetic"
        assert await pi_service.get_service(edited) is first
        assert len(configured) == 2, "an edited section reaches the running child"
    finally:
        ModelService.configure = original

    store = pi_service.credential_store_path()
    assert store == tmp_path / "tokens" / "pi-credentials.json"
    # Handed over, never written from here: the service writes the keys into
    # it itself, and a declared provider's key travels on its own row.
    assert not store.exists()
    assert configured[0]["credentials"] == str(store)
    assert [p["id"] for p in configured[0]["providers"]] == ["custom"]


async def test_the_service_answers_a_provider_built_from_configuration(isolated_service):
    """The lazy start: nothing runs until the first request needs it."""
    from opendde_harness.cli._helpers import make_provider

    built = make_provider(relay_config())
    built.provider_id, built.model_id = "faux", "echo"

    deltas = [delta async for delta in built.chat_stream([user("wired")])]

    assert deltas[-1].final_response is not None
    assert deltas[-1].final_response.tool_calls[0].arguments == {"text": "wired"}
    assert built.context_window() == 128_000


# ---------------------------------------------------------------------------
# What the service says about a model
# ---------------------------------------------------------------------------


async def test_one_listing_answers_every_fact_the_loop_decides_with(service):
    """Window, output ceiling and modalities come from the same primed row.

    Nothing is known before the first request -- there is no service to ask
    until one is started -- and unknown is answered as unknown rather than as a
    number nothing measured.
    """
    built = provider(service)

    assert built.context_window() is None
    assert built.max_output_tokens() is None
    assert built.input_modalities() is None

    async for _delta in built.chat_stream([user("hi")]):
        pass

    assert built.context_window() == 128_000
    assert built.max_output_tokens() == 16_384
    assert built.input_modalities() == ("text", "image")


async def test_the_catalogue_answers_for_a_model_nothing_configured(service):
    """``catalog`` reads pi's own built-in rows, whoever is configured.

    This service has had no ``configure`` at all, and the id asked about
    belongs to none of its faux providers -- which is the point: a declared
    relay's model is sized from the vendor's row while the configuration that
    declares it is still being built.
    """
    rows = await service.catalog("deepseek-v4-flash")

    assert rows, "pi carries a row for a vendor model id"
    assert {row["id"] for row in rows} == {"deepseek-v4-flash"}
    vendor = next(row for row in rows if row["provider"] == "deepseek")
    assert vendor["contextWindow"] > 0 and vendor["maxTokens"] > 0
    # The rates ride along, per million, which is what `list_rates` divides.
    assert vendor["cost"]["input"] > 0

    assert await service.catalog("a-model-pi-never-heard-of") == []


async def test_the_models_listing_carries_the_rates_too(service):
    """``models`` reports ``cost`` per row: the loop prices a call from it."""
    rows = await service.models()
    faux = next(row for row in rows if row["provider"] == "faux" and row["id"] == "echo")

    assert set(faux["cost"]) >= {"input", "output", "cacheRead", "cacheWrite"}


def test_rates_come_from_the_rows_own_cost_per_million():
    """pi publishes per million; the loop's arithmetic is per token."""
    built = PiModelProvider(None, "vendor", "model")
    built._row = {
        "cost": {
            "input": 3.0,
            "output": 15.0,
            "cacheRead": 0.3,
            "cacheWrite": 3.75,
            "tiers": [{"inputTokensAbove": 200_000, "input": 6.0, "output": 22.5, "cacheRead": 0.6, "cacheWrite": 7.5}],
        }
    }

    rates = built.list_rates()

    assert rates is not None
    assert rates.input == pytest.approx(3e-6)
    assert rates.cache_write == pytest.approx(3.75e-6)
    # A call past the threshold is priced at its own tier, whole.
    assert rates.for_input(200_001).input == pytest.approx(6e-6)
    assert rates.cost(1_000, 100, 0, 0) == pytest.approx(1_000 * 3e-6 + 100 * 15e-6)


def test_a_row_priced_at_zero_has_no_price_rather_than_a_free_one():
    """pi prices a provider this project declared at 0, because `configure`
    sends no rates for one. That is nobody knowing, not a free model."""
    built = PiModelProvider(None, "custom", "qwen3-32b")
    built._row = {"cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}

    assert built.list_rates() is None
    assert PiModelProvider(None, "custom", "qwen3-32b").list_rates() is None, "and none at all before the first call"


# ---------------------------------------------------------------------------
# The two ladders
# ---------------------------------------------------------------------------


def _row(**fields: Any) -> ModelEntry:
    """The row a user would write for ``faux/echo``, on its own.

    A limit is declared per model, as a row in the provider's own entry, so the
    row IS the declaration -- there is no separate overlay keyed by a spelling
    of the pair any more.
    """
    return ModelEntry(id="echo", **fields)


def _declares(**fields: Any) -> ProvidersConfig:
    """A providers section whose ``faux`` entry declares that row for ``echo``."""
    return build_config(declared("faux", models=[{"id": "echo", **fields}]), model=MODEL).providers


async def test_the_window_is_the_declared_row_then_the_service_then_unknown(tmp_path, service):
    built = provider(service)
    loop = loop_for(tmp_path, built)

    from opendde_harness.providers.rates import SOURCE_DECLARED, SOURCE_SERVICE, SOURCE_UNKNOWN

    assert loop.resolve_window().source == SOURCE_UNKNOWN, "nothing is known before the first call"

    async for _delta in built.chat_stream([user("hi")]):
        pass

    resolved = loop.resolve_window()
    assert (resolved.tokens, resolved.source) == (128_000, SOURCE_SERVICE)

    # A declaration is the operator describing their own deployment, and beats
    # the service's row for it.
    with_row = loop_for(tmp_path, built, providers=_declares(contextWindow=40_960))
    assert (with_row.resolve_window().tokens, with_row.resolve_window().source) == (40_960, SOURCE_DECLARED)


async def test_the_ceiling_is_the_declared_row_then_the_service_then_the_pin(tmp_path, service):
    from opendde_harness.providers.base import GenerationSettings, send_max_tokens

    built = provider(service)
    pinned = GenerationSettings(max_tokens=2_048)

    # Nothing knows it yet and nobody pinned: no ceiling to check and none to
    # send. Such a request is refused by the service, not floored at a token --
    # see test_model_service.test_a_model_with_no_declared_ceiling_is_refused_rather_than_sent.
    assert send_max_tokens(None, MODEL, provider=built) is None
    assert send_max_tokens(pinned, MODEL, provider=built) == 2_048

    async for _delta in built.chat_stream([user("hi")]):
        pass

    assert send_max_tokens(None, MODEL, provider=built) == 16_384
    assert send_max_tokens(pinned, MODEL, provider=built) == 2_048, "a pin is bounded by the ceiling, never above it"
    assert send_max_tokens(None, MODEL, overlay=_row(maxTokens=1_024), provider=built) == 1_024

    loop = loop_for(tmp_path, built)
    assert loop._reserved_output(None, MODEL) == 16_384
    assert loop._reserved_output(1_000, MODEL) == 1_000, "never more than the window"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_the_backends_own_price_for_a_call_is_what_the_tracker_records():
    """pi prices the call from the catalogue that served it; we defer to that."""
    from types import SimpleNamespace

    usage = {"prompt_tokens": 1000, "completion_tokens": 200, "cost": 0.0125}
    snapshot = AgentLoop._build_usage_snapshot(SimpleNamespace(usage=usage, model=MODEL), MODEL, "")

    assert snapshot.estimated_cost_usd == 0.0125
    assert (snapshot.input_tokens, snapshot.output_tokens) == (1000, 200)


def test_fresh_input_is_taken_as_reported_and_never_reduced_by_the_cache_beside_it():
    """pi's ``input`` already excludes what was read from cache. A call whose
    fresh tokens outnumber its cached ones -- a large tool result on a reused
    prefix -- was counted short by the cached amount, and its hit rate high."""
    from types import SimpleNamespace

    from opendde_harness.tracing.semconv import _llm_attrs

    usage = {"prompt_tokens": 6000, "completion_tokens": 10, "cache_read_input_tokens": 4608, "total_tokens": 10618}
    snapshot = AgentLoop._build_usage_snapshot(SimpleNamespace(usage=usage, model=MODEL), MODEL, "")
    span = _llm_attrs(SimpleNamespace(usage=usage), "faux", MODEL)

    assert (snapshot.input_tokens, snapshot.cache_read_tokens) == (6000, 4608)
    assert (span["llm.usage.input_tokens"], span["llm.usage.total_tokens"]) == (6000, 10618)


def test_a_price_of_zero_is_nobody_knowing_rather_than_free():
    """A declared provider carries no rates, so pi prices its calls at zero."""
    from types import SimpleNamespace

    usage = {"prompt_tokens": 1000, "completion_tokens": 200, "cost": 0.0}
    snapshot = AgentLoop._build_usage_snapshot(SimpleNamespace(usage=usage, model=MODEL), MODEL, "")

    assert snapshot.estimated_cost_usd is None, "the faux model is in no price table of ours either"


def test_a_plan_is_still_the_price_whatever_the_backend_computed():
    """A subscription is the price, whatever figure the backend put on the call.

    ``openai-codex`` is reached by a sign-in (``login: "oauth"``), and a
    per-token figure for a turn nobody is billed per token for is the wrong
    number whoever computed it. The entry is the only thing that says so, which
    is why the snapshot is handed the providers map -- the same one the loop
    hands it from its own settings.
    """
    from types import SimpleNamespace

    from tests._config import config

    plan = "openai-codex/gpt-5.6-luna"
    providers = config({"openai-codex": {"login": "oauth"}}, model=plan).providers
    usage = {"prompt_tokens": 1000, "completion_tokens": 200, "cost": 0.0125}
    response = SimpleNamespace(usage=usage, model=plan)

    assert AgentLoop._build_usage_snapshot(response, plan, "", None, providers).estimated_cost_usd is None
    # And a key under the same id is metered, so the figure stands: the plan is
    # the entry's own word, not a fact about the vendor.
    metered = config({"openai-codex": {"apiKey": "sk-x"}}, model=plan).providers
    assert AgentLoop._build_usage_snapshot(response, plan, "", None, metered).estimated_cost_usd == 0.0125


async def test_abandoning_a_half_read_stream_leaves_the_service_usable(service):
    """What a cancelled turn does: the loop closes the generator mid-answer.

    Closing it unwinds through the service's own iterator, which aborts the
    request it opened; what is checked here is the part this side owns -- the
    close neither hangs nor raises, and the next request is answered normally.
    """
    built = provider(service, SLOW, first_token_timeout=10.0, idle_timeout=10.0)
    stream = built.chat_stream([user("slow")])

    async with aclosing(stream) as deltas:
        async for delta in deltas:
            if delta.content:
                break

    rest = [event async for event in service.stream("faux-slow", "echo", {"messages": []})]
    assert rest[-1]["type"] in {"done", "error"}, "the service is still answering"


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


async def test_the_window_comes_from_the_services_own_listing(tmp_path, service):
    built = provider(service)
    loop = loop_for(tmp_path, built)

    assert built.context_window() is None, "nothing is claimed before the service has answered"

    await loop._llm_call_stream([user("hi")], None, MODEL)

    assert built.context_window() == 128_000
    resolved = loop.resolve_window(MODEL)
    assert (resolved.tokens, resolved.source) == (128_000, "model-service")


async def test_a_declared_window_still_beats_the_services_number(tmp_path, service):
    """The user describing their own deployment stays the first tier."""
    built = provider(service)
    loop = loop_for(tmp_path, built, providers=_declares(contextWindow=4096))

    await loop._llm_call_stream([user("hi")], None, MODEL)

    assert built.context_window() == 128_000
    assert loop.resolve_window(MODEL).tokens == 4096


# ---------------------------------------------------------------------------
# What make_provider builds
# ---------------------------------------------------------------------------


def relay_config() -> Config:
    """One declared OpenAI-compatible provider. Synthetic: no vendor, no key file.

    ``custom`` is a name pi does not ship, so it can only be the declared kind:
    an address and the protocol it speaks, and the model list is its whole
    catalogue because pi has none for a provider it never heard of. Nothing
    beside the model id names the provider -- the prefix of
    ``custom/qwen3-32b`` is it.
    """
    return build_config(
        declared(
            "custom",
            base_url="http://127.0.0.1:8000/v1",
            api="openai-completions",
            apiKey="sk-custom-synthetic",
            models=["qwen3-32b"],
        ),
        model="custom/qwen3-32b",
    )


def test_make_provider_builds_the_model_service_and_nothing_else():
    """One model layer, for every model and every provider, with nothing to
    choose between: no switch, no second route to fall back to."""
    from opendde_harness.cli._helpers import make_provider

    built = make_provider(relay_config())

    assert isinstance(built, PiModelProvider)
    assert (built.provider_id, built.model_id) == ("custom", "qwen3-32b")
    assert built.stored_model == "custom/qwen3-32b"
    assert built.get_default_model() == "custom/qwen3-32b"
    # The generation defaults are copied onto it.
    assert built.generation.first_token_timeout == relay_config().agents.defaults.llm_first_token_timeout


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


async def test_the_generation_settings_reach_pi_as_its_own_options(service):
    built = provider(service, first_token_timeout=10.0, reasoning_effort="high", max_tokens=900)

    token = CURRENT_SESSION_KEY.set("tui:one")
    try:
        async for _delta in built.chat_stream([user("hi")], tool_choice="auto"):
            pass
    finally:
        CURRENT_SESSION_KEY.reset(token)

    assert service.options[-1] == {"reasoning": "high", "maxTokens": 900, "toolChoice": "auto", "sessionId": "tui:one"}


async def _usage_of_one_call(built: PiModelProvider) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    async for delta in built.chat_stream([user("hello")]):
        if delta.usage:
            usage = delta.usage
    return usage


async def test_every_request_is_keyed_to_its_session_so_the_backend_reuses_the_prefix(service, monkeypatch):
    """pi files a prompt cache under ``sessionId``: Codex sends it as
    ``prompt_cache_key`` and its session header, and without one every call of
    a conversation opens a fresh cache under a fresh id -- eleven calls of one
    session read 3% of their input from cache. The key is the session the loop
    is serving; a provider serving none keys on itself; a compaction carries
    the same key as the turns it stands for."""
    built = provider(service)
    key = f"tui:{uuid.uuid4().hex}"

    token = CURRENT_SESSION_KEY.set(key)
    try:
        await _usage_of_one_call(built)
        second = await _usage_of_one_call(built)
    finally:
        CURRENT_SESSION_KEY.reset(token)

    assert [sent["sessionId"] for sent in service.options[-2:]] == [key, key]
    assert second["cache_read_input_tokens"] > 0, "the backend was shown the same prefix under the same key"

    await _usage_of_one_call(built)
    await _usage_of_one_call(provider(service))
    own, other = (sent["sessionId"] for sent in service.options[-2:])
    assert own and own != key and own != other, "no session: each provider keys on itself"

    compacted: list[dict[str, Any]] = []

    async def compact(provider_id, model_id, context, **kwargs):
        compacted.append(kwargs)
        return {"items": [{"type": "message"}], "usage": {}}

    monkeypatch.setattr(PiModelProvider, "supports_compaction", property(lambda self: True))
    monkeypatch.setattr(service, "compact", compact)
    token = CURRENT_SESSION_KEY.set(key)
    try:
        await built.compact([user("hello")])
    finally:
        CURRENT_SESSION_KEY.reset(token)
    assert compacted[0]["session_id"] == key


async def test_a_temperature_is_sent_only_where_a_model_row_asks_for_one(service):
    """There is no default temperature to inherit and no way for a caller to name
    one. One went out on every request -- to every model, including the ones that
    reject the parameter outright and refuse the whole turn over it ("Unsupported
    parameter: temperature") -- and the settings a session shares cannot be right
    for every model it switches between. The model's own row is the only place a
    temperature is declared."""
    built = provider(service, first_token_timeout=10.0)

    async for _delta in built.chat_stream([user("hi")]):
        pass
    assert "temperature" not in service.options[-1], "no row declares one"

    built.providers = _declares(temperature=0.2)
    async for _delta in built.chat_stream([user("hi")]):
        pass
    assert service.options[-1]["temperature"] == 0.2

    # And a call cannot ask for one at all: the parameter is gone from the
    # request path, so a caller reaching for it is a TypeError rather than a
    # number that overrides the row.
    with pytest.raises(TypeError):
        async for _delta in built.chat_stream([user("hi")], temperature=0.9):
            pass
    for method in (LLMProvider.chat, LLMProvider.chat_stream, LLMProvider.chat_with_retry):
        assert "temperature" not in inspect.signature(method).parameters, method.__name__


async def test_a_thinking_level_pi_has_no_name_for_is_left_out(service):
    built = provider(service, first_token_timeout=10.0, reasoning_effort="off")

    async for _delta in built.chat_stream([user("hi")], tool_choice="required"):
        pass

    sent = service.options[-1]
    assert "reasoning" not in sent, "pi spells 'off' by omission"
    assert "toolChoice" not in sent, "pi has no 'required'"


def test_a_command_loop_ends_the_service_it_started_before_the_loop_closes():
    """``asyncio.run`` closes its loop on return. A service started inside it
    and left there outlived it: the next command's loop abandoned the child
    (a warning naming its pid) and the transport the old loop held was
    finalised later, raising "Event loop is closed" inside ``__del__`` under
    some later line of the wizard's output."""
    import asyncio

    from opendde_harness.providers import pi_service

    class _Service:
        closed_on: asyncio.AbstractEventLoop | None = None

        async def close(self) -> None:
            self.closed_on = asyncio.get_running_loop()

        def abandon(self) -> None:
            raise AssertionError("ended on its own loop, never abandoned")

    fake = _Service()

    async def command() -> str:
        pi_service._service, pi_service._service_loop = fake, asyncio.get_running_loop()
        return "answer"

    assert pi_service.run_then_shutdown(command()) == "answer"
    assert fake.closed_on is not None and fake.closed_on.is_closed(), (
        "closed on the loop that owned it, before it closed"
    )
    assert pi_service._service is None and pi_service._service_loop is None

    failing = _Service()

    async def refused() -> None:
        pi_service._service, pi_service._service_loop = failing, asyncio.get_running_loop()
        raise RuntimeError("the vendor said no")

    with pytest.raises(RuntimeError, match="said no"):
        pi_service.run_then_shutdown(refused())
    assert failing.closed_on is not None, "a failure ends the service too"
