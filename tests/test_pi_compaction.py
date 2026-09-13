"""Server-side compaction through the model service, offline.

Needs the built bundle: ``cd ui-tui && npm run build`` produces
``ui-tui/dist/model-service.js``. Without it the module skips.

Nothing here reaches a vendor. The scripted providers live inside the service,
and ``faux-codex`` is the one shaped like a Codex model -- the compaction path
gates on that shape, so it is the only offline model that answers a ``compact``
and the only one whose request a replay reaches. Which *real* providers compact
is a separate rule, asserted on its own; the tests that exercise the mechanism
widen it onto the faux ones, which is the only thing they change.
"""

from __future__ import annotations

import copy
import shutil
from contextlib import aclosing
from pathlib import Path
from typing import Any

import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.agent.tools.base import Tool
from opendde_harness.providers.base import COMPACTION_KEY, GenerationSettings
from opendde_harness.providers.messages import text_of
from opendde_harness.providers.model_service import ModelService
from opendde_harness.providers.pi_provider import PI_COMPACTING, PiModelProvider, pi_compaction_provider
from opendde_harness.spine import ChatType, Origin, Source, TurnRequest
from tests import _messages as build

BUNDLE = Path(__file__).resolve().parent.parent / "ui-tui" / "dist" / "model-service.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    not BUNDLE.exists() or NODE is None,
    reason="ui-tui/dist/model-service.js is not built (cd ui-tui && npm run build) or node is missing",
)

#: The scripted model the compaction path recognises, and two that it does not.
CODEX = "faux-codex/echo"
MODEL = "faux/echo"
SLOW = "faux-slow/echo"

#: Stands in for what the backend hands back: one opaque item nobody here reads.
SUMMARY = {"type": "compaction", "encrypted_content": "everything-before-this"}


class Recording(ModelService):
    """The service, plus a copy of every request the provider handed it."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.contexts: list[dict] = []
        self.replays: list[dict | None] = []
        self.compacted: list[dict] = []

    async def stream_with_id(self, provider, model, context, options=None, *, replay=None, retry=None, timeouts=None):
        self.contexts.append(copy.deepcopy(context))
        self.replays.append(copy.deepcopy(replay))
        return await super().stream_with_id(
            provider, model, context, options, replay=replay, retry=retry, timeouts=timeouts
        )

    async def compact(self, provider, model, context, **kwargs):
        self.compacted.append({"context": copy.deepcopy(context), **copy.deepcopy(kwargs)})
        return await super().compact(provider, model, context, **kwargs)


@pytest.fixture
async def service():
    svc = Recording(node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1"})
    await svc.start()
    try:
        yield svc
    finally:
        await svc.close()


@pytest.fixture
def compacting(monkeypatch):
    """Let the offline models compact, and replay what they compacted.

    ``PI_COMPACTING`` is the rule about which of pi's routes have a backend
    that compacts; no faux provider is among them and none is meant to be.
    Widening it is what lets the mechanism be exercised without a vendor. Two of
    them, so a marker one made and the other refuses is refused for naming
    another backend rather than for compacting nothing.
    """
    from opendde_harness.providers import pi_provider

    monkeypatch.setattr(pi_provider, "PI_COMPACTING", PI_COMPACTING | {"faux-codex", "faux-slow"})


def provider(service: ModelService | None, model: str = CODEX, **generation) -> PiModelProvider:
    provider_id, _, model_id = model.partition("/")
    built = PiModelProvider(service, provider_id, model_id)
    if generation:
        built.generation = GenerationSettings(**generation)
    return built


def marker(*items: dict[str, Any], provider_id: str = "faux-codex", model: str = CODEX) -> dict[str, Any]:
    """A session's compaction marker, as ``compact`` writes one."""
    return build.marker(
        {"provider": provider_id, "model": model, "items": list(items)},
        "[Earlier conversation compacted by the model's own backend; replayed on this model.]",
    )


def compacted_history() -> list[dict[str, Any]]:
    """Instructions, a conversation the backend has swallowed, and one question."""
    return [
        build.system("the instructions"),
        build.user("the old question"),
        build.assistant("the old answer"),
        marker(SUMMARY),
        build.user("the new question"),
    ]


async def drain(built: PiModelProvider, messages: list[dict[str, Any]], **kwargs) -> None:
    """Make the request; the answer is not what is under test."""
    async with aclosing(built.chat_stream(messages, **kwargs)) as deltas:
        async for _delta in deltas:
            pass


async def send_first(built: PiModelProvider, messages: list[dict[str, Any]]) -> None:
    """Make the request and abandon it at its first event.

    For the slow provider, where reading the whole scripted answer is twenty
    seconds of nothing this test cares about.
    """
    async with aclosing(built.chat_stream(messages)) as deltas:
        async for _delta in deltas:
            break


# ---------------------------------------------------------------------------
# Which routes compact
# ---------------------------------------------------------------------------


def test_only_the_responses_routes_compact_and_they_name_themselves_pis_way():
    """pi's own rule: the direct OpenAI route and the Codex login, nothing else."""
    assert provider(None, "openai/gpt-5").supports_compaction
    assert provider(None, "openai-codex/gpt-5-codex").supports_compaction
    assert not provider(None, "anthropic/claude-sonnet-5").supports_compaction
    assert not provider(None, CODEX).supports_compaction, "the offline stand-in is not the rule"

    assert provider(None, "openai-codex/gpt-5-codex").compaction_provider == "openai-codex", "pi's id, verbatim"
    assert provider(None, "anthropic/claude-sonnet-5").compaction_provider == "", "no marker, so no name to answer to"
    assert pi_compaction_provider("openai-codex") == "openai-codex", (
        "answered from the id alone, before anything is built"
    )
    assert pi_compaction_provider("anthropic") == "", "a backend that compacts nothing writes no name"


def test_a_marker_is_replayed_on_every_spelling_of_the_model_that_made_it():
    """The loop names a model two ways and has to get one answer.

    A finished response reports the bare id and the configuration the stored
    ``<provider id>/<model id>`` one -- and the marker is written under one and
    asked about under the other. Comparing them literally had the budgeter
    decide a marker would not be replayed while the request replayed it, which
    budgets one history and sends another.
    """
    built = PiModelProvider(None, "openai-codex", "gpt-5-codex")
    written = marker(SUMMARY, provider_id="openai-codex", model="openai-codex/gpt-5-codex")[COMPACTION_KEY]

    for spelling in ("openai-codex/gpt-5-codex", "gpt-5-codex", None):
        assert built.replays_compaction(written, spelling), spelling

    assert not built.replays_compaction(written, "openai-codex/gpt-5.1-codex"), "another model of the same login"
    assert not built.replays_compaction({**written, "provider": "openai"}, None), "another backend that does compact"


# ---------------------------------------------------------------------------
# What a request carries
# ---------------------------------------------------------------------------


async def test_a_request_on_the_same_model_replays_the_marker_and_sends_only_what_followed(service, compacting):
    built = provider(service)

    await drain(built, compacted_history())

    assert service.replays[-1] == {"items": [SUMMARY]}
    sent = service.contexts[-1]
    assert sent["systemPrompt"] == "the instructions", "the instructions survive a cut made in front of them"
    assert [m["role"] for m in sent["messages"]] == ["user"]
    assert sent["messages"][0]["content"] == "the new question"


async def test_the_replayed_items_are_what_the_request_is_built_from(service, compacting):
    """Past our own call: the prompt the service ended up with.

    The scripted provider builds the request pi would have sent and runs the
    replay hook against it, which is the only place offline where a compaction
    can be seen standing in for the history it swallowed.
    """
    built = provider(service)

    await drain(built, compacted_history())

    payload = (await service.debug())["lastPayload"]["payload"]
    assert payload["input"][0] == SUMMARY, "the summary leads the prompt"
    assert payload["input"][-1]["content"][0]["text"] == "the new question"
    assert "the old question" not in str(payload["input"]), "what the summary stands for is not sent beside it"
    assert payload["instructions"] == "the instructions"


async def test_the_same_history_on_another_model_is_sent_whole_and_replays_nothing(service, compacting):
    built = provider(service, SLOW, first_token_timeout=10.0, idle_timeout=10.0)

    await send_first(built, compacted_history())

    assert service.replays[-1] is None, "the marker is another backend's and says nothing here"
    sent = service.contexts[-1]
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "assistant", "user"]
    assert sent["messages"][1]["content"][0]["text"] == "the old answer"


async def test_a_tool_result_whose_call_is_behind_the_boundary_is_dropped(service, compacting):
    """The call is inside the summary from here on, and a lone result is refused."""
    built = provider(service)
    history = [
        build.system("the instructions"),
        build.assistant(calls=[("swallowed", "echo", {})]),
        marker(SUMMARY),
        build.tool_result("swallowed", "echo", "answered before the compaction"),
        build.assistant(calls=[("kept", "echo", {})]),
        build.tool_result("kept", "echo", "answered after it"),
        build.user("the new question"),
    ]

    await drain(built, history)

    sent = service.contexts[-1]["messages"]
    assert [m["role"] for m in sent] == ["assistant", "toolResult", "user"]
    assert [m["toolCallId"] for m in sent if m["role"] == "toolResult"] == ["kept"], "the orphan went with its call"


# ---------------------------------------------------------------------------
# Asking the backend for one
# ---------------------------------------------------------------------------


async def test_the_service_compacts_a_conversation_into_a_marker_this_model_replays(service, compacting):
    built = provider(service)
    history = [
        build.system("the instructions"),
        build.user("a long conversation"),
        build.assistant("a long answer"),
    ]

    written, usage = await built.compact(history, tools=None, model=CODEX)

    assert written["role"] == "assistant"
    assert text_of(written).startswith("[Earlier conversation compacted")
    items = written[COMPACTION_KEY]["items"]
    assert items[-1]["type"] == "compaction" and items[-1]["encrypted_content"].startswith("faux-")
    assert len(items) > 1, "the recent user text is kept in clear beside the opaque item"
    assert built.replays_compaction(written[COMPACTION_KEY], CODEX)
    assert built.replays_compaction(written[COMPACTION_KEY], "echo"), "the spelling a finished response reports"
    assert usage["prompt_tokens"] > 0


async def test_the_compaction_request_carries_the_turns_own_catalogue_and_thinking_level(service, compacting):
    """A summary is asked for under the same terms as the turns it stands for.

    The service reads the catalogue from the context, the same field a stream
    carries it in, so the compacted request is built from the same body rather
    than from a second description of it.
    """
    built = provider(service, CODEX, reasoning_effort="high")
    catalogue = [
        {
            "type": "function",
            "function": {"name": "echo", "description": "echo it back", "parameters": {"type": "object"}},
        }
    ]
    history = [build.system("the instructions"), build.user("a conversation")]

    await built.compact(history, tools=catalogue, model=CODEX)

    asked = service.compacted[-1]
    assert [tool["name"] for tool in asked["context"]["tools"]] == ["echo"], "the turn's own tools"
    assert asked["context"]["systemPrompt"] == "the instructions"
    assert asked["reasoning"] == "high", "the level the turns it replaces were run at"

    built.generation = GenerationSettings(reasoning_effort="off")
    await built.compact(history, tools=catalogue, model=CODEX)

    assert service.compacted[-1]["reasoning"] is None, "pi spells 'off' by leaving the level out"


async def test_a_backend_that_does_not_compact_is_never_asked(service):
    """The gate is in front of the request, not behind a refusal from the wire."""
    built = provider(service)
    assert not built.supports_compaction

    with pytest.raises(Exception, match="does not compact conversations"):
        await built.compact([build.user("hi")])


async def test_compacting_an_already_compacted_history_replays_the_marker_it_extends(service, compacting):
    """The second summary is built on the first, not on the history it replaced."""
    built = provider(service)

    written, _usage = await built.compact(compacted_history(), tools=None, model=CODEX)

    items = written[COMPACTION_KEY]["items"]
    assert items[-1]["encrypted_content"].startswith("faux-")
    assert "the new question" in str(items), "what followed the marker was compacted with it"
    assert "the old question" not in str(items), "what the marker stood for was not sent again in clear"


# ---------------------------------------------------------------------------
# The loop, end to end
# ---------------------------------------------------------------------------


class Echo(Tool):
    """The tool the faux model always calls."""

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
        return f"echoed {text}"


def request(text: str) -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel="cli", chat_id="t", sender_id="user", chat_type=ChatType.DM),
        text=text,
        conversation="cli:t",
    )


async def _emit(_event) -> None:
    pass


async def test_a_turn_over_the_ratio_records_a_marker_and_the_next_turn_replays_it(tmp_path, service, compacting):
    """The whole path: the loop asks after the turn, and the request after it replays.

    The threshold is set so the scripted model's own reported prompt size clears
    it, because the trigger reads what the backend stated and nothing else.
    """
    built = provider(service)
    settings = AgentLoopSettings(model=CODEX, max_iterations=1)
    settings.context.server_compact_ratio = 1e-6
    loop = AgentLoop(built, tmp_path, settings)
    loop.tools.register(Echo())

    await loop.run_turn(request("the first question"), _emit, lambda: [], stream=True)

    stored = loop.sessions.get_or_create("cli:t").messages
    written = next((m for m in stored if COMPACTION_KEY in m), None)
    assert written is not None, "the turn ended with a marker in the session"
    assert written is stored[-1], "filed behind every message of the turn it stands for"
    assert written[COMPACTION_KEY]["model"] == CODEX and written[COMPACTION_KEY]["provider"] == "faux-codex"

    before = len(service.replays)
    await loop.run_turn(request("the second question"), _emit, lambda: [], stream=True)
    await loop.close_mcp()

    replayed = service.replays[before]
    assert replayed is not None, "the next turn replayed the marker"
    assert replayed["items"] == written[COMPACTION_KEY]["items"]
    sent = service.contexts[before]["messages"]
    assert all("the first question" not in str(m.get("content")) for m in sent), "the compacted turn was not re-sent"
    assert "the second question" in str(sent[-1]["content"])
