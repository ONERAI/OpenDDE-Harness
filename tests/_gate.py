"""The rig the integration-gate tests share: a real loop over the faux service.

Every gate test drives an :class:`AgentLoop` built by
:func:`opendde_harness.agent.loop.factory.build_agent_loop` from a synthetic
:class:`Config` whose workspace is the test's own temp directory, and answers
its model calls from the model service's scripted providers
(``OPENDDE_MODEL_SERVICE_FAUX=1``). Nothing here reaches a vendor, reads a
credential directory or writes outside ``tmp_path``.

The package-level suites each grew their own copy of the recording service and
the echo tool; the gate needs one rig for all eight scenarios, so it lives here
rather than being spelled eight more times.
"""

from __future__ import annotations

import asyncio
import copy
import shutil
from pathlib import Path
from typing import Any

import pytest

from opendde_harness.agent.loop.factory import build_agent_loop
from opendde_harness.agent.tools.base import Tool
from opendde_harness.config.schema import Config
from opendde_harness.providers.base import GenerationSettings
from opendde_harness.providers.binding import ModelBinding
from opendde_harness.providers.model_service import ModelService
from opendde_harness.providers.pi_provider import PiModelProvider
from opendde_harness.spine import ChatType, Origin, Source, TurnRequest

BUNDLE = Path(__file__).resolve().parent.parent / "ui-tui" / "dist" / "model-service.js"
NODE = shutil.which("node")

#: Every gate module carries this: without the built bundle there is no service.
requires_service = pytest.mark.skipif(
    not BUNDLE.exists() or NODE is None,
    reason="ui-tui/dist/model-service.js is not built (cd ui-tui && npm run build) or node is missing",
)

#: The scripted models. ``faux`` and ``faux-codex`` answer instantly; the latter
#: is shaped like a Codex Responses model, which is what the compaction path
#: gates on. ``faux-flaky`` reads its prompt as a failure script.
MODEL = "faux/echo"
CODEX = "faux-codex/echo"
FLAKY = "faux-flaky/echo"


class Recording(ModelService):
    """The service, plus every request it was handed and a way to hold one.

    ``hold_from`` is how a gate test makes "the next model call never comes
    back" happen at the seam a stalled vendor would sit at: the request is
    recorded, ``held`` is set so the test knows the call is outstanding, and
    nothing is sent until ``release`` is set. A scripted slow provider would do
    the same thing in wall-clock time -- twenty tokens a second against a
    request the loop has wrapped in a runtime-context header is minutes -- so
    the deadline is an event instead of a clock.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.contexts: list[dict] = []
        self.options: list[dict | None] = []
        self.replays: list[dict | None] = []
        self.models_asked: list[tuple[str, str]] = []
        self.compacted: list[dict] = []
        self.hold_from: int | None = None
        self.held = asyncio.Event()
        self.release = asyncio.Event()
        #: Hold every request until this many are outstanding at once, so two
        #: lanes are provably in flight together rather than one after the other.
        self.barrier: int | None = None
        self._reached = asyncio.Event()

    @property
    def streams(self) -> int:
        """How many model calls the service was asked to make."""
        return len(self.contexts)

    async def stream_with_id(self, provider, model, context, options=None, *, replay=None, retry=None, timeouts=None):
        self.contexts.append(copy.deepcopy(context))
        self.options.append(copy.deepcopy(options))
        self.replays.append(copy.deepcopy(replay))
        self.models_asked.append((provider, model))
        if self.barrier is not None:
            if len(self.contexts) >= self.barrier:
                self._reached.set()
            await self._reached.wait()
        if self.hold_from is not None and len(self.contexts) >= self.hold_from:
            self.held.set()
            await self.release.wait()
        return await super().stream_with_id(
            provider, model, context, options, replay=replay, retry=retry, timeouts=timeouts
        )

    async def compact(self, provider, model, context, **kwargs):
        self.compacted.append({"provider": provider, "model": model, "context": copy.deepcopy(context)})
        return await super().compact(provider, model, context, **kwargs)


@pytest.fixture
async def service():
    svc = Recording(node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1"})
    await svc.start()
    try:
        yield svc
    finally:
        svc.release.set()
        svc._reached.set()
        await svc.close()


class Echo(Tool):
    """The tool the scripted model always calls. Changes nothing outside this
    process, so the journal does not record it as started."""

    external_effects = False

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


class Launch(Echo):
    """The same call, declaring that it changed something this process cannot
    undo -- a submitted job, a written file. This is what a cancelled turn has
    to leave a record of.

    Each call reports its own job id, so one exchange can be told from another
    in a session that holds several.
    """

    external_effects = True

    @property
    def name(self) -> str:
        return "echo"

    async def execute(self, text: str = "", **_kwargs) -> str:
        self.seen.append(text)
        return f"job-{len(self.seen)} submitted"


def gate_config(workspace: Path, *, model: str = MODEL, providers: dict[str, Any], **blocks: Any) -> Config:
    """A config whose workspace is the test's own directory.

    ``providers`` is the whole section: the gate declares its scripted
    providers outright (pi ships no ``faux``), which is also where a declared
    context window goes -- a row in the provider's own entry.

    ``memory`` is off, so the loop the gate builds has no memory owner at all.
    Deliberate: a config that says nothing gets the host's own writer, which
    annotates queued turns with a model call of its own, and a gate that counts the
    turn's requests would be counting that too. A gate that wants a memory owner
    passes its own ``memory`` block or its own ``backend=``.
    """
    blocks.setdefault("memory", {"backend": "off"})
    return Config.model_validate(
        {
            "agents": {"defaults": {"model": model, "workspace": str(workspace), **blocks.pop("defaults", {})}},
            "providers": providers,
            **blocks,
        }
    )


def declare(provider: str, *, window: int | None = None, model: str = "echo") -> dict[str, Any]:
    """One declared entry for a scripted provider, optionally declaring the
    window its model holds."""
    row: dict[str, Any] = {"id": model}
    if window is not None:
        row["contextWindow"] = window
    return {
        provider: {
            "baseUrl": "http://127.0.0.1:8000/v1",
            "api": "openai-completions",
            "models": [row],
        }
    }


def bind(service: ModelService, model: str, config: Config | None = None) -> ModelBinding:
    """A binding on the scripted service, as a ``/model`` switch would make one.

    The real :class:`~opendde_harness.providers.pool.ProviderPool` builds its
    provider through the process-wide service, which is not this test's
    service; the pair it would produce is what a gate test needs, so it is
    built here -- carrying the same two things ``make_provider`` copies onto a
    provider it builds, the declared provider rows and the generation defaults.
    Without those the retry budget and the gap deadlines would be this
    constructor's defaults rather than the config's, and a gate test would be
    asserting on settings no configuration named.
    """
    provider_id, _, model_id = model.partition("/")
    built = PiModelProvider(
        service,
        provider_id,
        model_id,
        stored_model=model,
        providers=config.providers if config is not None else None,
    )
    if config is not None:
        defaults = config.agents.defaults
        built.generation = GenerationSettings(
            reasoning_effort=defaults.reasoning_effort,
            first_token_timeout=defaults.llm_first_token_timeout,
            idle_timeout=defaults.llm_idle_timeout,
            retries=defaults.llm_retries,
        )
    return ModelBinding(built, model, provider_name=provider_id)


def loop_for(service: ModelService, config: Config, *, tools: list[Tool] | None = None, **kwargs: Any):
    """The loop the gate drives: built by ``build_agent_loop``, nothing else."""
    binding = bind(service, config.agents.defaults.model, config)
    loop = build_agent_loop(config, provider=binding.provider, **kwargs)
    for tool in tools or ():
        loop.tools.register(tool)
    return loop


def request(text: str, conversation: str = "tui:gate") -> TurnRequest:
    channel, _, chat = conversation.partition(":")
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel=channel, chat_id=chat, sender_id="user", chat_type=ChatType.DM),
        text=text,
        conversation=conversation,
    )


class Events:
    """Collects what a turn emitted, so a test can count one kind of event."""

    def __init__(self) -> None:
        self.seen: list[Any] = []

    async def __call__(self, event: Any) -> None:
        self.seen.append(event)

    def of(self, kind: type) -> list[Any]:
        return [event for event in self.seen if isinstance(event, kind)]

    def kinds(self) -> list[str]:
        return [type(event).__name__ for event in self.seen]


def no_injections() -> list[Any]:
    """The ``drain`` a gate turn runs with: nothing is injected mid-turn."""
    return []


def system_prompt(context: dict[str, Any]) -> str:
    return str(context.get("systemPrompt") or "")


def block_of(prompt: str, header: str) -> str:
    """One ``# Header`` section of a rendered system prefix, verbatim.

    Returns "" when the section is absent, which is a different answer from a
    section that is present and empty.
    """
    start = prompt.find(header)
    if start < 0:
        return ""
    rest = prompt.find("\n# ", start + len(header))
    return prompt[start:] if rest < 0 else prompt[start:rest]


def history_roles(context: dict[str, Any]) -> list[str]:
    return [m.get("role") for m in context.get("messages", [])]


def texts(context: dict[str, Any]) -> str:
    """Everything the request's messages say, as one string to search."""
    return str(context.get("messages"))


__all__ = [
    "BUNDLE",
    "CODEX",
    "Echo",
    "Events",
    "FLAKY",
    "Launch",
    "MODEL",
    "NODE",
    "Recording",
    "bind",
    "block_of",
    "declare",
    "gate_config",
    "history_roles",
    "loop_for",
    "no_injections",
    "request",
    "requires_service",
    "system_prompt",
    "texts",
]
