"""The model is per conversation, and one conversation's switch is its own.

Five rules, in the order a user would state them:

1. different sessions can be on different models;
2. switching one session does not move another;
3. a new session starts on the configured default, not on whatever the last
   session switched to;
4. a subsystem with a model *and credentials* of its own uses them; without
   both it follows the model of the conversation it is running under;
5. a switch that arrives while a turn is running takes effect on the next
   turn, not in the middle of this one.

Rule 5 is not a mechanism here, it is a consequence: ``run_turn`` resolves the
session's binding once and holds it in a context var for the whole turn tree,
so a switch landing mid-turn is simply not visible to that turn. The same
context copy is what makes a detached subagent finish on the model it was
spawned under.
"""

from __future__ import annotations

import asyncio

import pytest

from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.agent.loop.main import AgentLoop
from opendde_harness.config.features import SkillForgeConfig
from opendde_harness.providers import model_id
from opendde_harness.providers.base import LLMProvider, LLMResponse
from opendde_harness.providers.binding import ModelBinding, active_binding
from opendde_harness.providers.pool import ProviderPool
from opendde_harness.spine import ChatType, Notice, NoticeKind, Origin, Source, TurnRequest
from opendde_harness.token_wise.base import CURRENT_SESSION_KEY
from tests._config import config as build_config
from tests._config import declared, keyed


class _Provider(LLMProvider):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def get_default_model(self) -> str:
        return f"{self.name}/default"

    async def chat(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        return LLMResponse(content="ok", finish_reason="stop")


def _loop(tmp_path, **kwargs) -> AgentLoop:
    return AgentLoop(
        _Provider("boot"), tmp_path, AgentLoopSettings(model="boot/model", skill_forge=SkillForgeConfig()), **kwargs
    )


def _req(session_key: str | None, channel: str = "tui", chat_id: str = "default") -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel=channel, chat_id=chat_id, sender_id="user", chat_type=ChatType.DM),
        text="hi",
        conversation=session_key,
    )


def _binding(name: str, model: str) -> ModelBinding:
    return ModelBinding(_Provider(name), model)


async def _run(loop: AgentLoop, session_key: str, body) -> object:
    loop._run_turn = body
    return await loop.run_turn(_req(session_key), None, None)


# ---------------------------------------------------------------------------
# 1 + 2 + 3: scope
# ---------------------------------------------------------------------------


def test_a_session_without_a_switch_is_on_the_default(tmp_path) -> None:
    loop = _loop(tmp_path)
    assert loop.session_model("tui:a") == "boot/model"
    assert loop.binding_for_session("tui:a") is loop.default_binding


def test_two_sessions_can_be_on_two_models(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop.set_session_binding("tui:a", _binding("prov-a", "vendor-a/model"))
    loop.set_session_binding("tui:b", _binding("prov-b", "vendor-b/model"))

    assert loop.session_model("tui:a") == "vendor-a/model"
    assert loop.session_model("tui:b") == "vendor-b/model"


def test_switching_one_session_leaves_the_others_alone(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop.set_session_binding("tui:a", _binding("prov-a", "vendor-a/model"))

    assert loop.session_model("tui:b") == "boot/model", "an untouched session stays on the default"
    assert loop.default_binding.model == "boot/model", "a session switch is not a default change"
    assert loop.session_model("tui:fresh") == "boot/model", "a session switch is not sticky"


def test_changing_the_default_moves_only_sessions_that_never_switched(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop.set_session_binding("tui:pinned", _binding("prov-a", "vendor-a/model"))

    loop.set_default_binding(_binding("prov-new", "vendor-new/model"))

    assert loop.session_model("tui:pinned") == "vendor-a/model"
    assert loop.session_model("tui:drifting") == "vendor-new/model"
    assert loop.subagents._fallback.model == "vendor-new/model", "out-of-turn fallbacks follow the default"


def test_dropping_a_session_override_returns_it_to_the_default(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop.set_session_binding("tui:a", _binding("prov-a", "vendor-a/model"))
    loop.clear_session_binding("tui:a")

    assert loop.session_model("tui:a") == "boot/model"
    assert not loop.has_session_binding("tui:a")


def test_a_switch_forgets_capability_verdicts_the_old_provider_answered(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop._vision_ok["boot/model"] = False
    loop.set_session_binding("tui:a", _binding("prov-a", "boot/model"))
    assert loop._vision_ok == {}


# ---------------------------------------------------------------------------
# The turn boundary
# ---------------------------------------------------------------------------


async def test_a_turn_runs_on_its_own_session_model_and_marks_its_session(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop.set_session_binding("tui:a", _binding("prov-a", "vendor-a/model"))

    seen: dict[str, object] = {}

    async def _body(*args, **kwargs):
        # Everything under the turn reads the same pair, including the holders
        # that used to keep a reference of their own.
        seen["loop"] = (loop.provider.name, loop.model)
        seen["subagents"] = (loop.subagents.provider.name, loop.subagents.model)
        seen["session"] = CURRENT_SESSION_KEY.get()
        return "done"

    assert await _run(loop, "tui:a", _body) == "done"
    assert seen["loop"] == ("prov-a", "vendor-a/model")
    assert seen["subagents"] == ("prov-a", "vendor-a/model")
    assert seen["session"] == "tui:a"
    assert CURRENT_SESSION_KEY.get() is None


async def test_outside_a_turn_the_loop_reports_the_default(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop.set_session_binding("tui:a", _binding("prov-a", "vendor-a/model"))

    assert active_binding() is None
    assert loop.model == "boot/model"


async def test_a_turn_is_sized_by_its_own_models_window(tmp_path, monkeypatch) -> None:
    # The window is declared as a row of the model's own provider entry, which
    # is the only place one can be declared: the turn's model has one, the
    # default model has none, and that difference is what this asserts.
    providers = build_config(
        declared("vendor-a", models=[{"id": "model", "contextWindow": 8_000}]),
        model="vendor-a/model",
    ).providers
    loop = AgentLoop(
        _Provider("boot"),
        tmp_path,
        AgentLoopSettings(model="boot/model", skill_forge=SkillForgeConfig(), providers=providers),
    )
    loop.set_session_binding("tui:a", _binding("prov-a", "vendor-a/model"))
    seen: dict[str, object] = {}

    async def _body(*args, **kwargs):
        seen["loop"] = loop.context_window_tokens
        seen["trimmer"] = loop.context_engine.history.context_window_tokens
        return "done"

    await _run(loop, "tui:a", _body)
    assert seen == {"loop": 8_000, "trimmer": 8_000}
    assert loop.context_window_tokens is None, "the default model has no declaration of its own"


def test_a_late_window_answer_for_the_default_model_reaches_the_fallbacks(tmp_path, monkeypatch) -> None:
    """Construction resolves cheaply and may answer unknown; the first call's
    answer must size the out-of-turn holders too, not only the running turn."""
    from opendde_harness.providers.rates import Resolved

    loop = _loop(tmp_path)
    assert loop.context_window_tokens is None

    loop._note_window("boot/model", Resolved(tokens=8_000, source="table"))

    assert loop.context_window_tokens == 8_000
    assert loop.context_engine.history.context_window_tokens == 8_000

    loop._note_window("vendor-other/model", Resolved(tokens=1_000, source="table"))
    assert loop.context_window_tokens == 8_000, "another model's answer is not the default's"


async def test_two_concurrent_turns_each_keep_their_own_model(tmp_path) -> None:
    """Rules 1 and 2 have to hold while both turns are in flight: a user turn
    and a cron turn run at the same time on this loop."""
    loop = _loop(tmp_path)
    loop.set_session_binding("tui:a", _binding("prov-a", "vendor-a/model"))
    loop.set_session_binding("cron:job", _binding("prov-cron", "vendor-cron/model"))

    a_started = asyncio.Event()
    release_a = asyncio.Event()
    seen: dict[str, str] = {}

    async def _body(req, *args, **kwargs):
        key = req.conversation
        if key == "tui:a":
            a_started.set()
            await release_a.wait()
        seen[key] = loop.model
        return key

    loop._run_turn = _body
    a = asyncio.create_task(loop.run_turn(_req("tui:a"), None, None))
    await a_started.wait()
    await loop.run_turn(_req("cron:job"), None, None)
    release_a.set()
    await a

    assert seen["cron:job"] == "vendor-cron/model"
    assert seen["tui:a"] == "vendor-a/model", "the cron turn must not have moved the user turn"


async def test_a_switch_mid_turn_lands_on_the_next_turn(tmp_path) -> None:
    """Rule 5, with no parking involved."""
    loop = _loop(tmp_path)
    loop.set_session_binding("tui:a", _binding("prov-old", "vendor-old/model"))

    started = asyncio.Event()
    release = asyncio.Event()
    during: list[str] = []

    async def _body(req, *args, **kwargs):
        started.set()
        await release.wait()
        during.append(loop.model)
        return "done"

    loop._run_turn = _body
    running = asyncio.create_task(loop.run_turn(_req("tui:a"), None, None))
    await started.wait()

    loop.set_session_binding("tui:a", _binding("prov-new", "vendor-new/model"))
    release.set()
    await running

    assert during == ["vendor-old/model"], "the turn in flight must not move"

    after: list[str] = []

    async def _next(req, *args, **kwargs):
        after.append(loop.model)
        return "done"

    await _run(loop, "tui:a", _next)
    assert after == ["vendor-new/model"]


async def test_the_turn_binding_is_released_when_the_turn_raises(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop.set_session_binding("tui:a", _binding("prov-a", "vendor-a/model"))

    async def _boom(*args, **kwargs):
        raise RuntimeError("turn failed")

    loop._run_turn = _boom
    with pytest.raises(RuntimeError):
        await loop.run_turn(_req("tui:a"), None, None)

    assert active_binding() is None
    assert CURRENT_SESSION_KEY.get() is None
    assert loop.model == "boot/model"


async def test_a_request_without_a_conversation_falls_back_to_its_channel_key(tmp_path) -> None:
    """Channels and cron arrive with no ``conversation``; ``channel:chat_id``
    is the only key they get, so a switch stored under it is the one their
    turn runs on."""
    loop = _loop(tmp_path)
    loop.set_session_binding("whatsapp:12345", _binding("prov-wa", "vendor-wa/model"))
    seen: list[str] = []

    async def _body(*args, **kwargs):
        seen.append(loop.model)
        return "done"

    loop._run_turn = _body
    await loop.run_turn(_req(None, channel="whatsapp", chat_id="12345"), None, None)
    assert seen == ["vendor-wa/model"]


# ---------------------------------------------------------------------------
# Rule 4: subsystems
# ---------------------------------------------------------------------------


async def test_a_spawned_subagent_keeps_its_conversations_model(tmp_path) -> None:
    loop = _loop(tmp_path)
    loop.set_session_binding("tui:a", _binding("prov-a", "vendor-a/model"))

    captured: dict[str, object] = {}

    async def _capture(task_id, task, label, origin, provider, model):
        captured["pair"] = (provider.name, model)

    loop.subagents._run_subagent = _capture
    loop.subagents._gate = asyncio.Semaphore(0)

    async def _body(*args, **kwargs):
        await loop.subagents.spawn("do it", label="it", session_key="tui:a")
        return "done"

    await _run(loop, "tui:a", _body)
    loop.set_session_binding("tui:a", _binding("prov-new", "vendor-new/model"))
    await asyncio.sleep(0)

    assert captured["pair"] == ("prov-a", "vendor-a/model")


# ---------------------------------------------------------------------------
# Restoring a stored choice
# ---------------------------------------------------------------------------


def _restorable_loop(tmp_path, stored: dict[str, dict[str, str]], monkeypatch) -> AgentLoop:
    """A loop whose session store already holds someone's earlier choice.

    The stub provider names itself after the provider the routed config asks
    for, which is the model id's own prefix -- nothing beside the id says who
    serves it.
    """
    monkeypatch.setattr(
        "opendde_harness.cli._helpers.make_provider",
        lambda cfg: _Provider(model_id.provider_of(cfg.agents.defaults.model)),
    )
    cfg = build_config(keyed("anthropic", key="sk-ant"), model="anthropic/claude-opus-4-5")
    loop = _loop(tmp_path, provider_pool=ProviderPool(cfg))
    for key, metadata in stored.items():
        record = loop.sessions.get_or_create(key)
        record.metadata.update(metadata)
        loop.sessions.save(record)
    return loop


async def test_a_channel_turn_restores_the_model_that_session_chose(tmp_path, monkeypatch) -> None:
    """No resume call anywhere: the loop reads the record on first ask, so a
    conversation on any surface comes back on its own model."""
    loop = _restorable_loop(
        tmp_path, {"whatsapp:alice": {"model": "anthropic/claude-sonnet-4-5", "provider": "anthropic"}}, monkeypatch
    )
    seen: list[tuple[str, str]] = []

    async def body(*a, **k):
        seen.append((loop.provider.name, loop.model))
        return None

    await _run(loop, "whatsapp:alice", body)
    assert seen == [("anthropic", "anthropic/claude-sonnet-4-5")]


def test_the_stored_model_is_read_once_per_session(tmp_path, monkeypatch) -> None:
    loop = _restorable_loop(
        tmp_path, {"tui:a": {"model": "anthropic/claude-sonnet-4-5", "provider": "anthropic"}}, monkeypatch
    )
    reads: list[str] = []
    real_peek = loop.sessions.peek

    def counting_peek(key: str):
        reads.append(key)
        return real_peek(key)

    loop.sessions.peek = counting_peek
    for _ in range(3):
        loop.session_model("tui:a")
        loop.session_model("tui:never-switched")

    assert reads == ["tui:a", "tui:never-switched"]


async def test_a_stored_model_that_cannot_be_built_falls_back_and_says_so_once(tmp_path, monkeypatch) -> None:
    """The session runs on the default rather than not running at all, and the
    substitution is not silent: one notice, on the next turn, naming the model
    it could not restore, why, what it is on instead, and how to choose again.
    Once -- a session that has been told does not need telling every turn."""
    loop = _restorable_loop(tmp_path, {"tui:a": {"model": "openai/gpt-5-mini", "provider": "openai"}}, monkeypatch)
    monkeypatch.setattr(
        "opendde_harness.cli._helpers.make_provider",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("no key")),
    )
    events: list[object] = []

    async def emit(event) -> None:
        events.append(event)

    outcome = await loop.run_turn(_req("tui:a"), emit, lambda: [], stream=False)

    assert loop.binding_for_session("tui:a") is loop.default_binding, "the turn ran on the default"
    assert loop.session_model("tui:a") == "boot/model"
    assert not loop.has_session_binding("tui:a")
    assert outcome.text == "ok", "the turn ran; it was not refused"

    notices = [e for e in events if isinstance(e, Notice) and e.kind is NoticeKind.MODEL_FALLBACK]
    assert len(notices) == 1
    said = notices[0].detail or ""
    assert "openai/gpt-5-mini" in said and "no key" in said
    assert "boot/model" in said and "/model <provider>/<model>" in said

    events.clear()
    await loop.run_turn(_req("tui:a"), emit, lambda: [], stream=False)
    assert [e for e in events if isinstance(e, Notice)] == [], "said once, not on every turn"
    await loop.close_mcp()


def test_has_session_binding_sees_a_choice_that_only_exists_on_disk(tmp_path, monkeypatch) -> None:
    loop = _restorable_loop(
        tmp_path, {"tui:a": {"model": "anthropic/claude-sonnet-4-5", "provider": "anthropic"}}, monkeypatch
    )
    assert loop.has_session_binding("tui:a")
    assert not loop.has_session_binding("tui:b")


def test_without_a_pool_a_stored_model_is_left_alone(tmp_path) -> None:
    loop = _loop(tmp_path)
    record = loop.sessions.get_or_create("tui:a")
    record.metadata["model"] = "anthropic/claude-sonnet-4-5"
    loop.sessions.save(record)
    assert loop.session_model("tui:a") == "boot/model"
    assert not loop.has_session_binding("tui:a")


def test_live_providers_are_the_default_and_each_switched_session_once(tmp_path) -> None:
    loop = _loop(tmp_path)
    shared = _binding("prov-a", "vendor-a/model")
    loop.set_session_binding("tui:a", shared)
    loop.set_session_binding("tui:b", shared)
    names = [p.name for p in loop.live_providers()]
    assert names == ["boot", "prov-a"]


def test_a_fork_carries_its_parents_model_in_the_record(tmp_path) -> None:
    loop = _loop(tmp_path)
    parent = loop.sessions.get_or_create("tui:parent")
    parent.metadata.update({"model": "vendor-a/model", "provider": "vendor-a"})
    parent.record({"role": "user", "content": "hi"})
    loop.sessions.save(parent)
    child = loop.sessions.fork("tui:parent")
    assert child is not None
    assert (child.metadata["model"], child.metadata["provider"]) == ("vendor-a/model", "vendor-a")


async def test_the_turn_binding_is_released_when_the_turn_is_cancelled(tmp_path) -> None:
    """Cancellation unwinds the same scope an exception does.

    The binding and the session mark are set around the turn and cleared in a
    ``finally``; a cancelled turn takes that path too. Without it the next turn
    on this loop -- a different conversation, on a different model -- would read
    the cancelled one's binding, and nothing downstream could tell.
    """
    loop = _loop(tmp_path)
    loop.set_session_binding("tui:a", _binding("prov-a", "vendor-a/model"))
    entered = asyncio.Event()

    async def _body(*args, **kwargs):
        entered.set()
        await asyncio.sleep(3600)

    loop._run_turn = _body
    turn = asyncio.create_task(loop.run_turn(_req("tui:a"), None, None))
    await entered.wait()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert active_binding() is None
    assert CURRENT_SESSION_KEY.get() is None
    assert loop.model == "boot/model"


def test_the_window_of_a_model_the_turn_did_not_enter_under_is_that_models_own(tmp_path) -> None:
    """A strategy can send the call to a model other than the turn's binding.

    That call's budget is the *other* model's window, and the only two things
    that can state one are its own declared row and the provider that serves it.
    The active provider's cached figure is neither: it answers for the model the
    turn entered under, so reading it here prices a request against a window it
    was never measured against.
    """

    class Sized(_Provider):
        """A provider whose service reports one window, for its own model."""

        def __init__(self, name: str, window: int) -> None:
            super().__init__(name)
            self._window = window

        def context_window(self) -> int | None:
            return self._window

    providers = build_config(
        declared("vendor-b", models=[{"id": "model", "contextWindow": 32_000}]),
        model="vendor-b/model",
    ).providers
    loop = AgentLoop(
        Sized("boot", 4_000),
        tmp_path,
        AgentLoopSettings(model="boot/model", skill_forge=SkillForgeConfig(), providers=providers),
    )

    assert loop.resolve_window("boot/model").tokens == 4_000, "the active provider answers for its own model"
    assert loop.resolve_window("vendor-b/model").tokens == 32_000, "and a declared row answers for the other"
    # The declaration wins over the service for the model it names, so the
    # 4,000 the active provider reports never reaches another model's budget.
    assert loop.resolve_window("vendor-b/model").source != loop.resolve_window("boot/model").source
