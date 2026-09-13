"""``config.set model`` has two scopes: this conversation, or the default new
ones start on. Both build the provider before anything is persisted.

The value is always ``"<provider>/<model>"``. There is no ``provider`` parameter
beside it any more: the id's prefix is the only thing that names the provider
serving it, so a bare id is refused here rather than sent to whichever vendor a
table guessed."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from opendde_harness.config import loader
from opendde_harness.session.manager import SessionManager
from opendde_harness.tui_rpc.errors import ConfigValidationError
from opendde_harness.tui_rpc.methods import config as config_mod
from opendde_harness.tui_rpc.methods import model as model_methods
from opendde_harness.tui_rpc.methods import session as session_methods
from opendde_harness.tui_rpc.methods.config import _remember_session_model, config_set
from tests._config import declared, keyed, write_config

DEFAULT = "my-vllm/deep-thinker"
OPUS = "anthropic/claude-opus-4-5"


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = write_config(
        tmp_path / "config.json",
        {
            **declared("my-vllm", base_url="http://relay/v1", models=["deep-thinker"]),
            **keyed("anthropic", key="sk-ant"),
        },
        model=DEFAULT,
    )
    monkeypatch.setattr(loader, "_current_config_path", path)
    monkeypatch.setattr(config_mod, "_config_path", lambda: path)
    loader._cache.clear()
    return path


class _FakeLoop:
    """AgentLoop's half of the switch contract."""

    def __init__(self, model: str = DEFAULT, sessions: SessionManager | None = None) -> None:
        self.model = model
        self.provider = SimpleNamespace(name="boot")
        # What the session banner enumerates besides the model.
        self.tools = SimpleNamespace(tool_names=[])
        self.context = SimpleNamespace(skills=SimpleNamespace(list_skills=lambda filter_unavailable=True: []))
        self.defaults: list[object] = []
        self.session_bindings: dict[str, object] = {}
        self.provider_pool = None
        self.sessions = sessions

    def session_model(self, session_key: str) -> str:
        binding = self.session_bindings.get(session_key)
        return binding.model if binding is not None else self.model

    def live_providers(self):
        return [self.provider, *(b.provider for b in self.session_bindings.values())]

    def has_session_binding(self, session_key: str) -> bool:
        return session_key in self.session_bindings

    def set_session_binding(self, session_key: str, binding: object) -> None:
        self.session_bindings[session_key] = binding

    def clear_session_binding(self, session_key: str) -> None:
        self.session_bindings.pop(session_key, None)

    def binding_for_session(self, session_key: str) -> object:
        return self.session_bindings.get(
            session_key, SimpleNamespace(provider=self.provider, model=self.model, provider_name=None)
        )

    def set_default_binding(self, binding: object) -> None:
        self.model = binding.model
        self.provider = binding.provider
        self.defaults.append(binding)


@pytest.fixture
def built(monkeypatch):
    """What ``make_provider`` returned, so the test can see it was handed on."""
    provider = SimpleNamespace(name="new-prov")
    monkeypatch.setattr(config_mod, "make_provider", lambda _cfg: provider)
    return provider


def _on_disk(path) -> dict:
    return json.loads(path.read_text())["agents"]["defaults"]


async def test_a_session_switch_moves_only_the_session_that_asked(config_path, built) -> None:
    loop = _FakeLoop(sessions=SessionManager(config_path.parent))

    result = await config_set(
        {"key": "model", "value": OPUS, "session_id": "tui:a"},
        agent_loop_factory=lambda: loop,
    )

    assert result == {
        "applied": True,
        "previous": DEFAULT,
        "value": OPUS,
        "scope": "session",
        "session_id": "tui:a",
        "applies_to_session": True,
    }
    assert loop.session_bindings["tui:a"].provider is built
    assert loop.session_bindings["tui:a"].model == OPUS
    assert "tui:b" not in loop.session_bindings
    assert loop.defaults == [], "a session switch is not a default change"
    assert _on_disk(config_path)["model"] == DEFAULT, "a new session still starts on the configured default"


async def test_a_session_switch_reports_the_model_it_replaced(config_path, built) -> None:
    loop = _FakeLoop()
    loop.session_bindings["tui:a"] = SimpleNamespace(provider=object(), model="was-on-this")
    result = await config_set({"key": "model", "value": OPUS, "session_id": "tui:a"}, agent_loop_factory=lambda: loop)
    assert result["previous"] == "was-on-this"


async def test_a_bare_id_is_refused_rather_than_routed_by_a_guess(config_path, built) -> None:
    """The hand-typed path. A bare ``claude-opus-4-5`` names nobody, and the
    table that used to guess a vendor for it is what sent one vendor's model to
    another vendor's key. The refusal is the schema's own sentence, so the TUI
    and ``ddeharness`` answer the same input with the same words."""
    from opendde_harness.config.schema import Config

    loop = _FakeLoop()

    with pytest.raises(ConfigValidationError) as refused:
        await config_set(
            {"key": "model", "value": "claude-opus-4-5", "session_id": "tui:a"}, agent_loop_factory=lambda: loop
        )

    assert refused.value.detail == Config.model_construct().explain_unrouted("claude-opus-4-5")
    assert loop.session_bindings == {} and loop.defaults == []
    assert _on_disk(config_path)["model"] == DEFAULT


async def test_a_default_scope_with_a_session_id_still_writes_the_default(config_path, built) -> None:
    """``/model X --default`` sends both, and the scope has to win."""
    loop = _FakeLoop()

    result = await config_set(
        {"key": "model", "value": OPUS, "scope": "default", "session_id": "tui:a"},
        agent_loop_factory=lambda: loop,
    )

    assert result["scope"] == "default"
    assert result["applies_to_session"] is True, "a session that never chose follows the default"
    assert [b.model for b in loop.defaults] == [OPUS]
    assert "tui:a" not in loop.session_bindings
    # The model id is the whole switch: `agents.defaults.provider` is gone, and
    # writing a second field naming the same provider is what let the two
    # disagree.
    assert _on_disk(config_path) == {"model": OPUS}


async def test_a_default_switch_reports_whether_it_moved_the_asking_session(config_path, built) -> None:
    loop = _FakeLoop()
    loop.session_bindings["tui:chose"] = SimpleNamespace(provider="own-prov", model="own/model")
    params = {"key": "model", "value": OPUS, "scope": "default"}

    assert (await config_set({**params, "session_id": "tui:chose"}, agent_loop_factory=lambda: loop))[
        "applies_to_session"
    ] is False
    assert (await config_set({**params, "session_id": "tui:followed"}, agent_loop_factory=lambda: loop))[
        "applies_to_session"
    ] is True


async def test_a_session_scope_without_a_session_id_is_refused_not_widened(config_path, built) -> None:
    loop = _FakeLoop()
    for absent in (None, ""):
        with pytest.raises(ConfigValidationError):
            await config_set(
                {"key": "model", "value": OPUS, "scope": "session", "session_id": absent},
                agent_loop_factory=lambda: loop,
            )
    assert loop.defaults == [] and loop.session_bindings == {}
    assert _on_disk(config_path)["model"] == DEFAULT


async def test_an_unknown_scope_is_rejected(config_path) -> None:
    with pytest.raises(ConfigValidationError):
        await config_set(
            {"key": "model", "value": OPUS, "session_id": "tui:a", "scope": "globl"},
            agent_loop_factory=None,
        )


async def test_a_switch_goes_through_the_pool_when_the_loop_has_one(config_path, built) -> None:
    """The qualified id is all the pool is given: it routes on the prefix, and
    naming the provider a second time is what let a caller redirect a model to a
    credential that does not serve it."""
    asked: list[str] = []
    pooled = SimpleNamespace(provider=SimpleNamespace(name="pooled"), model=OPUS, provider_name="anthropic")

    class _Pool:
        def bind(self, model: str):
            asked.append(model)
            return pooled

    loop = _FakeLoop()
    loop.provider_pool = _Pool()
    await config_set({"key": "model", "value": OPUS, "session_id": "tui:a"}, agent_loop_factory=lambda: loop)
    assert asked == [OPUS]
    assert loop.session_bindings["tui:a"] is pooled


async def test_a_default_switch_with_no_session_omits_applies_to_session(config_path, built) -> None:
    result = await config_set(
        {"key": "model", "value": OPUS, "scope": "default"},
        agent_loop_factory=lambda: _FakeLoop(),
    )
    assert "applies_to_session" not in result


async def test_a_session_switch_without_a_loop_is_not_reported_as_applied(config_path) -> None:
    result = await config_set({"key": "model", "value": OPUS, "session_id": "tui:a"}, agent_loop_factory=None)
    assert result["applied"] is False
    assert _on_disk(config_path)["model"] == DEFAULT


def test_a_model_switch_before_the_first_message_writes_no_session_file(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    _remember_session_model(SimpleNamespace(sessions=sessions), "tui:fresh", "vendor-a/model")

    assert sessions.exists("tui:fresh") is False
    assert sessions.get_or_create("tui:fresh").metadata["model"] == "vendor-a/model"


def test_a_model_switch_on_a_saved_session_is_persisted_at_once(tmp_path) -> None:
    """The model id is all that is stored: it names its provider, so a restore
    rebuilds the binding from the id it already has."""
    sessions = SessionManager(tmp_path)
    sessions.save(sessions.get_or_create("tui:saved"))
    _remember_session_model(SimpleNamespace(sessions=sessions), "tui:saved", "openrouter/vendor-b/model")

    reread = SessionManager(tmp_path).peek("tui:saved")
    assert reread.metadata["model"] == "openrouter/vendor-b/model"


# ---------------------------------------------------------------------------
# The other RPCs that read the session's model
# ---------------------------------------------------------------------------


async def test_model_options_stars_the_sessions_own_model(config_path, monkeypatch) -> None:
    async def entries(provider, model, **kwargs):
        return [{"provider": provider, "model": model}]

    monkeypatch.setattr(model_methods, "_entries", entries)
    loop = _FakeLoop()
    loop.session_bindings["tui:a"] = SimpleNamespace(provider=object(), model=OPUS, provider_name=None)

    own = await model_methods.model_options({"session_id": "tui:a"}, lambda: loop)
    assert (own["model"], own["provider"]) == (OPUS, "anthropic")

    # A gateway's ids carry the upstream vendor in their own name; the prefix in
    # front of it is the provider, and the route the binding was built on agrees.
    loop.session_bindings["tui:g"] = SimpleNamespace(
        provider=object(), model="openrouter/mistralai/mistral-large-latest", provider_name="openrouter"
    )
    routed = await model_methods.model_options({"session_id": "tui:g"}, lambda: loop)
    assert (routed["model"], routed["provider"]) == ("openrouter/mistralai/mistral-large-latest", "openrouter")

    inherited = await model_methods.model_options({"session_id": "tui:b"}, lambda: loop)
    assert (inherited["model"], inherited["provider"]) == (DEFAULT, "my-vllm")


def test_the_selection_is_read_from_one_binding_not_two_accessors() -> None:
    """A switch between two separate reads would pair one binding's model
    with another's route; both halves come from the same object."""
    loop = _FakeLoop()
    loop.session_bindings["tui:a"] = SimpleNamespace(provider=object(), model="openai/gpt-5.1", provider_name="openai")
    loop.session_model = lambda session_key: (_ for _ in ()).throw(AssertionError("read the binding, not the model"))

    assert model_methods._session_selection(lambda: loop, "tui:a") == ("openai/gpt-5.1", "openai")


async def test_the_session_banner_reports_the_sessions_own_model(config_path) -> None:
    from opendde_harness.config.loader import load_config

    loop = _FakeLoop()
    loop.session_bindings["tui:a"] = SimpleNamespace(provider=object(), model=OPUS)
    loop.resolve_window = lambda model, binding=None: SimpleNamespace(tokens=None, source="unknown")

    own = await session_methods._default_session_info(loop, load_config(config_path), "tui:a")
    fresh = await session_methods._default_session_info(loop, load_config(config_path), None)
    assert own["model"] == OPUS
    assert fresh["model"] == DEFAULT


async def test_a_branch_continues_on_its_parents_model_and_delete_clears_it(config_path, tmp_path) -> None:
    sessions = SessionManager(tmp_path / "ws")
    loop = _FakeLoop(sessions=sessions)
    own = SimpleNamespace(provider=object(), model=OPUS)
    loop.session_bindings["tui:parent"] = own
    parent = sessions.get_or_create("tui:parent")
    parent.record({"role": "user", "content": "hi"})
    sessions.save(parent)

    branched = await session_methods.session_branch({"session_id": "tui:parent"}, agent_loop_factory=lambda: loop)
    assert loop.session_bindings[branched["session_id"]] is own

    await session_methods.session_delete({"session_id": "tui:parent"}, agent_loop_factory=lambda: loop)
    assert "tui:parent" not in loop.session_bindings


async def test_a_delete_that_could_not_remove_the_record_keeps_the_binding(config_path, tmp_path, monkeypatch) -> None:
    sessions = SessionManager(tmp_path / "ws")
    loop = _FakeLoop(sessions=sessions)
    loop.session_bindings["tui:kept"] = SimpleNamespace(provider=object(), model=OPUS)
    record = sessions.get_or_create("tui:kept")
    record.record({"role": "user", "content": "hi"})
    sessions.save(record)
    monkeypatch.setattr(sessions, "delete", lambda key: False)

    result = await session_methods.session_delete({"session_id": "tui:kept"}, agent_loop_factory=lambda: loop)

    assert result == {"deleted": None}
    assert "tui:kept" in loop.session_bindings
