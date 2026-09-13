"""The session banner names the thinking level the current model will run with."""

from types import SimpleNamespace

from opendde_harness.providers import messages as msg
from opendde_harness.tui_rpc.methods.session import _default_session_info
from tests import _messages as build
from tests._config import config as build_config
from tests._config import declared, keyed

#: The model a declared provider serves, and the row that describes it.
DECLARED_MODEL = "my-vllm/deep-thinker"

#: pi's Codex login, which is the one route that compacts server-side and the
#: one arrangement that is a plan rather than a metered key.
CODEX_MODEL = "openai-codex/gpt-5.6-luna"
CODEX_PLAN = {"openai-codex": {"login": "oauth"}}


async def test_the_banner_carries_the_models_own_row_effort_over_the_global_default():
    """The level a model is described with lives in its row, because one level
    cannot be right for every model a session switches between."""
    config = build_config(
        declared("my-vllm", models=[{"id": "deep-thinker", "reasoningEffort": "high"}]),
        model=DECLARED_MODEL,
        agents={"defaults": {"model": DECLARED_MODEL, "reasoningEffort": "low"}},
    )

    info = await _default_session_info(None, config)

    assert info["reasoning_effort"] == "high"

    # Nothing declared about this model any more: the global default answers.
    config.providers["my-vllm"].models = ["deep-thinker"]
    assert (await _default_session_info(None, config))["reasoning_effort"] == "low"


def test_resume_hands_the_ui_tool_results_without_the_models_data_markers():
    """The untrusted fence is for the model; a person reading a transcript
    should never see it. Resume used to send it raw, wrappers and all."""
    from opendde_harness.security.trust import wrap_untrusted
    from opendde_harness.tui_rpc.methods.session import _map_to_wire

    stored = [
        build.user("fold 3RRQ"),
        build.tool_result("c1", "bash", wrap_untrusted("PDB 3RRQ first: ATOM      1  N", source="bash")),
        build.tool_result("c2", "write", wrap_untrusted("Successfully wrote 1495 bytes", source="write")),
    ]

    wire = _map_to_wire(stored, "tui:abc")

    assert [row["text"] for row in wire] == [
        "fold 3RRQ",
        "PDB 3RRQ first: ATOM      1  N",
        "Successfully wrote 1495 bytes",
    ]
    assert not any("UNTRUSTED" in row["text"] for row in wire)
    # The wire's own shape is unchanged: a stored ``toolResult`` reaches the UI
    # as the ``tool`` row it has always been, named by the tool that answered.
    assert [row["role"] for row in wire] == ["user", "tool", "tool"]
    assert [row.get("name") for row in wire] == [None, "bash", "write"]


def test_resume_unwraps_a_multimodal_message_and_leaves_ordinary_text_alone():
    from opendde_harness.security.trust import wrap_untrusted
    from opendde_harness.tui_rpc.methods.session import _map_to_wire

    stored = [
        build.user(
            [
                msg.text_block(wrap_untrusted("a pasted page", source="web")),
                msg.image_block("AAA", "image/png"),
            ]
        ),
        build.assistant("the UNTRUSTED marker lives in trust.py"),
    ]

    wire = _map_to_wire(stored, "tui:abc")

    assert wire[0]["text"] == "a pasted page"
    # Prose that merely says the word is not a fence and is not touched.
    assert wire[1]["text"] == "the UNTRUSTED marker lives in trust.py"


def _info_for(model: str, providers: dict | None = None, **overrides):
    """The init bundle a session on ``model`` would be given.

    With no ``providers`` the model's own provider is configured with a key,
    which is the metered arrangement: a plan is a ``login`` on the entry, and
    only a test that writes one gets one.
    """
    import asyncio

    from opendde_harness.tui_rpc.methods.session import _default_session_info

    return asyncio.run(_default_session_info(None, build_config(providers, model=model, **overrides)))


def test_a_plan_billed_provider_is_reported_as_a_subscription():
    """The footer says `(sub)` because no per-token price describes a plan.
    Read from the provider's own entry: what a plan covers is the account, and
    only the account's entry says whether this is one."""
    assert _info_for(CODEX_MODEL, CODEX_PLAN)["subscription"] is True
    assert _info_for("anthropic/claude-opus-4-8", {"anthropic": {"login": "oauth"}})["subscription"] is True


def test_a_metered_provider_is_not_a_subscription_however_it_signs_in():
    # The entry decides, not the vendor: anthropic offers a sign-in, and the
    # same provider reached with a key is billed per token. A provider-name list
    # would get this wrong.
    assert _info_for("anthropic/claude-opus-4-8", keyed("anthropic"))["subscription"] is False
    assert _info_for("openai/gpt-5")["subscription"] is False


def test_server_side_compaction_counts_as_auto_compaction():
    assert _info_for(CODEX_MODEL, CODEX_PLAN)["auto_compact"] is True


def test_a_zero_ratio_turns_server_side_compaction_off():
    # 0 disables the trigger, so nothing compacts server-side however capable
    # the backend is.
    info = _info_for(CODEX_MODEL, CODEX_PLAN, context={"serverCompactRatio": 0})

    assert info["auto_compact"] is False


def test_a_running_context_engine_counts_too():
    """Its history selector fits the session to the window on every turn, so a
    loop that has one never runs out of context whatever the backend does."""
    from types import SimpleNamespace

    from opendde_harness.tui_rpc.methods.session import _auto_compaction

    # Anthropic, because the question is whether the engine counts: a provider
    # the model layer compacts server-side (the Codex login) would answer yes on
    # the first mechanism and prove nothing about this one.
    config = build_config(model="anthropic/claude-sonnet-4-6")
    loop = SimpleNamespace(
        provider=SimpleNamespace(compaction_provider=""),
        context_engine=SimpleNamespace(),
    )

    assert _auto_compaction(loop, config) is True
    # Nothing built yet and a backend that does not compact: nothing will.
    assert _auto_compaction(None, config) is False


def test_a_plan_is_the_entrys_login_and_nothing_else_about_the_provider():
    """How the credential was obtained is the whole question, and the entry is
    the only thing that answers it. The same provider is a plan under a sign-in
    and metered under a key, so nothing about the vendor's name decides."""
    from opendde_harness.tui_rpc.methods.session import _on_subscription

    signed_in = build_config(CODEX_PLAN, model=CODEX_MODEL)
    with_a_key = build_config(keyed("openai-codex"), model=CODEX_MODEL)

    assert _on_subscription(CODEX_MODEL, signed_in) is True
    assert _on_subscription(CODEX_MODEL, with_a_key) is False
    # A model whose provider this config does not configure at all reads as
    # metered: claiming a plan nobody declared would suppress every real figure.
    assert _on_subscription("openai/gpt-5", signed_in) is False


def _rated():
    """A provider stating its model's published rates, as the service reports them."""
    from types import SimpleNamespace

    from opendde_harness.providers.rates import ListRates

    rates = ListRates(input=1.25e-6, output=10e-6, cache_read=0.125e-6, cache_write=1.25e-6)
    return SimpleNamespace(list_rates=lambda: rates)


def test_a_call_is_priced_at_list_even_when_a_plan_covers_it():
    """`cost_usd` is None on a plan -- the subscription is the price -- so a
    status line has nothing to show. The list figure is what the tokens are
    worth, which is what pi prints beside its `(sub)`.

    The backend reports a figure for the call even on a plan; it is the plan
    that makes it the wrong number to show, not the absence of one. The
    providers map is what says which this is, and is handed in for that reason.
    """
    from types import SimpleNamespace

    from opendde_harness.agent.loop.main import AgentLoop

    response = SimpleNamespace(
        usage={"prompt_tokens": 8_600, "completion_tokens": 7, "total_tokens": 8_607, "cost": 0.011},
        model=CODEX_MODEL,
    )
    plan = build_config(CODEX_PLAN, model=CODEX_MODEL).providers
    snapshot = AgentLoop._build_usage_snapshot(response, CODEX_MODEL, "tui:abc", _rated(), plan)

    assert snapshot.estimated_cost_usd is None
    assert snapshot.list_cost_usd is not None
    assert snapshot.list_cost_usd > 0


def test_a_metered_call_is_priced_both_ways():
    from types import SimpleNamespace

    from opendde_harness.agent.loop.main import AgentLoop

    response = SimpleNamespace(
        usage={"prompt_tokens": 1_000, "completion_tokens": 100, "total_tokens": 1_100, "cost": 0.00225},
        model="openai/gpt-5",
    )
    metered = build_config(keyed("openai"), model="openai/gpt-5").providers
    snapshot = AgentLoop._build_usage_snapshot(response, "openai/gpt-5", "tui:abc", _rated(), metered)

    # Both figures exist for a metered provider, and both price the same call:
    # what the backend charged, and what the tokens are worth at list.
    assert snapshot.estimated_cost_usd == 0.00225
    assert snapshot.list_cost_usd is not None


def test_a_call_nobody_priced_carries_no_figure_rather_than_a_zero():
    """A declared provider: pi carries no rates for it and prices the call at
    0, and this project has no table left to answer instead. Unknown, not free."""
    from types import SimpleNamespace

    from opendde_harness.agent.loop.main import AgentLoop

    model = "my-vllm/x"
    response = SimpleNamespace(usage={"prompt_tokens": 1_000, "completion_tokens": 100, "cost": 0.0}, model=model)
    unpriced = SimpleNamespace(list_rates=lambda: None)
    declared_only = build_config(declared("my-vllm", models=["x"]), model=model).providers
    snapshot = AgentLoop._build_usage_snapshot(response, model, "tui:abc", unpriced, declared_only)

    assert snapshot.estimated_cost_usd is None
    assert snapshot.list_cost_usd is None


def test_session_info_answers_for_an_open_session_without_replacing_it():
    import asyncio

    from opendde_harness.tui_rpc.methods.session import session_create, session_info

    created = asyncio.run(session_create({}))
    again = asyncio.run(session_info({"session_id": created["session_id"]}))

    assert again["info"] == created["info"]
    assert set(again) == {"info"}


class _Binding:
    """What ``binding_for_session`` returns, minus building a provider."""

    def __init__(self, model: str, provider_name: str, compaction_provider: str = ""):
        self.model = model
        self.provider_name = provider_name
        self.provider = SimpleNamespace(compaction_provider=compaction_provider)


class _Loop:
    """An AgentLoop's answers about its sessions, without one."""

    def __init__(self, bindings: dict, default: _Binding):
        self._bindings = bindings
        self._default = default
        self.context_engine = SimpleNamespace()
        self.provider = default.provider

    def resolve_window(self, _model: str, _binding=None):
        return SimpleNamespace(tokens=272_000, source="vendor")

    # The rest of what the bundle asks a loop for; none of it is under test
    # here, and a real loop would need a provider built to answer.
    tools = SimpleNamespace(tool_names=[])
    context = SimpleNamespace(skills=SimpleNamespace(list_skills=lambda **_kwargs: []))

    def binding_for_session(self, session_key: str) -> _Binding:
        return self._bindings.get(session_key, self._default)

    def session_model(self, session_key: str) -> str:
        return self.binding_for_session(session_key).model


def _codex_default_config():
    # The configured default is on a plan, so a session that reads the default
    # rather than its own binding reports `(sub)` -- which is exactly what made
    # a switched session's answer wrong.
    return build_config(CODEX_PLAN, model=CODEX_MODEL)


def test_a_switched_session_reports_its_own_billing_not_the_defaults():
    import asyncio

    from opendde_harness.tui_rpc.methods.session import _default_session_info

    codex = _Binding(CODEX_MODEL, "openai-codex", "openai-codex")
    metered = _Binding("openai/gpt-5", "openai")
    loop = _Loop({"tui:metered": metered}, codex)

    info = asyncio.run(_default_session_info(loop, _codex_default_config(), "tui:metered"))

    assert info["model"] == "openai/gpt-5"
    assert info["provider"] == "openai"
    # The default is a plan; this session is not on it.
    assert info["subscription"] is False


def test_a_switched_session_reports_its_own_server_side_compaction_not_the_defaults():
    """The bound provider says whether *it* compacts server-side, and is
    believed: falling back to the configured default behind its answer reported
    that mechanism for a session which had switched away from it.

    Asked of the mechanism directly, with no loop, because a running context
    engine answers yes on the other one and would mask this.
    """
    from opendde_harness.tui_rpc.methods.session import _auto_compaction

    codex = _Binding(CODEX_MODEL, "openai-codex", "openai-codex")
    metered = _Binding("openai/gpt-5", "openai")
    config = _codex_default_config()

    assert _auto_compaction(None, config, metered) is False, "this session's backend compacts nothing"
    assert _auto_compaction(None, config, codex) is True, "and the one it switched from does"
    assert _auto_compaction(None, config) is True, "the default answers only when no binding does"


def test_the_default_still_answers_for_a_session_that_never_switched():
    import asyncio

    from opendde_harness.tui_rpc.methods.session import _default_session_info

    codex = _Binding(CODEX_MODEL, "openai-codex", "openai-codex")
    loop = _Loop({}, codex)

    info = asyncio.run(_default_session_info(loop, _codex_default_config(), "tui:fresh"))

    assert info["subscription"] is True
    assert info["auto_compact"] is True


def test_a_resumed_session_reports_what_its_stored_history_holds():
    """The window is full before this process sends anything, and only the
    server can say by how much. Resume used to open every session at 0%."""
    import asyncio

    from opendde_harness.tui_rpc.methods.session import _default_session_info

    codex = _Binding(CODEX_MODEL, "openai-codex", "openai-codex")
    loop = _Loop({}, codex)
    stored = [build.user("fold 3RRQ"), build.assistant("x" * 40_000)]

    info = asyncio.run(_default_session_info(loop, _codex_default_config(), "tui:old", messages=stored))

    assert info["usage"]["context_used"] > 1_000
    assert info["usage"]["context_percent"] > 0
    # And a session with nothing in it still opens at zero.
    empty = asyncio.run(_default_session_info(loop, _codex_default_config(), "tui:new"))
    assert empty["usage"]["context_used"] == 0
    assert empty["usage"]["context_percent"] == 0


def test_resume_and_info_agree_about_how_full_the_window_is(tmp_path, monkeypatch):
    """Two answers about one session. They read the same stored messages, so
    the footer does not jump when a model switch refreshes the facts."""
    import asyncio

    from opendde_harness.session.manager import SessionManager
    from opendde_harness.tui_rpc.methods import session as session_methods
    from opendde_harness.tui_rpc.methods.session import session_info, session_resume

    manager = SessionManager(tmp_path)
    monkeypatch.setattr(session_methods, "_get_or_build_manager", lambda _config: manager)
    session = manager.get_or_create("tui:stored")
    session.record(build.user("fold 3RRQ " * 500))
    session.record(build.assistant("here is the structure " * 500))
    manager.save(session)

    resumed = asyncio.run(session_resume({"session_id": "tui:stored"}))
    asked = asyncio.run(session_info({"session_id": "tui:stored"}))

    assert resumed["info"]["usage"]["context_used"] > 1_000
    assert resumed["info"]["usage"]["context_used"] == asked["info"]["usage"]["context_used"]


def _stored_session(tmp_path, monkeypatch):
    """A session manager holding one populated session, wired into the module."""
    from opendde_harness.session.manager import SessionManager
    from opendde_harness.tui_rpc.methods import session as session_methods

    manager = SessionManager(tmp_path)
    monkeypatch.setattr(session_methods, "_get_or_build_manager", lambda _config: manager)
    session = manager.get_or_create("tui:stored")
    session.record(build.user("fold 3RRQ " * 500))
    session.record(build.assistant("here is the structure " * 500))
    manager.save(session)

    return manager


def test_a_session_still_opens_when_the_estimator_cannot_answer(tmp_path, monkeypatch):
    """A fresh install has no tokenizer cache and, under our own rule, no
    network to fetch one. The window figure is a status bar's nicety; nothing
    about it may stand between a person and their session."""
    import asyncio

    from opendde_harness.tui_rpc.methods.session import session_create, session_info, session_resume
    from opendde_harness.utils import helpers

    _stored_session(tmp_path, monkeypatch)

    def raise_instead(*_args, **_kwargs):
        raise RuntimeError("cl100k_base is not cached; provide its existing cache via TIKTOKEN_CACHE_DIR")

    monkeypatch.setattr(helpers, "tokenizer_is_loaded", lambda: True)
    monkeypatch.setattr(helpers, "estimate_prompt_tokens", raise_instead)

    created = asyncio.run(session_create({}))
    resumed = asyncio.run(session_resume({"session_id": "tui:stored"}))
    asked = asyncio.run(session_info({"session_id": "tui:stored"}))

    for answer in (created, resumed, asked):
        assert answer["info"]["model"]
        # Absent, not wrong: the footer opens at 0.0%, as it did before the
        # baseline existed, and the first turn replaces it.
        assert answer["info"]["usage"]["context_used"] == 0
        assert answer["info"]["usage"]["context_percent"] == 0

    assert resumed["session_id"] == "tui:stored"
    assert len(resumed["messages"]) == 2


def test_sizing_a_history_never_reaches_for_a_vocabulary(tmp_path, monkeypatch):
    """tiktoken downloads its vocabulary on first use. Opening a session is not
    a network call and must not become one, so the tokenizer is asked only when
    it can answer from memory."""
    import asyncio

    from opendde_harness.tui_rpc.methods.session import session_resume
    from opendde_harness.utils import helpers

    _stored_session(tmp_path, monkeypatch)

    def would_fetch(*_args, **_kwargs):
        raise AssertionError("the tokenizer was asked for a vocabulary it does not have")

    monkeypatch.setattr(helpers, "tokenizer_is_loaded", lambda: False)
    monkeypatch.setattr(helpers, "estimate_prompt_tokens", would_fetch)

    resumed = asyncio.run(session_resume({"session_id": "tui:stored"}))

    # Characters over four instead: the rule the TUI's own live estimate uses.
    assert resumed["info"]["usage"]["context_used"] > 1_000


def test_a_loaded_tokenizer_is_used_when_it_costs_nothing(tmp_path, monkeypatch):
    """Characters over four is the fallback, not the answer: it reads a CJK
    history at a quarter of its size, and the owner writes in Chinese."""
    import asyncio

    from opendde_harness.tui_rpc.methods.session import session_resume
    from opendde_harness.utils import helpers

    _stored_session(tmp_path, monkeypatch)
    asked = []

    monkeypatch.setattr(helpers, "tokenizer_is_loaded", lambda: True)
    monkeypatch.setattr(helpers, "estimate_prompt_tokens", lambda messages: asked.append(messages) or 4_242)

    resumed = asyncio.run(session_resume({"session_id": "tui:stored"}))

    assert resumed["info"]["usage"]["context_used"] == 4_242
    assert len(asked) == 1


def test_undo_says_how_much_of_the_window_is_free_again(tmp_path, monkeypatch):
    """The client's last figure came from a call that saw the longer
    conversation. Nothing else would tell it the window had emptied until the
    next turn reported, so the bar sat at a percentage of messages the session
    no longer has."""
    import asyncio

    from opendde_harness.session.manager import SessionManager
    from opendde_harness.tui_rpc.methods import session as session_methods
    from opendde_harness.tui_rpc.methods.session import session_resume, session_undo

    manager = SessionManager(tmp_path)
    monkeypatch.setattr(session_methods, "_get_or_build_manager", lambda _config: manager)
    session = manager.get_or_create("tui:two")
    for _ in range(2):
        session.record(build.user("fold 3RRQ " * 400))
        session.record(build.assistant("here is the structure " * 400))
    manager.save(session)

    before = asyncio.run(session_resume({"session_id": "tui:two"}))["info"]["usage"]["context_used"]
    undone = asyncio.run(session_undo({"session_id": "tui:two"}))

    assert undone["removed"] == 2
    assert 0 < undone["context_used"] < before
    # And what resume would now say about the same session agrees with it.
    after = asyncio.run(session_resume({"session_id": "tui:two"}))["info"]["usage"]["context_used"]
    assert after == undone["context_used"]


def test_an_undo_that_removed_nothing_reports_no_new_size(tmp_path, monkeypatch):
    """Nothing changed, so there is nothing to say, and no reason to walk a
    long history to say it."""
    import asyncio

    from opendde_harness.session.manager import SessionManager
    from opendde_harness.tui_rpc.methods import session as session_methods
    from opendde_harness.tui_rpc.methods.session import session_undo

    manager = SessionManager(tmp_path)
    monkeypatch.setattr(session_methods, "_get_or_build_manager", lambda _config: manager)
    manager.save(manager.get_or_create("tui:empty"))

    undone = asyncio.run(session_undo({"session_id": "tui:empty"}))

    assert undone == {"removed": 0}


class _Compacting:
    """A provider that replays its own marker, as the Codex login does."""

    compaction_provider = "fake-backend"

    def __init__(self, replays: bool = True):
        self._replays = replays

    def replays_compaction(self, marker, model) -> bool:
        return self._replays and marker.get("provider") == self.compaction_provider


class _BoundTo:
    """What ``binding_for_session`` returns, with a provider that compacts."""

    def __init__(self, provider, model: str = CODEX_MODEL):
        self.model = model
        self.provider_name = "openai-codex"
        self.provider = provider


def _compacted_history() -> list[dict]:
    from opendde_harness.providers.base import COMPACTION_KEY

    assert COMPACTION_KEY == "compaction"
    return [
        build.user("old question " * 3_000),
        build.assistant("old answer " * 3_000),
        build.marker({"provider": "fake-backend", "model": CODEX_MODEL, "items": [{"t": "x"}]}),
        build.user("and now?"),
    ]


def test_a_session_past_a_replayable_marker_reports_an_unmeasured_window():
    """The messages before the marker are not what the next call sends: the
    backend replays one opaque item in their place. Summing them reported 7% of
    a window that will never receive them."""
    import asyncio

    from opendde_harness.tui_rpc.methods.session import _baseline_usage

    loop = _Loop({}, _Binding(CODEX_MODEL, "openai-codex", "openai-codex"))
    binding = _BoundTo(_Compacting())

    usage = asyncio.run(_baseline_usage(loop, binding.model, _compacted_history(), binding))

    assert usage["context_used"] is None
    assert usage["context_percent"] is None
    # The window itself is still known; it is the occupancy that is not.
    assert usage["context_max"] == 272_000


def test_undo_past_a_replayable_marker_reports_the_same_unmeasured_window_as_info(tmp_path, monkeypatch):
    """Undo removes an exchange; it does not measure anything.

    The undo response used to size the raw history itself, so the one call that
    knew about the marker said null while the other said 20,005 about the same
    retained session. `/retry` reads the undo response too, so the wrong figure
    reached the footer on both paths.
    """
    import asyncio

    from opendde_harness.session.manager import SessionManager
    from opendde_harness.tui_rpc.methods import session as session_methods
    from opendde_harness.tui_rpc.methods.session import session_info, session_undo

    manager = SessionManager(tmp_path)
    monkeypatch.setattr(session_methods, "_get_or_build_manager", lambda _config: manager)
    session = manager.get_or_create("tui:compacted")
    for message in _compacted_history():
        session.record(message)
    # One more exchange after the marker, which is the one undo takes back.
    session.record(build.user("one more question"))
    session.record(build.assistant("one more answer"))
    manager.save(session)

    binding = _BoundTo(_Compacting())
    loop = _Loop({"tui:compacted": binding}, binding)
    undone = asyncio.run(session_undo({"session_id": "tui:compacted"}, agent_loop_factory=lambda: loop))

    assert undone["removed"] == 2
    # The marker is still in front of the history, so nothing here measures
    # what the next call will send.
    assert undone["context_used"] is None

    asked = asyncio.run(session_info({"session_id": "tui:compacted"}, agent_loop_factory=lambda: loop))

    assert asked["info"]["usage"]["context_used"] is None


def test_undo_still_measures_a_session_whose_marker_this_backend_ignores(tmp_path, monkeypatch):
    """Another provider's marker is inert text, so the history around it is
    exactly what gets sent and undo says how much of it is left."""
    import asyncio

    from opendde_harness.session.manager import SessionManager
    from opendde_harness.tui_rpc.methods import session as session_methods
    from opendde_harness.tui_rpc.methods.session import session_undo

    manager = SessionManager(tmp_path)
    monkeypatch.setattr(session_methods, "_get_or_build_manager", lambda _config: manager)
    session = manager.get_or_create("tui:inert")
    for message in _compacted_history():
        session.record(message)
    session.record(build.user("one more question"))
    session.record(build.assistant("one more answer"))
    manager.save(session)

    binding = _BoundTo(_Compacting(replays=False))
    loop = _Loop({"tui:inert": binding}, binding)
    undone = asyncio.run(session_undo({"session_id": "tui:inert"}, agent_loop_factory=lambda: loop))

    assert undone["removed"] == 2
    assert undone["context_used"] > 1_000


def test_a_marker_this_backend_would_not_replay_leaves_the_history_measured():
    """Another provider's marker is inert text, and the history around it is
    exactly what gets sent."""
    import asyncio

    from opendde_harness.tui_rpc.methods.session import _baseline_usage

    loop = _Loop({}, _Binding(CODEX_MODEL, "openai-codex", "openai-codex"))
    binding = _BoundTo(_Compacting(replays=False))

    usage = asyncio.run(_baseline_usage(loop, binding.model, _compacted_history(), binding))

    assert usage["context_used"] > 1_000
    assert usage["context_percent"] is not None


def test_a_session_with_no_marker_at_all_is_measured_as_before():
    import asyncio

    from opendde_harness.tui_rpc.methods.session import _baseline_usage

    loop = _Loop({}, _Binding(CODEX_MODEL, "openai-codex", "openai-codex"))
    binding = _BoundTo(_Compacting())
    plain = [build.user("a question " * 3_000)]

    usage = asyncio.run(_baseline_usage(loop, binding.model, plain, binding))

    assert usage["context_used"] > 1_000


def test_resume_says_which_turn_was_interrupted_and_what_it_left_uncertain():
    """A resumed transcript that shows an interrupted turn's work without saying
    so invites the user to assume it finished. The marker uses the shape that
    already exists -- an unrecognised role renders as a system line -- so nothing
    on the wire changes shape."""
    from opendde_harness.session.manager import TOOL_STARTED, TURN_STARTED, Session
    from opendde_harness.tui_rpc.methods.session import _interrupted_notes, _map_to_wire

    session = Session(key="tui:abc")
    session.record_lifecycle(TURN_STARTED, turn_id="t1")
    session.record({**build.user("launch it"), "turn_id": "t1"})
    session.record_lifecycle(TOOL_STARTED, turn_id="t1", call_id="c1", name="launch_job")
    session.record({**build.assistant("launching"), "turn_id": "t1"})

    wire = _map_to_wire(session.messages, "tui:abc", interrupted=_interrupted_notes(session))

    assert [m["role"] for m in wire] == ["user", "assistant", "system"]
    assert "interrupted before finishing" in wire[-1]["text"]
    assert "launch_job" in wire[-1]["text"]


def test_resume_leaves_a_finished_turn_unmarked_and_an_old_session_alone():
    """A session stored before the journal existed has no turn ids, so it gets no
    markers at all."""
    from opendde_harness.session.manager import TURN_COMPLETED, TURN_STARTED, Session
    from opendde_harness.tui_rpc.methods.session import _interrupted_notes, _map_to_wire

    done = Session(key="tui:abc")
    done.record_lifecycle(TURN_STARTED, turn_id="t1")
    done.record({**build.user("hello"), "turn_id": "t1"})
    done.record_lifecycle(TURN_COMPLETED, turn_id="t1")
    assert _interrupted_notes(done) == {}

    # A record in the pre-pi shape, as a file written before the switch holds it.
    old = Session(key="tui:old")
    old.record({"role": "user", "content": "from before"})
    wire = _map_to_wire(old.messages, "tui:old", interrupted=_interrupted_notes(old))
    assert [m["role"] for m in wire] == ["user"]
