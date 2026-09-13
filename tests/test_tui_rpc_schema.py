"""Validate the registered handlers and actual notifications, not duplicate DTOs."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft7Validator, ValidationError

from opendde_harness.config import loader
from opendde_harness.session.manager import SessionManager
from opendde_harness.tui_rpc.approval_broker import ApprovalBroker
from opendde_harness.tui_rpc.confirm_broker import ConfirmBroker
from opendde_harness.tui_rpc.dispatcher import Dispatcher
from opendde_harness.tui_rpc.methods import config as config_methods
from opendde_harness.tui_rpc.methods import register_aligned_methods
from opendde_harness.tui_rpc.methods import session as session_methods
from opendde_harness.tui_rpc.methods import setup as setup_methods
from opendde_harness.tui_rpc.question_broker import QuestionBroker
from opendde_harness.tui_rpc.subscriptions import SubscriptionEmitter
from tests import _messages as build
from tests._config import declared, write_config

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "ui-tui/rpc-schema/openrpc.json").read_text())
METHODS = {method["name"]: method for method in SCHEMA["methods"]}


def validate(schema, value):
    # Resolve component refs within the same document, with no remote lookup.
    validator = Draft7Validator({**schema, "components": SCHEMA["components"]})
    validator.validate(json.loads(json.dumps(value)))


#: The provider this config declares, the wire it speaks, and the model it
#: serves. Declared rather than one of pi's own because the picker calls under
#: test write an address and a wire, which only a declared provider carries.
PROVIDER = "my-vllm"
BASE_URL = "http://localhost:1/v1"
API = "openai-completions"
MODEL = "my-vllm/test-model"


@pytest.fixture
async def rpc(tmp_path, monkeypatch):
    config_path = write_config(
        tmp_path / "config.json",
        declared(PROVIDER, base_url=BASE_URL, api=API, models=["test-model"], apiKey="test"),
        model=MODEL,
        agents={"defaults": {"workspace": str(tmp_path), "model": MODEL}},
    )
    monkeypatch.setattr(loader, "_current_config_path", config_path)
    monkeypatch.setattr(config_methods, "_config_path", lambda: config_path)
    monkeypatch.setattr(setup_methods, "_config_path", lambda: config_path)
    manager = SessionManager(tmp_path)
    monkeypatch.setattr(session_methods, "_get_or_build_manager", lambda config: manager)
    monkeypatch.setattr(session_methods, "update_notice", lambda version, wait=0.0: ("99.0.0", "ddeharness upgrade"))
    # Nothing model-side is replaced: the picker reads its own config and
    # whatever the model service can answer, so real readers, writers, provider
    # rows, declared model rows and endpoint redaction all run here and nothing
    # reaches a vendor.
    frames = []

    async def send_frame(frame):
        frames.append(frame)

    emitter = SubscriptionEmitter(send_frame)
    approval = ApprovalBroker(send_frame)
    confirm = ConfirmBroker(send_frame)
    question = QuestionBroker(send_frame)
    dispatcher = Dispatcher()
    register_aligned_methods(
        dispatcher,
        emitter=emitter,
        approval_broker=approval,
        confirm_broker=confirm,
        question_broker=question,
        # The sign-in group's steps are pushed, so it is registered only with a
        # sink to push them to.
        send_frame=send_frame,
    )
    called = set()

    async def call(name, params=None):
        params = {} if params is None else params
        method = METHODS[name]
        validate(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {p["name"]: p["schema"] for p in method["params"]},
                "required": [p["name"] for p in method["params"] if p.get("required")],
            },
            params,
        )
        response = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": name, "params": params})
        assert "error" not in response, response
        validate(method["result"]["schema"], response["result"])
        from opendde_harness.tui_rpc.models import METHOD_MODELS

        if name in METHOD_MODELS:
            METHOD_MODELS[name][1].model_validate(response["result"])
        called.add(name)
        return response["result"]

    yield SimpleNamespace(
        call=call,
        dispatcher=dispatcher,
        manager=manager,
        config_path=config_path,
        frames=frames,
        emitter=emitter,
        approval=approval,
        confirm=confirm,
        question=question,
        called=called,
    )

    # `model.options` asks the model service what each provider serves, which
    # starts one. Closed here rather than left to the interpreter: the child
    # outlives this test's event loop otherwise, and its transport is collected
    # against a loop that is already closed.
    from opendde_harness.providers.pi_service import shutdown_service

    await shutdown_service()


#: The provider the sign-in tests use: one of the six pi actually signs in to.
OAUTH_PROVIDER = "anthropic"

#: pi's own login-method menu, as the model service reports one.
MENU_STEP = {
    "type": "login_prompt",
    "ask": {
        "type": "select",
        "message": "Select OpenAI Codex login method:",
        "options": [
            {"id": "browser", "label": "Browser login (default)"},
            {"id": "device_code", "label": "Device code login (headless)"},
        ],
    },
}


def _fake_login_service(steps):
    """A ``_login_service`` whose flow yields ``steps`` and then a stored grant.

    pi's own flow is not run: it talks to a vendor. What is under test here is
    the wire — the result shape, and the shape of what gets pushed.
    """

    class Service:
        async def login_with_id(self, provider, *, mode=None):
            async def script():
                for step in steps:
                    yield step
                yield {"provider": provider, "type": "oauth"}

            return 1, script()

        async def login_answer(self, login_id, answer):
            return {"answered": True}

        async def abort(self, login_id):
            return None

    async def answer():
        return Service()

    return answer


def test_schema_covers_exactly_registered_methods_and_emitted_names(rpc):
    assert set(rpc.dispatcher.methods()) == set(METHODS)
    # Catch new literal event/notification producers even before a scenario is
    # added below. Scan production wiring too, not only methods/__init__.py.
    event_names = set(SCHEMA["components"]["schemas"]["TurnEvent"]["discriminator"]["mapping"])
    notification_names = {n["name"] for n in SCHEMA["x-notifications"]}
    sources = [*(ROOT / "opendde_harness/tui_rpc").rglob("*.py"), ROOT / "opendde_harness/cli/tui_commands.py"]
    emitted = set()
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=True):
                if not isinstance(key, ast.Constant) or not isinstance(value, ast.Constant):
                    continue
                if key.value == "type" and isinstance(value.value, str):
                    assert value.value in event_names, (path, value.value)
                    emitted.add(value.value)
                if key.value == "method" and isinstance(value.value, str):
                    assert value.value in notification_names, (path, value.value)
    assert "protein_design.progress" in emitted
    for schema in SCHEMA["components"]["schemas"].values():
        Draft7Validator.check_schema(schema)
    for method in METHODS.values():
        Draft7Validator.check_schema(method["result"]["schema"])


async def test_all_registered_handler_responses_match_schema(rpc, monkeypatch):
    call = rpc.call
    assert (await call("system.hello", {"client_version": "0.1.0"}))["session"]["default_channel"] == "tui"
    await call("system.ping")
    assert (await call("system.version"))["schema_version"] == SCHEMA["info"]["version"]
    assert (await call("session.most_recent")) == {"session_id": None}
    assert (await call("session.list")) == {"sessions": []}
    created = await call("session.create")
    sid = created["session_id"]
    assert not rpc.manager.exists(sid)
    assert (await call("session.resume", {"session_id": "tui:unknown"}))["messages"] == []
    assert (await call("session.resume"))["messages"] == []
    assert (await call("session.title", {"session_id": sid}))["title"] is None
    assert (await call("session.title", {"session_id": sid, "title": "contract"}))["pending"]
    session = rpc.manager.get_or_create(sid)
    session.record(build.user("hello"))
    session.record(build.assistant("world"))
    session.record({**build.tool_result("c1", "read", "result"), "context": {"path": "test.txt"}})
    rpc.manager.save(session)
    resumed = await call("session.resume", {"session_id": sid})
    assert [m["text"] for m in resumed["messages"]] == ["hello", "world", "result"]
    assert (await call("session.most_recent"))["session_id"] == sid
    # The same bundle create and resume return, for a session already open.
    assert (await call("session.info", {"session_id": sid}))["info"] == resumed["info"]
    lazy_info = (await call("session.info"))["info"]
    assert lazy_info["lazy"] is True
    # Nothing to ask: no servers, no prompt to measure. The version's release
    # date comes from the changelog, not from the agent, so it is there either way.
    assert lazy_info["mcp_servers"] == []
    assert lazy_info["release_date"] == session_methods._OPENDDE_HARNESS_RELEASE_DATE
    assert (await call("session.list", {"limit": 1}))["sessions"][0]["id"] == sid
    await call("session.title", {"session_id": sid, "title": "saved"})
    await call("session.close", {"session_id": sid})
    await call("session.close")
    child = await call("session.branch", {"session_id": sid, "name": "child"})
    assert child["message_count"] == 3
    assert (await call("session.branch", {"session_id": "tui:unknown"}))["session_id"] is None
    exported = await call("session.export", {"session_id": sid})
    assert Path(exported["path"]).is_file()
    assert not (await call("session.export"))["exported"]
    # The AGENTS.md / ODH.md files this session would send. What the search
    # finds depends on where pytest was launched, so the contract asserted here
    # is the part that does not: a toggle naming nothing changes nothing, and
    # says so rather than reporting a success the next turn would contradict.
    listed = await call("session.instructions", {"session_id": sid})
    assert listed["cwd"] and listed["error"] is None and listed["changed"] is None
    missing = await call("session.instructions", {"action": "off", "path": "no-such-file.md", "session_id": sid})
    assert missing["changed"] is None and "no-such-file.md" in missing["error"]
    assert [f["enabled"] for f in missing["files"]] == [f["enabled"] for f in listed["files"]]
    undone = await call("session.undo", {"session_id": sid})
    assert undone["removed"] == 3
    # And what the conversation holds now it is empty, so a status line can
    # stop reporting the window the removed exchange filled.
    assert undone["context_used"] == 0
    await call("session.clear", {"session_id": sid})
    assert (await call("session.delete", {"session_id": sid}))["deleted"] == sid
    assert (await call("session.delete", {"session_id": sid}))["deleted"] is None
    for name in ["session.title", "session.branch", "session.clear", "session.undo", "session.delete"]:
        await call(name)
    values = {
        "tui.theme": "dark",
        "tui.show_token_usage": False,
    }
    assert set((await call("config.get"))["config"]) == set(values)
    for key, value in values.items():
        assert (await call("config.set", {"key": key, "value": value}))["previous"] is None
        assert (await call("config.set", {"key": key, "value": value}))["previous"] == value
        section, field = key.split(".")
        assert getattr(getattr(loader.load_config(rpc.config_path), section), field) == value
    assert (await call("config.get", {"keys": ["unknown", "tui.theme"]}))["config"] == {"tui.theme": "dark"}
    assert (await call("config.get", {"keys": []}))["config"] == {}
    await call("config.set", {"key": "model", "value": MODEL})
    await call("config.set", {"key": "model", "value": MODEL, "session_id": "tui:offline"})
    assert (await call("reload.mcp")) == {"ok": True, "reloaded": 0, "tools_changed": False}
    await call("terminal.resize", {"cols": 100, "rows": 30})
    await call("setup.status")
    await call("cli.dispatch", {"argv": ["status"], "width": 100})
    await call("session.status", {"session_id": child["session_id"]})
    await call("slash.exec", {"command": "status"})
    await call("slash.exec", {"command": "no-such-command"})
    # Count/catalog DB stays temporary; Typer reflection is real.
    from opendde_harness.tui_rpc.methods import commands

    monkeypatch.setattr(commands, "_compute_skill_count", lambda: (0, "temporary empty skill store"))
    await call("commands.catalog")
    assert (await call("complete.path")) == {"items": []}
    assert (await call("complete.slash")) == {"items": [], "replace_from": 1}
    await call("model.options", {"include_catalog": False})
    await call("model.options", {"slug": PROVIDER})
    await call("model.save_key", {"slug": PROVIDER, "api_key": "test", "base_url": BASE_URL, "api": API})
    # An id pi does not ship, so the picker's form is what declares it; the wire
    # is not a parameter, because Chat Completions is the one it declares.
    await call(
        "model.declare_provider",
        {"provider": "declared-by-the-picker", "base_url": BASE_URL, "model": "test-model", "api_key": "test"},
    )
    # A sign-in, with pi's own flow faked: the real one talks to a vendor. No
    # step is scripted, so nothing is pushed here -- the pushed shape has its own
    # test below, and `frames` is asserted whole further down.
    from opendde_harness.tui_rpc.methods import model as model_methods

    monkeypatch.setattr(model_methods, "_login_service", _fake_login_service([]))
    login = await call("model.login", {"provider": OAUTH_PROVIDER})
    assert login["provider"]["slug"] == OAUTH_PROVIDER
    # The flow has ended, so neither answers anything any more.
    assert (await call("model.login_answer", {"login_id": login["login_id"], "answer": "browser"}))["answered"] is False
    assert (await call("model.login_cancel", {"login_id": login["login_id"]}))["cancelled"] is False
    # pi's scoped models: saved, read back, and cleared.
    assert (await call("model.scope", {"models": [MODEL], "write": True})) == {"models": [MODEL]}
    assert (await call("model.scope")) == {"models": [MODEL]}
    assert (await call("model.scope", {"models": None, "write": True})) == {"models": None}
    # A refresh asks every declared endpoint for its list; the two declared here
    # are not reachable, and each is reported by id with its sentence.
    errors = (await call("model.options", {"refresh": True, "include_catalog": False}))["refresh_errors"]
    assert set(errors) == {PROVIDER, "declared-by-the-picker"}
    assert all(isinstance(text, str) and text for text in errors.values())
    await call("model.overlay", {"field": "reasoning_effort", "value": "high"})
    await call("model.overlay", {"field": "reasoning_effort", "value": "default"})
    await call("model.overlay", {"field": "context_window", "value": "128k"})
    # pi's logout: the declared provider's key goes, its declaration stays.
    assert (await call("model.logout", {"slug": "declared-by-the-picker"})) == {"forgotten": True}
    assert (await call("model.logout", {"slug": "declared-by-the-picker"})) == {"forgotten": False}
    for name in ["approval.respond", "confirm.respond", "clarify.respond"]:
        assert (await call(name)) == {"ok": False}
    sub = await call("turn.subscribe", {"session_key": sid})
    try:
        assert (await call("turn.send", {"session_key": sid, "content": "hello"}))["accepted"]
        assert not (await call("turn.cancel", {"session_key": sid}))["cancelled"]
        await asyncio.sleep(0.05)
        assert {f["params"]["event"]["type"] for f in rpc.frames} == {"message.start", "error"}
        for frame in rpc.frames:
            validate({"$ref": "#/components/schemas/ServerNotification"}, frame)
    finally:
        await call("turn.unsubscribe", sub)
    assert rpc.called == set(METHODS), set(METHODS) - rpc.called


@pytest.mark.parametrize("choice", ["allow", "deny", "cancelled", "timeout"])
async def test_approval_notifications_and_responses_match_schema(rpc, choice):
    if choice == "timeout":
        rpc.approval._hard_timeout_s = 0.02
    task = asyncio.create_task(
        rpc.approval.await_approval(
            conversation_id="tui:test",
            turn_id="turn",
            tool_call_id="tool",
            command="echo test",
            description="test",
        )
    )
    try:
        await asyncio.sleep(0)
        request = rpc.frames[0]["params"]
        if choice in {"allow", "deny"}:
            assert (
                await rpc.call(
                    "approval.respond",
                    {
                        "approval_id": request["approval_id"],
                        "session_id": "tui:test",
                        "choice": choice,
                    },
                )
            )["ok"]
        elif choice == "cancelled":
            rpc.approval.cancel_all()
        decision = await asyncio.wait_for(task, 1)
        assert decision.approved is (choice == "allow")
        # The reason the caller gets and the reason the close notification
        # announces are the same string.
        assert decision.reason == choice
        assert rpc.frames[-1]["params"]["reason"] == choice
        assert len(rpc.frames) == 2
        for frame in rpc.frames:
            validate({"$ref": "#/components/schemas/ServerNotification"}, frame)
    finally:
        rpc.approval.cancel_all()
        await task


@pytest.mark.parametrize("kind", ["confirm", "clarify", "clarify_cancel"])
async def test_prompt_notifications_and_responses_match_schema(rpc, kind):
    is_confirm = kind == "confirm"
    task = asyncio.create_task(
        rpc.confirm.await_confirm("Proceed?", default=False)
        if is_confirm
        else rpc.question.await_question("tui:test", prompt="Which?", choices=["one", "two"], default="one")
    )
    try:
        await asyncio.sleep(0)
        frame = rpc.frames[0]
        validate({"$ref": "#/components/schemas/ServerNotification"}, frame)
        params = {"request_id": frame["params"]["request_id"], "answer": True if is_confirm else "two"}
        if kind == "clarify_cancel":
            params["cancelled"] = True
        assert (await rpc.call("confirm.respond" if is_confirm else "clarify.respond", params))["ok"]
        assert await asyncio.wait_for(task, 1) == (True if is_confirm else "one" if kind == "clarify_cancel" else "two")
    finally:
        rpc.confirm.cancel_all()
        rpc.question.cancel_all()
        await task


async def test_turn_outlet_and_overflow_events_match_schema(rpc):
    from opendde_harness.spine.events import (
        EpisodeStart,
        Notice,
        NoticeKind,
        Reasoning,
        Text,
        ToolEvent,
        ToolPhase,
        TurnRetry,
        TurnUsage,
    )
    from opendde_harness.tui_rpc.spine import TuiOutlet

    sid = "tui:events"
    sub_id = (await rpc.call("turn.subscribe", {"session_key": sid}))["subscription_id"]
    outlet = TuiOutlet("tui", rpc.emitter)
    try:
        for event in [
            EpisodeStart(0, conversation_id=sid),
            Reasoning("thinking", conversation_id=sid),
            Text("text", conversation_id=sid),
            ToolEvent(ToolPhase.START, "tool", name="read", conversation_id=sid),
            ToolEvent(ToolPhase.COMPLETE, "tool", result_preview="result", conversation_id=sid),
            TurnRetry(1, 3, "retry", True, conversation_id=sid),
            TurnUsage(10, 2, 1, conversation_id=sid),
            # On the wire: the user is not running on the model they chose.
            Notice(NoticeKind.MODEL_FALLBACK, detail="running on the default", conversation_id=sid),
            # Eaten: the transcript already narrates the turn's own progress.
            Notice(NoticeKind.PROGRESS, detail="thinking about it", conversation_id=sid),
            Notice(NoticeKind.TOOL_HINT, detail='read("x")', conversation_id=sid),
        ]:
            await outlet.deliver(event)
        await outlet.send_stream_chunk("events", sid, "delta")
        await outlet.emit_complete(sid, None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        await outlet.emit_complete(
            sid,
            "turn",
            {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cost_usd": None,
                # What a plan-billed call reports instead: no vendor figure,
                # and what the tokens are worth at list beside it.
                "list_cost_usd": 0.056,
                "cache_hit_percent": None,
                "context_used": 12,
                "context_max": 0,
                "context_percent": 0,
                "context_source": "unknown",
                # A compaction ran after this call, so the figure above
                # describes a prompt the session no longer has.
                "context_compacted": True,
            },
        )
        await outlet.emit_error(sid, -32099, "turn_failed", "internal", "test failure")
        await asyncio.sleep(0.05)
        await rpc.emitter._close_overflow(rpc.emitter._by_id[sub_id])
        assert {f["params"]["event"]["type"] for f in rpc.frames} == {
            "episode.start",
            "thinking.delta",
            "token.delta",
            "tool.start",
            "tool.complete",
            "turn.retry",
            "turn.usage",
            "turn.notice",
            "message.complete",
            "error",
        }
        notices = [f["params"]["event"] for f in rpc.frames if f["params"]["event"]["type"] == "turn.notice"]
        assert notices == [
            {"type": "turn.notice", "payload": {"kind": "model_fallback", "text": "running on the default"}}
        ], "one kind reaches the TUI; progress and tool hints do not"
        for frame in rpc.frames:
            validate({"$ref": "#/components/schemas/ServerNotification"}, frame)
    finally:
        await rpc.emitter.unregister(sub_id)


async def test_production_progress_broadcast_matches_schema(rpc, monkeypatch):
    from opendde_harness.cli import tui_commands
    from opendde_harness.plugin.protein_design.core.progress import DesignProgressEvent
    from opendde_harness.tui_rpc import server

    done = asyncio.Event()
    received = asyncio.Event()
    frames = []
    dispatchers = []

    class Transport:
        def __init__(self, *, dispatcher, **kwargs):
            self.dispatcher = dispatcher
            dispatchers.append(dispatcher)

        async def send_frame(self, frame):
            frames.append(frame)
            received.set()

        async def serve_forever(self):
            await self.dispatcher.dispatch(
                {"jsonrpc": "2.0", "id": 1, "method": "system.hello", "params": {"client_version": "0.1.0"}}
            )
            sub = await self.dispatcher.dispatch(
                {"jsonrpc": "2.0", "id": 2, "method": "turn.subscribe", "params": {"session_key": "tui:progress"}}
            )
            try:
                for status in ["started", "progress", "completed", "failed"]:
                    tui_commands._TUI_PROGRESS_SINK(
                        DesignProgressEvent.create(
                            task_id="task",
                            event_type="fold",
                            status=status,
                            input_payload={"sample": 1},
                            error="test failure" if status == "failed" else None,
                        )
                    )
                await asyncio.wait_for(received.wait(), 1)
                done.set()
                await asyncio.Event().wait()
            finally:
                await self.dispatcher.dispatch(
                    {"jsonrpc": "2.0", "id": 3, "method": "turn.unsubscribe", "params": sub["result"]}
                )

    monkeypatch.setattr(server, "RpcServer", Transport)
    monkeypatch.setattr(tui_commands, "_build_tui_agent_loop", lambda: None)
    monkeypatch.setattr(tui_commands, "_TUI_PROGRESS_SINK", None)
    assert await asyncio.wait_for(tui_commands._run_rpc_server_until_done(None, "test", 1, done), 2)
    assert set(dispatchers[0].methods()) == set(METHODS)
    assert len(frames) == 4
    for frame in frames:
        validate({"$ref": "#/components/schemas/ServerNotification"}, frame)
        assert frame["params"]["event"]["payload"]["has_details"]


@pytest.mark.parametrize(
    "name,value",
    [
        ("session.create", {"session": {}}),
        ("session.resume", {"session_id": "tui:test", "info": {}, "last_messages": []}),
        ("config.get", {"config": {"unknown": True}}),
        ("complete.path", {"items": ["file"]}),
        ("reload.mcp", {"ok": True}),
    ],
)
def test_schema_rejects_wrong_response_shapes(name, value):
    with pytest.raises(ValidationError):
        validate(METHODS[name]["result"]["schema"], value)


@pytest.mark.parametrize("model", [MODEL, "openai-codex/gpt-5"])
async def test_live_session_info_matches_schema(rpc, monkeypatch, model):
    config = loader.load_config()
    config.agents.defaults.model = model
    monkeypatch.setattr(session_methods, "load_config", lambda: config)
    loop = SimpleNamespace(
        sessions=rpc.manager,
        session_model=lambda sid: model,
        tools=SimpleNamespace(tool_names=["read"]),
        context=SimpleNamespace(
            skills=SimpleNamespace(
                list_skills=lambda **kwargs: [{"name": "design", "source": "local"}],
            ),
        ),
        resolve_window=lambda model, binding=None: SimpleNamespace(tokens=128000, source="declared"),
        mcp_servers={
            "files": SimpleNamespace(type=None, command="mcp-files", url=None),
            "search": SimpleNamespace(type=None, command=None, url="https://example.test/sse"),
        },
        mcp_tool_counts={"files": 3},
    )
    for name, handler in [
        ("session.create", session_methods.session_create),
        ("session.resume", session_methods.session_resume),
    ]:
        result = await handler({}, agent_loop_factory=lambda: loop)
        validate(METHODS[name]["result"]["schema"], result)
        assert result["info"]["tools"] == {"builtin": ["read"]}
        assert result["info"]["skills"] == {"local": ["design"]}
        assert not result["info"]["lazy"]
        # The prompt is not on the wire in any form: neither its text nor its size.
        assert not [key for key in result["info"] if "system_prompt" in key]
        # Sorted by name, each with the transport the connector would use, and
        # connected only for the server that actually registered tools.
        assert result["info"]["mcp_servers"] == [
            {"name": "files", "transport": "stdio", "connected": True, "tool_count": 3},
            {"name": "search", "transport": "sse", "connected": False, "tool_count": 0},
        ]


async def test_active_turn_cancel_event_matches_schema(rpc, monkeypatch):
    from opendde_harness.tui_rpc.methods import turn

    sid = "tui:cancel"
    cancelled = []

    async def result():
        turn.clear_active(sid)

    monkeypatch.setitem(turn._active_turns, sid, SimpleNamespace(cancel=lambda: cancelled.append(True), result=result))
    subscription = await rpc.call("turn.subscribe", {"session_key": sid})
    try:
        assert (await rpc.call("turn.cancel", {"session_key": sid})) == {"cancelled": True}
        await asyncio.sleep(0.05)
        assert cancelled == [True]
        assert len(rpc.frames) == 1
        validate({"$ref": "#/components/schemas/ServerNotification"}, rpc.frames[0])
        assert rpc.frames[0]["params"]["event"]["payload"]["reason"] == "cancelled_by_client"
    finally:
        await rpc.call("turn.unsubscribe", subscription)


@pytest.mark.asyncio
async def test_config_hot_key_defaults_round_trip(rpc):
    from opendde_harness.config.schema import Config

    expected = {
        "tui.theme": "default",
        "tui.show_token_usage": True,
    }
    assert set(config_methods.CONFIG_WRITABLE_KEYS) == set(expected)
    assert (await rpc.call("config.get"))["config"] == expected
    for key, value in expected.items():
        await rpc.call("config.set", {"key": key, "value": value})
    loaded = loader.load_config(rpc.config_path)
    for key, value in expected.items():
        section, field = key.split(".")
        assert getattr(getattr(loaded, section), field) == value
        assert getattr(getattr(Config(), section), field) == value
    loader.save_config(loaded, rpc.config_path)
    assert (await rpc.call("config.get"))["config"] == expected
    assert loader.load_config(rpc.config_path) == loaded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("tui.theme", "light"),
        ("tui.show_token_usage", False),
    ],
)
async def test_config_hot_keys_replace_serialized_alias(rpc, key, value):
    loaded = loader.load_config(rpc.config_path)
    section, field = key.split(".")
    setattr(getattr(loaded, section), field, value)
    loader.save_config(loaded, rpc.config_path)
    assert (await rpc.call("config.get", {"keys": [key]}))["config"] == {key: value}
    default = config_methods._DEFAULTS[key]
    result = await rpc.call("config.set", {"key": key, "value": default})
    assert result == {"applied": True, "previous": value}
    assert getattr(getattr(loader.load_config(rpc.config_path), section), field) == default


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("tui.theme", ""),
        ("tui.theme", "bad name"),
        ("tui.show_token_usage", 1),
    ],
)
async def test_config_hot_keys_reject_invalid_values_without_writing(rpc, key, value):
    from opendde_harness.tui_rpc.errors import ConfigValidationError

    before = rpc.config_path.read_bytes()
    with pytest.raises(ConfigValidationError):
        await config_methods.config_set({"key": key, "value": value})
    assert rpc.config_path.read_bytes() == before


def test_release_date_reads_the_changelog_and_tolerates_its_absence(monkeypatch, tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## [Unreleased]\n\n## [0.4.2] - 2026-09-10\n\n## [0.4.1] - 2026-08-01\n")
    monkeypatch.setattr(session_methods, "_changelog_path", lambda: changelog)

    assert session_methods._release_date("0.4.2") == "2026-09-10"
    assert session_methods._release_date("v0.4.1") == "2026-08-01"
    # A prerelease is dated by the release it builds on rather than showing nothing.
    assert session_methods._release_date("0.4.2rc1") == "2026-09-10"
    # An undated heading, an unknown version, and no changelog at all: no date.
    assert session_methods._release_date("9.9.9") is None
    assert session_methods._release_date("0.0.0+unknown") is None
    monkeypatch.setattr(session_methods, "_changelog_path", lambda: None)
    assert session_methods._release_date("0.4.2") is None


def test_the_packaged_changelog_dates_this_version():
    """The wheel carries the changelog, so an installed release has a date too."""
    path = session_methods._changelog_path()
    assert path is not None and path.is_file()
    assert session_methods._release_date(session_methods._OPENDDE_HARNESS_VERSION) is not None


@pytest.mark.parametrize(
    "cfg,expected",
    [
        (SimpleNamespace(type="stdio", command=None, url=None), "stdio"),
        (SimpleNamespace(type=None, command="mcp-files", url=None), "stdio"),
        (SimpleNamespace(type=None, command=None, url="https://x.test/sse"), "sse"),
        (SimpleNamespace(type=None, command=None, url="https://x.test/mcp"), "streamableHttp"),
        # What the connector skips, the panel does not name.
        (SimpleNamespace(type="carrier-pigeon", command="x", url=None), None),
        (SimpleNamespace(type=None, command=None, url=None), None),
    ],
)
def test_mcp_transport_matches_what_the_connector_would_use(cfg, expected):
    assert session_methods._mcp_transport(cfg) == expected


def test_mcp_servers_are_empty_without_an_agent_and_skip_unusable_configs():
    assert session_methods._enumerate_mcp_servers(None) == []

    loop = SimpleNamespace(
        mcp_servers={
            "nowhere": SimpleNamespace(type=None, command=None, url=None),
            "files": SimpleNamespace(type="stdio", command="mcp-files", url=None),
        },
        mcp_tool_counts={},
    )
    # Connection is lazy: before the first turn every server is configured and
    # not yet connected.
    assert session_methods._enumerate_mcp_servers(loop) == [
        {"name": "files", "transport": "stdio", "connected": False, "tool_count": 0}
    ]


@pytest.mark.asyncio
async def test_an_error_frame_carries_the_handlers_sentence_beside_its_data():
    """A client shows ``error.data.detail`` and nothing else. A handler that
    also passed structured data (a slug, a field) had its sentence dropped in
    favour of that data, so the model picker's key form said "the gateway
    rejected that value" whatever the gateway had actually said."""
    from opendde_harness.tui_rpc.dispatcher import Dispatcher
    from opendde_harness.tui_rpc.errors import ConfigValidationError

    dispatcher = Dispatcher()

    async def refuse(params):
        raise ConfigValidationError("that id is one of pi's own", data={"slug": "openai", "field": "slug"})

    dispatcher.register("probe.refuse", refuse)
    frame = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 7, "method": "probe.refuse", "params": {}})

    assert frame["error"]["data"] == {"slug": "openai", "field": "slug", "detail": "that id is one of pi's own"}


async def test_login_step_notifications_match_schema(rpc, monkeypatch):
    """Every step of pi's flow is pushed as it arrives, in pi's own shape."""
    from opendde_harness.tui_rpc.methods import model as model_methods

    monkeypatch.setattr(model_methods, "_login_service", _fake_login_service([MENU_STEP]))

    result = await rpc.call("model.login", {"provider": OAUTH_PROVIDER})

    assert [frame["method"] for frame in rpc.frames] == ["login.step"]
    pushed = rpc.frames[0]["params"]
    assert pushed == {"login_id": result["login_id"], "provider": OAUTH_PROVIDER, "step": MENU_STEP["ask"]}
    for frame in rpc.frames:
        validate({"$ref": "#/components/schemas/ServerNotification"}, frame)


async def test_session_info_carries_the_update_notice_and_gives_a_refresh_a_moment(rpc, monkeypatch):
    """The status bar and the transcript say what PyPI has: the notice
    travels as the latest version and the command that installs it. And the
    refresh started at launch is given a moment, so the first launch after a
    release is the one that says so rather than the next."""
    asked: dict = {}

    def notice(version, wait=0.0):
        asked["version"], asked["wait"] = version, wait
        return "99.0.0", "uv tool upgrade opendde-harness"

    monkeypatch.setattr(session_methods, "update_notice", notice)
    config = loader.load_config()
    monkeypatch.setattr(session_methods, "load_config", lambda: config)
    loop = SimpleNamespace(
        sessions=rpc.manager,
        session_model=lambda sid: MODEL,
        tools=SimpleNamespace(tool_names=[]),
        context=SimpleNamespace(skills=SimpleNamespace(list_skills=lambda **kwargs: [])),
        resolve_window=lambda model, binding=None: SimpleNamespace(tokens=None, source="unknown"),
        mcp_servers={},
        mcp_tool_counts={},
    )

    result = await session_methods.session_create({}, agent_loop_factory=lambda: loop)

    assert (result["info"]["update_available"], result["info"]["update_command"]) == (
        "99.0.0",
        "uv tool upgrade opendde-harness",
    )
    assert asked["version"] == result["info"]["version"]
    assert asked["wait"] > 0, "a refresh in flight is waited for, briefly"
