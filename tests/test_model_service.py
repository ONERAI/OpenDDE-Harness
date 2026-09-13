"""The pi-ai model service over stdio, offline, against its faux providers.

Needs the built bundle: ``cd ui-tui && npm run build`` produces
``ui-tui/dist/model-service.js``. Without it the module skips.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from opendde_harness.providers.model_service import ModelService, ModelServiceError, resolve_bundle

BUNDLE = Path(__file__).resolve().parent.parent / "ui-tui" / "dist" / "model-service.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    not BUNDLE.exists() or NODE is None,
    reason="ui-tui/dist/model-service.js is not built (cd ui-tui && npm run build) or node is missing",
)


def context(text: str = "hello") -> dict:
    return {"messages": [{"role": "user", "content": text, "timestamp": 1}]}


@pytest.fixture
async def service():
    svc = ModelService(node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1"})
    await svc.start()
    try:
        yield svc
    finally:
        await svc.close()


async def test_a_stream_yields_every_pi_event_and_ends_with_the_full_message(service):
    events = [e async for e in service.stream("faux", "echo", context("ping"))]
    types = [e["type"] for e in events]
    assert types[0] == "start"
    assert types[-1] == "done"
    for expected in ("thinking_start", "thinking_delta", "thinking_end", "text_start", "text_delta", "text_end"):
        assert expected in types
    assert types.count("text_delta") > 1
    assert "toolcall_end" in types
    assert not any("partial" in e for e in events)

    done = events[-1]
    assert done["reason"] == "toolUse"
    message = done["message"]
    assert message["role"] == "assistant"
    kinds = [block["type"] for block in message["content"]]
    assert kinds == ["thinking", "text", "toolCall"]
    assert message["content"][2]["arguments"] == {"text": "ping"}
    assert "".join(e["delta"] for e in events if e["type"] == "text_delta") == message["content"][1]["text"]


async def test_abort_ends_one_stream_and_leaves_another_running(service):
    slow_id, slow = await service.stream_with_id("faux-slow", "echo", context("slow"))
    first = await anext(slow)
    assert first["type"] == "start"
    fast_id, fast = await service.stream_with_id("faux-slow", "echo", context("other"))
    assert fast_id != slow_id

    await service.abort(slow_id)
    rest = [e async for e in slow]
    assert rest[-1]["type"] == "error"
    assert rest[-1]["reason"] == "aborted"

    other = [e async for e in fast]
    assert other[-1]["type"] == "done"
    assert other[-1]["message"]["content"][2]["arguments"] == {"text": "other"}


async def test_closing_the_iterator_early_aborts_the_stream(service):
    _id, events = await service.stream_with_id("faux-slow", "echo", context("x"))
    await anext(events)
    await events.aclose()
    # The service is still healthy afterwards.
    assert any(m["provider"] == "faux" for m in await service.models())


async def test_models_lists_the_faux_model_beside_the_builtin_catalogue(service):
    models = await service.models()
    faux = [m for m in models if m["provider"] == "faux"]
    assert faux == [
        {
            "provider": "faux",
            "id": "echo",
            "name": "Faux echo",
            "contextWindow": 128000,
            "maxTokens": 16384,
            "reasoning": True,
            "input": ["text", "image"],
            # pi's own rates, per million. A scripted provider has none, which
            # is nobody knowing rather than a model that is free.
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }
    ]
    assert any(m["provider"] == "openai" for m in models)


async def test_an_unknown_model_is_refused_without_a_stream(service):
    with pytest.raises(ModelServiceError) as info:
        async for _ in service.stream("nobody", "nothing", context()):
            pass
    assert info.value.code == "model_not_found"


async def test_a_malformed_line_gets_an_error_and_the_service_keeps_answering(service):
    service._proc.stdin.write(b"this is not json\n")
    await service._proc.stdin.drain()
    service._proc.stdin.write(b'{"id": 77, "method": "abort"}\n')  # a bad request with an id
    await service._proc.stdin.drain()
    # Neither line had a queue waiting; both were answered (id null / id 77)
    # and logged as unknown. The service is still there for the next call.
    events = [e async for e in service.stream("faux", "echo", context("after"))]
    assert events[-1]["type"] == "done"


async def test_a_dead_process_fails_every_pending_stream(service):
    a_id, a = await service.stream_with_id("faux-slow", "echo", context("a"))
    b_id, b = await service.stream_with_id("faux-slow", "echo", context("b"))
    await anext(a)
    await anext(b)
    service._proc.kill()

    async def drain(it):
        return [e async for e in it]

    ended_a, ended_b = await asyncio.wait_for(asyncio.gather(drain(a), drain(b)), 5)
    for ended in (ended_a, ended_b):
        assert ended[-1]["type"] == "error"
        assert "exited" in ended[-1]["error"]["errorMessage"]
    with pytest.raises(ModelServiceError) as info:
        await service.models()
    assert info.value.code == "gone"


# --- configure, auth, login -----------------------------------------------
#
# All offline. The declared provider's address is a port nothing listens on, so
# a stream reaches the connect and fails there -- which is the point: that
# failure is what pi classifies as retryable.

REFUSED = "http://127.0.0.1:9"


def declared(tmp_path, **overrides) -> dict:
    """A `configure` payload naming one unreachable OpenAI-compatible provider."""
    payload = {
        "credentials": str(tmp_path / "pi-auth.json"),
        "apiKeys": {"anthropic": "sk-synthetic-not-a-key"},
        "providers": [
            {
                "id": "lab",
                "name": "Lab",
                "baseUrl": REFUSED,
                "api": "openai-completions",
                "apiKey": "sk-lab-synthetic",
                "models": [{"id": "lab-1", "contextWindow": 8192, "maxTokens": 1024}],
            }
        ],
    }
    payload.update(overrides)
    return payload


async def test_configure_registers_a_declared_provider_and_models_lists_it(service, tmp_path):
    result = await service.configure(declared(tmp_path))
    assert "lab" in result["providers"]
    assert result["models"] > 1

    listed = [m for m in await service.models() if m["provider"] == "lab"]
    assert listed == [
        {
            "provider": "lab",
            "id": "lab-1",
            "name": "lab-1",
            "contextWindow": 8192,
            "maxTokens": 1024,
            "reasoning": False,
            "input": ["text"],
            # `configure` sends no rates for a declared provider, so pi has none.
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }
    ]


async def test_the_catalogue_is_answered_whatever_is_configured(service, tmp_path):
    """``catalog`` reads pi's own built-in rows, before and after a configure.

    The rows are what sizes a declared relay's model, and that question is asked
    while the configuration is still being assembled -- so it must not depend on
    the provider set a configure has replaced.
    """
    before = await service.catalog("deepseek-v4-flash")
    await service.configure(declared(tmp_path))
    after = await service.catalog("deepseek-v4-flash")

    assert before == after and before, "pi carries a row for a vendor model id"
    assert {row["id"] for row in before} == {"deepseek-v4-flash"}
    vendor = next(row for row in before if row["provider"] == "deepseek")
    assert vendor["contextWindow"] > 0 and vendor["maxTokens"] > 0 and vendor["cost"]["input"] > 0
    # And the declared provider's own model is not in pi's catalogue at all.
    assert await service.catalog("lab-1") == []


async def test_configure_replaces_the_previous_set_rather_than_adding_to_it(service, tmp_path):
    await service.configure(declared(tmp_path))
    second = declared(tmp_path)
    second["providers"][0]["id"] = "lab-two"
    result = await service.configure(second)

    assert "lab-two" in result["providers"]
    assert "lab" not in result["providers"]


async def test_a_refused_connection_is_an_error_event_pi_calls_retryable(service, tmp_path):
    await service.configure(declared(tmp_path))
    events = [e async for e in service.stream("lab", "lab-1", context("ping"))]

    assert events[-1]["type"] == "error"
    assert events[-1]["retryable"] is True
    assert events[-1]["error"]["errorMessage"]
    # A transport failure is not one of pi's own: nothing to classify it by.
    assert "code" not in events[-1]


async def test_a_provider_with_no_credential_carries_pis_own_error_code(tmp_path):
    """A built-in provider nothing configured fails as auth, not as a message to match.

    The key is blanked in the child's environment rather than removed from
    this process: pi resolves a built-in provider from its own env var, and a
    developer with one exported would otherwise reach DeepSeek for real.
    """
    svc = ModelService(node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1", "DEEPSEEK_API_KEY": ""})
    await svc.start()
    try:
        await svc.configure({"credentials": str(tmp_path / "pi-auth.json"), "apiKeys": {}, "providers": []})
        model = next(m for m in await svc.models() if m["provider"] == "deepseek")
        events = [e async for e in svc.stream("deepseek", model["id"], context())]
    finally:
        await svc.close()

    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "auth"
    assert events[-1]["retryable"] is False
    assert "deepseek" in events[-1]["error"]["errorMessage"]


def test_the_built_bundle_is_found_without_being_named():
    """`resolve_bundle` reads `node_runtime`, which is the only finder there is.

    The other tests hand the path over, so nothing else here would notice this
    going back to importing names out of the Typer command module -- where
    they no longer exist, so the import alone is the regression.
    """
    from opendde_harness.node_runtime import packaged_dist_dir

    found = resolve_bundle()
    assert found is not None and found.exists()
    # A wheel's copy wins where there is one; this checkout has the source tree's.
    packaged = packaged_dist_dir() / "model-service.js"
    assert found == (packaged if packaged.exists() else BUNDLE)


async def test_a_model_nobody_sized_is_sent_without_a_ceiling_on_an_openai_wire(service, tmp_path):
    """A relay's ``/models`` row often carries nothing but an id. On the
    OpenAI-shaped wires the request then names no ceiling and the server's own
    default applies, which is what every OpenAI-compatible client sends for a
    model it has not sized; it used to be refused outright."""
    payload = declared(tmp_path)
    payload["providers"][0]["models"] = [{"id": "unmeasured"}]
    await service.configure(payload)

    # Sent: the stream reaches the (unreachable) endpoint and ends on its error,
    # not on a refusal of ours.
    events = [e async for e in service.stream("lab", "unmeasured", context())]
    assert events[-1]["type"] == "error"
    assert "no_max_tokens" not in str(events[-1])


async def test_a_model_nobody_sized_is_refused_on_a_wire_that_sends_the_zero(service, tmp_path):
    """anthropic-messages sends a zero ceiling as written and the vendor refuses
    it, so the request is refused here, naming the fix. It runs once the request
    names its own ceiling."""
    payload = declared(tmp_path)
    payload["providers"][0]["api"] = "anthropic-messages"
    payload["providers"][0]["models"] = [{"id": "unmeasured", "api": "anthropic-messages"}]
    await service.configure(payload)

    with pytest.raises(ModelServiceError) as info:
        async for _ in service.stream("lab", "unmeasured", context()):
            pass
    assert info.value.code == "no_max_tokens"
    assert "provider set" in str(info.value)
    events = [e async for e in service.stream("lab", "unmeasured", context(), {"maxTokens": 256})]
    assert events[-1]["type"] == "error"


async def test_auth_lists_what_is_stored_without_any_secret_in_it(service, tmp_path):
    await service.configure(declared(tmp_path))
    report = await service.auth()

    assert {"providerId": "anthropic", "type": "api_key"} in report["stored"]
    anthropic = next(p for p in report["providers"] if p["id"] == "anthropic")
    assert anthropic["configured"] is True
    assert anthropic["type"] == "api_key"
    assert "sk-synthetic-not-a-key" not in json.dumps(report)
    assert "sk-lab-synthetic" not in json.dumps(report)


async def test_logout_deletes_the_stored_credential_and_says_whether_there_was_one(service, tmp_path):
    """The sign-out, run where the store's writes are serialised.

    Against the file it would race the refresh the service runs under its own
    lock, and `forgotten` is what lets a caller say "signed out" rather than
    "nothing was signed in" without reading the file itself.
    """
    store = tmp_path / "pi-auth.json"
    store.write_text(json.dumps({"openai-codex": {"type": "oauth", "access": "a", "refresh": "r", "expires": 1}}))
    await service.configure({"apiKeys": {}, "credentials": str(store), "providers": []})
    assert {"providerId": "openai-codex", "type": "oauth"} in (await service.auth())["stored"]

    assert await service.logout("openai-codex") == {"forgotten": True, "provider": "openai-codex"}
    assert (await service.auth())["stored"] == []
    # Idempotent: nothing stored is not a failure.
    assert await service.logout("openai-codex") == {"forgotten": False, "provider": "openai-codex"}
    assert json.loads(store.read_text()) == {}


async def test_the_codex_sign_in_flow_is_in_the_bundle_rather_than_beside_it(service, tmp_path):
    """pi loads each OAuth flow through a variable specifier, so a bundler
    cannot follow the import into its Node-only callback servers and PKCE. This
    service *is* a bundle, and that import resolved next to the output -- where
    there is nothing -- so every `login` failed on a missing module.

    Reaching pi's own login-method prompt is what proves the module loaded: it is
    the flow's first step, and it comes before anything reaches a vendor. A login
    with no ``mode`` waits there, so this one is cancelled rather than answered --
    either method would start talking to OpenAI.
    """
    await service.configure({"apiKeys": {}, "credentials": str(tmp_path / "pi-auth.json"), "providers": []})

    login_id, steps = await service.login_with_id("openai-codex")
    first = await anext(steps)

    assert first["ask"]["type"] == "select"
    assert {option["id"] for option in first["ask"]["options"]} == {"browser", "device_code"}

    await service.abort(login_id)
    with pytest.raises(ModelServiceError) as info:
        async for _ in steps:
            pass

    assert info.value.code == "login_failed"
    assert "Cannot find module" not in str(info.value)


async def test_a_waiting_login_is_answered_by_a_later_request(service, tmp_path):
    """The menu is answered over the wire, which is what lets somebody at the
    other end of the pipe choose a method.

    Answered with a method pi does not offer: the flow then refuses the answer
    rather than dialling a vendor, which is the whole round trip -- ask, answer,
    act on it -- with nothing leaving this machine.
    """
    await service.configure({"apiKeys": {}, "credentials": str(tmp_path / "pi-auth.json"), "providers": []})

    login_id, steps = await service.login_with_id("openai-codex")
    assert (await anext(steps))["ask"]["type"] == "select"

    assert await service.login_answer(login_id, "carrier-pigeon") == {"answered": True}
    with pytest.raises(ModelServiceError) as info:
        async for _ in steps:
            pass

    assert "Unknown OpenAI Codex login method" in str(info.value)
    # The flow is over, so its id answers nothing: a late answer must not reach
    # whichever login is running next.
    assert await service.login_answer(login_id, "browser") == {"answered": False}


async def test_providers_reports_pis_own_auth_methods_and_labels(service):
    """The material a picker offers a choice with, read off pi's own providers.

    pi keeps both ways in on the provider and the sentence it shows for each;
    this is that, unconverted. Static: nothing is resolved and no vendor is
    asked, which is why the picker can ask on every open.
    """
    rows = {row["id"]: row for row in await service.providers()}

    # A subscription and a key, and pi's own label for the subscription.
    assert rows["xai"]["methods"] == ["oauth", "api_key"]
    assert rows["xai"]["loginLabel"] == "Sign in with SuperGrok or X Premium"
    assert rows["xai"]["keyName"] == "xAI API key"
    assert rows["xai"]["name"] == "xAI"
    # A sign-in and nothing else: pi offers this one no key at all, so a picker
    # asking which way in would be asking about one option.
    assert rows["openai-codex"]["methods"] == ["oauth"]
    assert "keyName" not in rows["openai-codex"]
    # A key and nothing else.
    assert rows["deepseek"]["methods"] == ["api_key"]
    # pi carries no label of its own for this one, so its generic sentence is
    # what a menu shows.
    assert "loginLabel" not in rows["anthropic"]
    assert rows["anthropic"]["methods"] == ["oauth", "api_key"]


async def test_providers_describes_a_declared_provider_as_reached_by_a_key(service, tmp_path):
    """A provider this config declares is pi's too once configured, and pi gives
    it key auth: there is no sign-in to offer for an endpoint of one's own."""
    await service.configure(declared(tmp_path))

    rows = {row["id"]: row for row in await service.providers()}

    assert rows["lab"]["methods"] == ["api_key"]


async def test_a_login_nothing_is_waiting_for_answers_nothing(service):
    assert await service.login_answer(4242, "browser") == {"answered": False}


async def test_a_login_for_a_provider_nobody_serves_is_a_request_error(service, tmp_path):
    await service.configure(declared(tmp_path))
    with pytest.raises(ModelServiceError) as info:
        async for _ in service.login("nobody"):
            pass
    assert info.value.code == "login_failed"
    assert "nobody" in str(info.value)


async def test_the_child_honours_the_proxy_environment(monkeypatch):
    """Node's fetch ignores HTTP(S)_PROXY unless NODE_USE_ENV_PROXY is set; the
    service must be spawned with it, or every proxied vendor fails with
    "fetch failed" while the Python side, which honours the variables, works."""
    seen: dict[str, str] = {}

    async def fake_exec(*args, **kwargs):
        seen.update(kwargs.get("env") or {})
        raise RuntimeError("spawn intercepted")

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    service = ModelService(node="node", bundle=BUNDLE)
    with pytest.raises(RuntimeError, match="spawn intercepted"):
        await service.start()
    assert seen.get("NODE_USE_ENV_PROXY") == "1"


# ---------------------------------------------------------------------------
# Letting go of the child
#
# A service outliving the loop that owns its pipes is worse than a stray
# process: its subprocess transport is finalised later by the garbage
# collector, and a finaliser running after its loop has closed raises inside
# ``__del__`` -- an "Event loop is closed" traceback printed after the last
# line of output, about an object nobody can still reach. These are the three
# ways the child is let go of.
# ---------------------------------------------------------------------------


async def test_closing_twice_is_the_second_call_finding_nothing_to_close():
    """More than one thing ends a service, so the second one must not raise.

    A fixture's teardown, ``pi_service.shutdown_service`` and the process's exit
    hook can all reach the same object.
    """
    svc = ModelService(node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1"})
    await svc.start()
    assert svc.running

    await svc.close()
    assert not svc.running
    assert svc.pid is None

    await svc.close()  # nothing held, nothing to do, nothing raised


async def test_abandoning_signals_the_child_and_holds_nothing_after():
    """``abandon`` is the answer where there is no loop to close pipes on.

    Only a signal goes out -- that needs no loop -- and the references are
    dropped, so the service reads as ended to everything that asks and there is
    no transport left for a finaliser to trip over.
    """
    svc = ModelService(node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1"})
    await svc.start()
    pid = svc.pid
    assert pid is not None

    svc.abandon()

    assert not svc.running
    assert svc.pid is None
    svc.abandon()  # idempotent: there is no second child to signal
    # The child is gone or going; either way this process is no longer waiting
    # on it, which is what the finaliser used to be left holding.
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        await asyncio.sleep(0.05)


async def test_a_service_whose_loop_is_gone_is_abandoned_rather_than_closed(monkeypatch):
    """``shutdown_service`` answers for a service it cannot reach.

    Closing pipes needs the loop that owns them. When that is not the loop
    running now -- a synchronous caller ran its own and let it close -- there is
    nothing to close on, so the child is signalled instead of the call failing.
    """
    from opendde_harness.providers import pi_service

    class Recorder:
        def __init__(self) -> None:
            self.abandoned = False
            self.closed = False

        def abandon(self) -> None:
            self.abandoned = True

        async def close(self) -> None:
            self.closed = True

    stale, dead_loop = Recorder(), asyncio.new_event_loop()
    dead_loop.close()
    monkeypatch.setattr(pi_service, "_service", stale)
    monkeypatch.setattr(pi_service, "_service_loop", dead_loop)

    await pi_service.shutdown_service()

    assert stale.abandoned and not stale.closed
    assert pi_service._service is None, "the module lets go of it either way"

    # And on the loop that does own it, the pipes are closed properly.
    mine = Recorder()
    monkeypatch.setattr(pi_service, "_service", mine)
    monkeypatch.setattr(pi_service, "_service_loop", asyncio.get_running_loop())

    await pi_service.shutdown_service()

    assert mine.closed and not mine.abandoned


async def test_a_service_another_running_loop_owns_is_neither_taken_over_nor_ended(monkeypatch):
    """A worker thread's loop finds the TUI's service and leaves it alone.

    The TUI streams through the child on its own loop while an RPC handler runs
    a synchronous helper on a worker thread. That helper's ``asyncio.run`` used
    to find the service "on another loop", abandon it -- signal the child -- and
    start its own: one ``/logout`` ended whatever turn was streaming. A loop
    that is still running owns its service; nobody else uses it or ends it.
    """
    from opendde_harness.providers import pi_service
    from tests._config import config as build_config

    class Recorder:
        abandoned = False
        closed = False

        def abandon(self) -> None:
            self.abandoned = True

        async def close(self) -> None:
            self.closed = True

    live = Recorder()
    monkeypatch.setattr(pi_service, "_service", live)
    monkeypatch.setattr(pi_service, "_service_loop", asyncio.get_running_loop())
    monkeypatch.setattr(pi_service, "_fingerprint", "x")

    def on_a_worker_thread() -> str:
        async def command() -> str:
            with pytest.raises(RuntimeError, match="still running"):
                await pi_service.get_service(build_config({}))
            await pi_service.shutdown_service()
            return "left alone"

        return asyncio.run(command())

    assert await asyncio.to_thread(on_a_worker_thread) == "left alone"

    assert not live.abandoned and not live.closed
    assert pi_service._service is live, "still this loop's"


def test_the_synchronous_model_listing_ends_the_service_it_started(monkeypatch):
    """``models_for_provider`` runs its own loop, so it closes what it opened.

    The service is process-wide: left running, its pipes belong to a loop that
    has closed, and the next caller on a live loop has to signal it and start
    another.
    """
    from opendde_harness.providers import common_models
    from tests._config import config, keyed

    ended: list[str] = []

    async def rows(_config):
        return [{"provider": "deepseek", "id": "deepseek-chat"}]

    async def shutdown():
        ended.append("shutdown")

    monkeypatch.setattr(common_models, "service_rows", rows)
    monkeypatch.setattr("opendde_harness.providers.pi_service.shutdown_service", shutdown)

    offered = common_models.models_for_provider(config(keyed("deepseek")), "deepseek")

    assert [row["id"] for row in offered] == ["deepseek-chat"]
    assert ended == ["shutdown"], "the loop that started a service ends it"


async def test_a_service_a_test_leaves_running_is_let_go_of_before_the_next_one(tmp_path, monkeypatch):
    """The guard in ``conftest`` has something to do, and this is what.

    Deliberately leaves the process-wide service running: no ``shutdown_service``
    here. What must not happen is the child surviving into the next test with its
    pipes attached to this loop, which closes when this test ends -- the
    autouse ``no_model_service_outlives_its_loop`` fixture is what signals it and
    drops the references. The assertion this test can make is that the service
    really was started, so the fixture's path is reached rather than skipped.
    """
    from opendde_harness.providers import pi_service
    from tests._config import config, declared

    monkeypatch.setattr(pi_service, "token_dir", lambda: tmp_path)
    monkeypatch.setattr(pi_service.ModelService, "__init__", _faux_init(pi_service.ModelService.__init__))

    service = await pi_service.get_service(
        config(declared("faux", base_url="http://127.0.0.1:1/v1", models=["echo"]), model="faux/echo")
    )

    assert service.running, "the process-wide service is up, and this test does not end it"
    assert pi_service._service is service, "the module is holding it for the fixture to let go of"


def _faux_init(real):
    """``ModelService.__init__`` pinned to this suite's node, bundle and faux mode.

    ``get_service`` builds its own service and takes no arguments for any of the
    three, and this suite must never reach a real vendor.
    """

    def init(self, **_kwargs):
        real(self, node=NODE, bundle=BUNDLE, env={"OPENDDE_MODEL_SERVICE_FAUX": "1"})

    return init
