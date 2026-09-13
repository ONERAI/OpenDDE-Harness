"""``model.overlay`` writes one field of the current model's row and puts it in front of the
live loop; the picker lists what the model service says each configured provider serves.

The picker's own source is the service, so the tests that are about the list
run one: ``OPENDDE_MODEL_SERVICE_FAUX=1`` registers four scripted providers
inside it under their own pi ids, and a provider this config *declares* named
``faux`` is the one it replaces. Nothing here reaches a vendor: ``models`` is
answered from pi's own built-in rows plus those scripted providers.

What ``model.overlay`` writes is a row in ``providers.<id>.models`` -- the
method keeps its name because the TUI calls it that -- so the tests that used to
read a ``modelOverlay`` keyed by a model's spelling now read the entry's own
list, which is the only place a model's declared facts live.
"""

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from opendde_harness.config import loader
from opendde_harness.config.schema import Config, ProviderEntry
from opendde_harness.providers import model_id
from opendde_harness.tui_rpc.errors import ConfigValidationError
from opendde_harness.tui_rpc.methods import model as model_methods
from opendde_harness.tui_rpc.methods.model import model_overlay
from tests._config import config_dict, declared, write_config
from tests._login import CODE_STEP, MENU_STEP, OAUTH_PROVIDER
from tests._login import FakeLoginService as _FakeService

BUNDLE = Path(__file__).resolve().parent.parent / "ui-tui" / "dist" / "model-service.js"
NODE = shutil.which("node")

#: The model this config's default names, and the provider that declares it.
PROVIDER = "my-vllm"
MODEL = "my-vllm/deep-thinker"

#: The scripted provider's one model, and the name its row carries.
FAUX_MODEL = "faux/echo"
FAUX_NAME = "Faux echo"


def _providers() -> dict:
    """One declared provider serving the default model, as the file holds it."""
    return declared(PROVIDER, base_url="http://relay/v1", models=["deep-thinker"])


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = write_config(tmp_path / "config.json", _providers(), model=MODEL)
    monkeypatch.setattr(loader, "_current_config_path", path)
    loader._cache.clear()
    return path


@pytest.fixture
def offline(monkeypatch):
    """The service cannot be asked, which is the picker's fallback case.

    Both of the picker's questions: what each provider serves, and how each one
    is signed in to. A fixture that refused only the first started a real service
    for the second, which is not what "offline" means.
    """

    async def refuse(_config):
        raise RuntimeError("no model service here")

    monkeypatch.setattr(model_methods, "service_rows", refuse)
    monkeypatch.setattr(model_methods, "provider_auth", refuse)
    monkeypatch.setattr(model_methods, "refresh_models", refuse)


@pytest.fixture
async def faux_service(tmp_path, monkeypatch):
    """A real model service in its scripted mode, plus a provider mapped to it.

    ``faux`` is declared rather than built in, which is the case the picker
    could say the least about before: pi carries no rows for it, so everything
    offered is either the entry's own list or what the service reports. The
    scripted provider registers under that same id and replaces it, which is
    what makes the two sources distinguishable here.
    """
    if not BUNDLE.exists() or NODE is None:
        pytest.skip("ui-tui/dist/model-service.js is not built (cd ui-tui && npm run build) or node is missing")
    monkeypatch.setenv("OPENDDE_MODEL_SERVICE_FAUX", "1")
    path = write_config(
        tmp_path / "config.json",
        {
            **_providers(),
            **declared("faux", base_url="http://faux/v1", models=["local-thing"]),
            # A built-in, with a curated list and no key: the gate calls that
            # unconfigured, and pi still carries rows for it.
            "deepseek": {"models": ["private-build"]},
        },
        model=MODEL,
    )
    monkeypatch.setattr(loader, "_current_config_path", path)
    loader._cache.clear()
    yield path
    from opendde_harness.providers.pi_service import shutdown_service

    await shutdown_service()


def _loop(config_path):
    """A loop whose two providers maps are separate objects, as a switch makes them.

    The settings hold the map the boot provider was built from; a session that
    switched runs on a provider built from its own copy of the config. A row
    written once has to land in both, so the two are validated separately here
    rather than shared.
    """
    raw = json.loads(config_path.read_text())
    inner = SimpleNamespace(providers=Config.model_validate(raw).providers)
    lazy = SimpleNamespace(_provider=SimpleNamespace(_inners=[inner]))
    loop = SimpleNamespace(
        settings=SimpleNamespace(providers=Config.model_validate(raw).providers),
        provider=lazy,
        context_window=None,
    )

    def resolve_window(model, binding=None):
        """The loop's own ladder, with only its first tier: what the user
        declared for this model. The stub provider reports nothing, so the
        second tier answers None -- whichever binding is handed over -- which is
        what these tests are about."""
        from opendde_harness.providers.base import declared_tokens
        from opendde_harness.providers.rates import SOURCE_DECLARED, SOURCE_UNKNOWN, Resolved

        tokens = declared_tokens(model_id.row_for(loop.settings.providers, model), "context_window")
        return Resolved(tokens, SOURCE_DECLARED) if tokens else Resolved(None, SOURCE_UNKNOWN)

    def refresh():
        loop.context_window = resolve_window(MODEL).tokens

    loop.resolve_window = resolve_window
    loop.refresh_context_window = refresh
    loop.has_session_binding = lambda session_key: False
    loop.live_providers = lambda: [lazy]
    return loop, inner


def _row(providers, provider=PROVIDER, model=MODEL):
    """The declared row for one model in a live providers map."""
    return model_id.row_for(providers, model_id.join(provider, model))


async def test_a_row_declared_by_a_switched_session_reaches_its_own_provider(config_path):
    """The session is on a provider of its own, not the default's; the level
    it declares must land on the one its next turn will call."""
    loop, default_inner = _loop(config_path)
    own = SimpleNamespace(providers=Config.model_validate(json.loads(config_path.read_text())).providers)
    loop.has_session_binding = lambda session_key: session_key == "tui:a"
    loop.binding_for_session = lambda session_key: SimpleNamespace(model=MODEL, provider_name=PROVIDER, provider=own)
    loop.live_providers = lambda: [loop.provider, own]

    await model_overlay({"field": "reasoning_effort", "value": "high", "session_id": "tui:a"}, lambda: loop)

    assert _row(own.providers).reasoning_effort == "high"
    assert _row(default_inner.providers).reasoning_effort == "high"


async def test_the_level_is_persisted_and_applied_to_the_running_providers(config_path):
    loop, inner = _loop(config_path)

    result = await model_overlay({"field": "reasoning_effort", "value": "high"}, lambda: loop)

    assert result == {"model": MODEL, "field": "reasoning_effort", "value": "high"}
    stored = Config.model_validate(json.loads(config_path.read_text())).providers
    assert _row(stored).reasoning_effort == "high"
    assert _row(loop.settings.providers).reasoning_effort == "high"
    assert _row(inner.providers).reasoning_effort == "high"

    assert (await model_overlay({"field": "reasoning_effort", "value": "default"}, lambda: loop))["value"] is None
    assert _row(inner.providers).reasoning_effort is None


async def test_a_context_window_declared_in_session_sizes_the_loop_at_once(config_path):
    loop, _ = _loop(config_path)

    result = await model_overlay({"field": "context_window", "value": "128k"}, lambda: loop)

    assert result["value"] == 128_000 and result["context_window"] == 128_000
    assert loop.context_window == 128_000

    cleared = await model_overlay({"field": "context_window", "value": "default"}, lambda: loop)
    assert cleared["context_window"] is None


async def test_a_row_field_the_catalogue_would_otherwise_answer_is_written_as_typed(config_path):
    """Every field of the row is settable, not only the two this project added:
    a relay that renamed what it fronts is described by ``catalog_model``, and
    the wire one model speaks by ``api``."""
    loop, inner = _loop(config_path)

    assert (await model_overlay({"field": "catalog_model", "value": "openai/gpt-4o"}, lambda: loop))["value"] == (
        "openai/gpt-4o"
    )
    assert (await model_overlay({"field": "api", "value": "openai-responses"}, lambda: loop))["value"] == (
        "openai-responses"
    )

    row = _row(inner.providers)
    assert (row.catalog_model, row.api) == ("openai/gpt-4o", "openai-responses")


@pytest.mark.parametrize(
    "params",
    [
        {"field": "reasoning_effort", "value": "ultra"},
        {"field": "api", "value": "grpc"},
        {"field": "context_window", "value": "lots"},
        {"field": "context_window", "value": "0"},
        {"field": "context_window_tokens", "value": "128k"},
        {"field": "colour", "value": "blue"},
    ],
)
async def test_a_bad_field_or_value_is_refused_before_anything_is_written(config_path, params):
    before = config_path.read_text()

    with pytest.raises(ConfigValidationError):
        await model_overlay(params, None)

    assert config_path.read_text() == before


@pytest.fixture
def count_http(monkeypatch):
    """Count every attempt to reach the network, including swallowed ones.

    Raising is not enough on its own: the discovery this replaced caught
    whatever the transport raised, returned an empty list and cached the
    failure, so a test that only checked the call succeeded passed while a
    request went out. The counter records the attempt before anything can
    swallow it.
    """
    import socket

    import httpx

    attempts: list[str] = []

    def refuse(name):
        def attempt(*args, **kwargs):
            attempts.append(name)
            raise OSError(f"blocked: {name}")

        return attempt

    for module, name in (
        (httpx, "Client"),
        (httpx, "AsyncClient"),
        (httpx, "get"),
        (httpx, "post"),
        (httpx, "request"),
    ):
        monkeypatch.setattr(module, name, refuse(f"httpx.{name}"))
    monkeypatch.setattr(socket, "create_connection", refuse("socket.create_connection"))

    real_socket = socket.socket

    def guarded_socket(family=socket.AF_INET, *args, **kwargs):
        # Only the internet families: asyncio opens an AF_UNIX pair of its own
        # while the loop starts, and blocking that blocks the test, not a
        # request.
        if family in (socket.AF_INET, socket.AF_INET6):
            attempts.append("socket.socket")
            raise OSError("blocked: socket.socket")
        return real_socket(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", guarded_socket)
    monkeypatch.setattr("urllib.request.urlopen", refuse("urlopen"))
    return attempts


async def test_the_picker_lists_what_the_service_says_the_provider_serves(faux_service, count_http):
    """The point of the change: the ids come from the model service.

    A declared provider gets its own list *and* every row the service reports
    for it, each written in this project's storage spelling -- the service
    reports the bare id the endpoint serves, and a bare id names no provider.
    """
    result = await model_methods.model_options({"slug": "faux"})

    assert count_http == []
    row = result["providers"][0]
    assert row["models"] == ["faux/local-thing", FAUX_MODEL]
    assert row["total_models"] == 2


async def test_a_model_is_labelled_by_its_own_row_first_and_by_the_service_next(faux_service, count_http):
    """Two sources, in that order. The user describing their own deployment
    outranks the name the service carries; a model the user said nothing about
    still shows what it is rather than only its id."""
    raw = json.loads(faux_service.read_text())
    raw["providers"]["faux"]["models"] = [{"id": "echo", "name": "The one I renamed"}, "local-thing"]
    faux_service.write_text(json.dumps(raw))
    loader._cache.clear()

    labelled = (await model_methods.model_options({"slug": "faux"}))["providers"][0]["model_labels"]

    assert count_http == []
    assert labelled[FAUX_MODEL]["label"] == "The one I renamed"

    raw["providers"]["faux"]["models"] = ["echo", "local-thing"]
    faux_service.write_text(json.dumps(raw))
    loader._cache.clear()

    unlabelled = (await model_methods.model_options({"slug": "faux"}))["providers"][0]["model_labels"]

    assert unlabelled[FAUX_MODEL]["label"] == FAUX_NAME


async def test_an_unconfigured_provider_offers_only_what_it_configured(faux_service, count_http):
    """pi's collection carries every vendor it knows whether or not a
    credential resolves, so listing them for a provider with no key would offer
    models whose every request fails on the missing key."""
    result = await model_methods.model_options({"slug": "deepseek"})

    assert count_http == []
    row = result["providers"][0]
    assert row["authenticated"] is False
    assert row["models"] == ["deepseek/private-build"]


async def test_the_row_carries_pis_own_ways_in_and_pis_own_labels(faux_service, count_http):
    """What a picker offers a choice with comes from pi, not from a table here.

    pi keeps both ways in on its provider objects with the sentence it shows for
    each; the model service reads them off and the row passes them through. A
    provider with two is the one a client has to ask about, and pi is the only
    thing that knows which those are.
    """
    result = await model_methods.model_options({"include_catalog": False})
    rows = {row["slug"]: row for row in result["providers"]}

    assert count_http == []
    # A sign-in and a key, with pi's own label for the sign-in.
    assert rows["xai"]["auth_methods"] == ["oauth", "key"]
    assert rows["xai"]["login_label"] == "Sign in with SuperGrok or X Premium"
    assert rows["xai"]["key_label"] == "xAI API key"
    # A sign-in and nothing else: nothing to ask.
    assert rows["openai-codex"]["auth_methods"] == ["oauth"]
    assert rows["openai-codex"]["login_label"] is None
    # A key and nothing else.
    assert rows["deepseek"]["auth_methods"] == ["key"]
    # pi writes no label for this one, so a menu falls back to pi's own generic
    # sentence rather than to one of ours.
    assert rows["anthropic"]["auth_methods"] == ["oauth", "key"]
    assert rows["anthropic"]["login_label"] is None


async def test_a_picker_that_cannot_ask_pi_offers_no_choice(config_path, offline, count_http):
    """The service is what knows the ways in, so a picker that cannot reach it
    says nothing about them: an empty list is what sends the client back to the
    row's own ``auth_type``, which is where it looked before pi was asked."""
    row = (await model_methods.model_options({"slug": "anthropic"}))["providers"][0]

    assert count_http == []
    assert row["auth_methods"] == []
    assert row["login_label"] is None and row["key_label"] is None


async def test_the_curated_shortlist_answers_only_when_the_service_cannot(config_path, offline, count_http):
    """No Node, no bundle, a configuration the service refused: the picker is
    still openable, and a shortlist beats a provider that appears to serve
    nothing. This is the only path that reaches it."""
    result = await model_methods.model_options({"slug": "deepseek"})

    assert count_http == []
    assert "deepseek/deepseek-v4-flash" in result["providers"][0]["models"]


async def test_a_provider_with_no_shortlist_offers_nothing_rather_than_guessing(config_path, offline, count_http):
    """Offline and uncurated: a provider nothing can answer for is named by hand
    (``model.add_model``). Offering an id nobody declared would send a request
    the provider refuses."""
    baseten = await model_methods.model_options({"slug": "baseten"})

    assert count_http == []
    assert baseten["providers"][0]["models"] == []


async def test_the_expansion_the_old_client_sends_comes_back_complete(config_path, offline, count_http):
    """The picker opens with an explicit ``include_catalog: false`` and then
    expands one provider by sending ``{slug}`` alone, relying on the default
    the schema declares. Answering that with nothing is read as "failed to
    load more models"."""
    opening = await model_methods.model_options({"include_catalog": False})
    expansion = await model_methods.model_options({"slug": "deepseek"})

    assert count_http == []
    assert next(row for row in opening["providers"] if row["slug"] == "deepseek")["models_loaded"] is False
    assert expansion["providers"][0]["models_loaded"] is True
    assert "deepseek/deepseek-v4-flash" in expansion["providers"][0]["models"]


async def test_the_model_in_use_heads_its_providers_list(config_path, offline, count_http):
    result = await model_methods.model_options({"slug": PROVIDER, "include_catalog": True})

    assert count_http == []
    assert result["providers"][0]["models"][0] == MODEL


def test_a_declared_endpoint_offers_what_the_service_says_it_publishes(count_http):
    """The rows the service reports for a declared provider are the list its
    ``/models`` published (fetched by the service, never from here): they are
    offered after the ids the operator declared by hand, qualified."""
    entry = ProviderEntry.model_validate(
        {"baseUrl": "https://relay/v1", "api": "openai-completions", "apiKey": "sk", "models": ["by-hand"]}
    )

    row = model_methods._build_provider_entry(
        PROVIDER,
        entry,
        current_provider=None,
        include_catalog=True,
        served=[
            {"provider": PROVIDER, "id": "by-hand", "name": "By hand"},
            {"provider": PROVIDER, "id": "anthropic/claude-opus-5", "name": "Anthropic: Claude Opus 5"},
            {"provider": "openai", "id": "gpt-5", "name": "GPT-5"},
        ],
    )

    assert count_http == []
    assert row["models"] == ["my-vllm/by-hand", "my-vllm/anthropic/claude-opus-5"]
    assert row["model_labels"]["my-vllm/anthropic/claude-opus-5"]["label"] == "Anthropic: Claude Opus 5"


def test_a_service_row_is_offered_in_the_spelling_the_picker_stores(count_http):
    """The rows carry the bare id the endpoint serves; what the picker hands
    back goes straight into ``model.add_model`` and ``config.set``, which need
    the id to name its provider."""
    entry = ProviderEntry.model_validate({"apiKey": "sk"})

    row = model_methods._build_provider_entry(
        "deepseek",
        entry,
        current_provider=None,
        include_catalog=True,
        served=[{"provider": "deepseek", "id": "deepseek-chat", "name": "DeepSeek Chat"}],
    )

    assert count_http == []
    assert row["models"] == ["deepseek/deepseek-chat"]
    assert row["model_labels"]["deepseek/deepseek-chat"]["label"] == "DeepSeek Chat"


def test_a_declared_provider_is_asked_for_an_address_and_a_builtin_for_a_key(count_http):
    """What the credential form asks for, decided by the gate rather than by the
    form: an address and the wire it serves for a provider this config declares,
    a key for one of pi's own, and never both."""
    declared_row = model_methods._build_provider_entry("my-relay", None, current_provider=None)
    builtin_row = model_methods._build_provider_entry("deepseek", None, current_provider=None)

    assert count_http == []
    assert (declared_row["auth_type"], declared_row["needs_base_url"], declared_row["needs_api_key"]) == (
        "endpoint",
        True,
        False,
    )
    assert (builtin_row["auth_type"], builtin_row["needs_base_url"], builtin_row["needs_api_key"]) == (
        "key",
        False,
        True,
    )


def test_a_provider_whose_credential_is_an_environment_chain_is_not_asked_for_a_key(count_http):
    """Bedrock and Vertex take the AWS chain and Google's application default
    credentials: the gate accepts an entry with no key for them, so a form that
    demanded one demanded a string nobody has, and the wizard's key form and
    the TUI's both read this flag to decide whether blank is an answer."""
    row = model_methods._build_provider_entry("amazon-bedrock", None, current_provider=None)

    assert count_http == []
    assert row["auth_type"] == "key"
    assert row["needs_api_key"] is False


def test_a_key_already_in_the_environment_is_reported_and_never_asked_for(monkeypatch, count_http):
    """pi resolves the environment itself, so the variable's name travels and
    its value does not -- and a form that asked for a key would be asking for
    one the entry does not need to hold."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-the-environment")

    row = model_methods._build_provider_entry("deepseek", None, current_provider=None)

    assert count_http == []
    assert row["key_env"] == "DEEPSEEK_API_KEY"
    assert row["needs_api_key"] is False
    assert "sk-from-the-environment" not in json.dumps(row)


def test_a_provider_the_config_declares_is_never_offered_under_add_provider(config_path):
    """It needs an address and a wire, neither of which is a thing to guess at
    in a picker, so ``ddeharness provider set`` is where one is written. The
    addable half is pi's built-ins and nothing else."""
    listed = model_methods._listed_providers(loader.load_config(config_path))
    addable = [provider for provider, entry in listed if entry is None]

    from opendde_harness.providers import pi_ids

    assert PROVIDER not in addable
    assert set(addable) <= set(pi_ids.BUILTIN)
    assert [provider for provider, _ in listed][0] == PROVIDER, "the configured entries come first, in file order"


def test_azure_is_offered_only_once_it_is_declared(tmp_path):
    """pi ships the id and not the provider: a resource is one tenant's own
    address serving deployment names only they have, so a key alone reaches
    nothing and a row offering one could not be finished. Declared with
    ``ddeharness provider set``, it is a configured entry like any other."""
    from opendde_harness.providers import pi_ids

    assert "azure-openai-responses" in pi_ids.BUILTIN
    addable = [provider for provider, entry in model_methods._listed_providers(None) if entry is None]
    assert "azure-openai-responses" not in addable

    path = write_config(
        tmp_path / "azure.json",
        declared("azure-openai-responses", base_url="https://contoso.openai.azure.com", api="azure-openai-responses"),
        model="azure-openai-responses/test-model",
    )
    listed = [provider for provider, _ in model_methods._listed_providers(loader.load_config(path))]
    assert listed[0] == "azure-openai-responses"


#: What the picker's endpoint form submits: an id pi does not ship, an address,
#: and one model. The key is written and never read back, so it is only ever
#: asserted through the redacting reader.
DECLARED_ID = "my-relay"
DECLARED_URL = "https://relay.internal:8443/v1"
DECLARED_MODEL = "gpt-oss-120b"


def _submission(**overrides) -> dict:
    return {
        "provider": DECLARED_ID,
        "base_url": DECLARED_URL,
        "model": DECLARED_MODEL,
        "api_key": "sk-typed-into-the-form",
        **overrides,
    }


async def test_declaring_an_endpoint_writes_the_entry_the_cli_would(config_path, offline, count_http):
    """One submission, one entry: the address, the wire it is declared as
    speaking, the key, and the model. The same write path ``ddeharness provider
    set`` uses, so a picker declaration and a command line cannot disagree."""
    from opendde_harness.config.update_providers import get_provider_config

    result = await model_methods.model_declare_provider(_submission())

    assert count_http == []
    # Read back through `provider show`'s own reader, which redacts every secret:
    # the key is typed into one form and stored, and a test that printed it would
    # put it in every failure report of this file.
    stored = get_provider_config(DECLARED_ID, config_path=config_path)
    assert (stored["base_url"], stored["api"], stored["models"]) == (
        DECLARED_URL,
        "openai-completions",
        [DECLARED_MODEL],
    )
    assert stored["api_key"] == "****set****"
    # And the file holds pi's own spelling, with nothing else under the entry: a
    # fresh one writes only what was said.
    assert set(json.loads(config_path.read_text())["providers"][DECLARED_ID]) == {
        "baseUrl",
        "api",
        "apiKey",
        "models",
    }
    # The row the picker lands on: configured by construction, since a declared
    # provider is reached by the address it carries, and offering the one model.
    row = result["provider"]
    assert row["slug"] == DECLARED_ID
    assert row["authenticated"] is True
    assert row["models"] == [model_id.join(DECLARED_ID, DECLARED_MODEL)]


@pytest.mark.parametrize(
    ("overrides", "says"),
    [
        ({"provider": "openai"}, "pi ships"),
        ({"provider": "deepseek"}, "pi ships"),
        ({"provider": "gemini"}, "google"),
        ({"provider": "github-copilot"}, "removed"),
        ({"provider": "my/relay"}, "not a provider id"),
        ({"provider": ""}, "not a provider id"),
        ({"base_url": "relay.internal/v1"}, "http"),
        ({"base_url": "ftp://relay.internal/v1"}, "http"),
    ],
)
async def test_a_declaration_the_picker_cannot_write_is_refused_untouched(
    config_path, offline, count_http, overrides, says
):
    """Every refusal before the write, and the sentence in the detail rather than
    in a ``data`` of its own: the frame carries one of the two, and it is the
    sentence the form has to show."""
    before = config_path.read_text()

    with pytest.raises(ConfigValidationError) as refused:
        await model_methods.model_declare_provider(_submission(**overrides))

    assert says in refused.value.detail
    assert refused.value.data is None
    assert config_path.read_text() == before
    assert count_http == []


async def test_an_endpoint_is_asked_what_it_serves_the_moment_it_is_declared(config_path, count_http, monkeypatch):
    """The way pi treats OpenRouter: the address and the key are typed, the
    list is published. The declaration is written, the service (reconfigured
    with the new entry) fetches the endpoint's ``/models``, and the row lands
    with what it published, counted."""
    asked: list[tuple[list[str] | None, bool]] = []

    async def refresh(config, providers=None, *, force=False):
        asked.append((list(providers) if providers else None, force))
        return {}

    async def rows(config):
        return [
            {"provider": DECLARED_ID, "id": "anthropic/claude-opus-5", "name": "Anthropic: Claude Opus 5"},
            {"provider": DECLARED_ID, "id": "qwen3-32b", "name": "qwen3-32b"},
        ]

    async def no_auth(config):
        return []

    monkeypatch.setattr(model_methods, "refresh_models", refresh)
    monkeypatch.setattr(model_methods, "service_rows", rows)
    monkeypatch.setattr(model_methods, "provider_auth", no_auth)

    result = await model_methods.model_declare_provider(_submission(model=""))

    assert count_http == []
    # Forced: the endpoint was just declared, and a list it published for
    # another declaration under this id is not one to trust for a day.
    assert asked == [([DECLARED_ID], True)]
    assert result["discovered"] == 2
    assert result["provider"]["models"] == [
        model_id.join(DECLARED_ID, "anthropic/claude-opus-5"),
        model_id.join(DECLARED_ID, "qwen3-32b"),
    ]
    assert json.loads(config_path.read_text())["providers"][DECLARED_ID].get("models", []) == []


async def test_an_endpoint_that_publishes_nothing_and_was_given_no_ids_is_refused_and_not_left_behind(
    config_path, offline, count_http
):
    """Declared empty it would offer nothing to pick, so the entry is taken
    back out and the sentence says to name the ids by hand."""
    with pytest.raises(ConfigValidationError) as refused:
        await model_methods.model_declare_provider(_submission(model=""))

    assert "lists no models" in refused.value.detail
    assert "comma-separated" in refused.value.detail
    assert DECLARED_ID not in json.loads(config_path.read_text())["providers"]
    assert count_http == []


async def test_ids_typed_by_hand_are_split_on_commas_and_kept_when_the_endpoint_publishes_none(
    config_path, offline, count_http
):
    result = await model_methods.model_declare_provider(_submission(model=" a , b,, "))

    assert json.loads(config_path.read_text())["providers"][DECLARED_ID]["models"] == ["a", "b"]
    assert result["discovered"] == 0
    assert result["provider"]["models"] == [model_id.join(DECLARED_ID, "a"), model_id.join(DECLARED_ID, "b")]


async def test_the_picker_can_ask_the_endpoints_to_refresh_and_hears_who_could_not(config_path, offline, monkeypatch):
    calls: list[bool] = []

    async def refresh(config, providers=None, *, force=False):
        # Not forced: the picker's refresh leaves a list checked today alone.
        calls.append(force)
        return {"my-relay": "my-relay answered 503 to GET /models"}

    monkeypatch.setattr(model_methods, "refresh_models", refresh)

    result = await model_methods.model_options({"include_catalog": False, "refresh": True})
    assert result["refresh_errors"] == {"my-relay": "my-relay answered 503 to GET /models"}
    assert calls == [False]

    # Without the flag nothing is asked, and a service that cannot be reached
    # at all is reported under `*` rather than failing the picker.
    assert (await model_methods.model_options({"include_catalog": False}))["refresh_errors"] == {}
    assert calls == [False]
    monkeypatch.setattr(model_methods, "refresh_models", offline_refuse)
    assert "*" in (await model_methods.model_options({"include_catalog": False, "refresh": True}))["refresh_errors"]


async def offline_refuse(_config, _providers=None):
    raise RuntimeError("no model service here")


async def test_the_scope_is_saved_under_agents_and_read_back(config_path, offline):
    """pi's scoped models: an explicit ordered list, or None for all."""
    assert await model_methods.model_scope({}) == {"models": None}

    saved = await model_methods.model_scope({"models": [MODEL, "openai/gpt-5"], "write": True})
    assert saved == {"models": [MODEL, "openai/gpt-5"]}
    assert json.loads(config_path.read_text())["agents"]["scopedModels"] == [MODEL, "openai/gpt-5"]
    loader._cache.clear()
    assert await model_methods.model_scope({}) == {"models": [MODEL, "openai/gpt-5"]}

    assert await model_methods.model_scope({"models": None, "write": True}) == {"models": None}
    assert "scopedModels" not in json.loads(config_path.read_text())["agents"]


async def test_a_pi_provider_id_is_refused_before_it_can_shadow_pis_own(config_path, offline, count_http):
    """pi carries that provider's address and its catalogue, and the provider
    list already offers it a credential. A declaration filed under its id would
    replace both with four typed fields."""
    with pytest.raises(ConfigValidationError) as refused:
        await model_methods.model_declare_provider(_submission(provider="anthropic"))

    assert "Anthropic" in refused.value.detail
    assert "anthropic" not in json.loads(config_path.read_text())["providers"]
    assert count_http == []


def test_a_config_the_picker_cannot_parse_still_lists_what_could_be_added(tmp_path, monkeypatch):
    """A broken file is exactly what somebody opens the picker to fix, so the
    addable half still answers; the configured half cannot, because nothing can
    read it."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**config_dict(), "providers": {"not-a-pi-id": {"apiKey": "k"}}}))
    monkeypatch.setattr(loader, "_current_config_path", path)
    loader._cache.clear()

    assert model_methods._loaded_config() is None
    assert model_methods._listed_providers(None), "the built-ins are listed without a config"


def test_the_endpoint_rpcs_are_not_registered_any_more():
    """The picker's endpoint stages are gone and so is what they called.

    Asked of the dispatcher rather than of the module, because a handler that
    stayed registered would answer a client this release no longer ships and
    write a key into a field nothing reads.

    The sign-in group is absent as well without a notification sink: its steps
    are pushed, so a gateway that cannot push would offer a sign-in that shows
    the user nothing and then waits for an answer to a question never asked.
    """
    from opendde_harness.tui_rpc.dispatcher import Dispatcher
    from opendde_harness.tui_rpc.methods.model import register_model_methods

    dispatcher = Dispatcher()
    register_model_methods(dispatcher)
    registered = {name for name in dispatcher._handlers if name.startswith("model.")}

    assert registered == {
        "model.options",
        "model.save_key",
        "model.declare_provider",
        "model.logout",
        "model.scope",
        "model.overlay",
    }


def test_the_sign_in_group_is_registered_with_a_notification_sink():
    from opendde_harness.tui_rpc.dispatcher import Dispatcher
    from opendde_harness.tui_rpc.methods.model import register_model_methods

    async def send_frame(_frame):
        return None

    dispatcher = Dispatcher()
    register_model_methods(dispatcher, send_frame=send_frame)

    assert {"model.login", "model.login_answer", "model.login_cancel"} <= set(dispatcher._handlers)


# ---------------------------------------------------------------------------
# The sign-in: pi's flow run in the model service and shown step by step
# ---------------------------------------------------------------------------


@pytest.fixture
def signing_in(tmp_path, monkeypatch, offline):
    """A config with no sign-in yet, and a fake service in place of the child.

    ``offline`` keeps the real model service out of the row build: what the
    provider serves is not what these tests are about, and starting Node for it
    would make them depend on a built bundle.
    """
    path = write_config(tmp_path / "config.json", _providers(), model=MODEL)
    monkeypatch.setattr(loader, "_current_config_path", path)
    loader._cache.clear()

    frames: list[dict] = []

    async def send_frame(frame: dict) -> None:
        frames.append(frame)

    def install(service: _FakeService) -> _FakeService:
        async def answer():
            return service

        monkeypatch.setattr(model_methods, "_login_service", answer)
        return service

    return SimpleNamespace(config_path=path, frames=frames, install=install, send_frame=send_frame)


def _store_a_grant() -> None:
    """Put an OAuth credential in the sandbox store, as a finished login does.

    Written as the file rather than through the store, because the store is the
    model service's to write and the service here is a fake. The path is the
    per-run sandbox ``conftest`` points ``CHATGPT_TOKEN_DIR`` at.
    """
    from opendde_harness.providers.pi_service import credential_store_path

    store = credential_store_path()
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps({OAUTH_PROVIDER: {"type": "oauth", "access": "a", "refresh": "r", "expires": 0}}))


async def test_every_step_is_pushed_and_the_row_comes_back_connected(signing_in):
    """The whole flow: pi's steps travel as they arrive, and the answer is the
    picker row for a provider that is now reached by a sign-in."""
    service = signing_in.install(_FakeService([MENU_STEP, CODE_STEP]))
    task = asyncio.create_task(
        model_methods.model_login({"provider": OAUTH_PROVIDER}, send_frame=signing_in.send_frame)
    )
    await asyncio.wait_for(service.streamed.wait(), 2)

    # Every step pushed, in order, each naming the login it belongs to.
    pushed = [frame["params"] for frame in signing_in.frames]
    assert [frame["method"] for frame in signing_in.frames] == ["login.step", "login.step"]
    assert [step["step"] for step in pushed] == [MENU_STEP["ask"], CODE_STEP["notify"]]
    assert {step["login_id"] for step in pushed} == {pushed[0]["login_id"]}
    assert {step["provider"] for step in pushed} == {OAUTH_PROVIDER}
    # No mode: forwarding pi's menu is the point.
    assert service.asked == {"provider": OAUTH_PROVIDER, "mode": None}

    _store_a_grant()
    service.release()
    result = await asyncio.wait_for(task, 2)

    assert result["login_id"] == pushed[0]["login_id"]
    assert result["provider"]["slug"] == OAUTH_PROVIDER
    assert result["provider"]["authenticated"] is True
    assert result["provider"]["auth_type"] == "oauth"
    # And the entry now says how this provider is reached, which is the whole of
    # what the config records about a sign-in.
    written = json.loads(signing_in.config_path.read_text())["providers"][OAUTH_PROVIDER]
    assert written["login"] == "oauth"


async def test_an_answer_reaches_the_flow_that_is_waiting(signing_in):
    """The menu is answered by the id its own step carried, not by a guess."""
    service = signing_in.install(_FakeService([MENU_STEP]))
    task = asyncio.create_task(
        model_methods.model_login({"provider": OAUTH_PROVIDER}, send_frame=signing_in.send_frame)
    )
    await asyncio.wait_for(service.streamed.wait(), 2)
    login_id = signing_in.frames[0]["params"]["login_id"]

    assert await model_methods.model_login_answer({"login_id": login_id, "answer": "device_code"}) == {"answered": True}
    assert service.answers == [(11, "device_code")]

    _store_a_grant()
    service.release()
    await asyncio.wait_for(task, 2)

    # The login is over, so its id answers nothing: a late answer must not reach
    # whatever flow happens to be running next.
    assert await model_methods.model_login_answer({"login_id": login_id, "answer": "browser"}) == {"answered": False}


async def test_a_cancel_aborts_the_flow_and_fails_the_request(signing_in):
    service = signing_in.install(_FakeService([MENU_STEP]))
    task = asyncio.create_task(
        model_methods.model_login({"provider": OAUTH_PROVIDER}, send_frame=signing_in.send_frame)
    )
    await asyncio.wait_for(service.streamed.wait(), 2)
    login_id = signing_in.frames[0]["params"]["login_id"]

    assert await model_methods.model_login_cancel({"login_id": login_id}) == {"cancelled": True}
    assert service.aborted == [11]

    with pytest.raises(ConfigValidationError) as refused:
        await asyncio.wait_for(task, 2)

    assert "did not finish" in refused.value.detail
    # Nothing was written: a sign-in that was abandoned must not leave an entry
    # claiming a grant nothing holds.
    assert OAUTH_PROVIDER not in json.loads(signing_in.config_path.read_text())["providers"]
    assert await model_methods.model_login_cancel({"login_id": login_id}) == {"cancelled": False}


async def test_a_flow_that_ends_without_a_grant_is_not_reported_as_signed_in(signing_in):
    service = signing_in.install(_FakeService([MENU_STEP], result={"provider": OAUTH_PROVIDER, "type": "api_key"}))
    task = asyncio.create_task(
        model_methods.model_login({"provider": OAUTH_PROVIDER}, send_frame=signing_in.send_frame)
    )
    await asyncio.wait_for(service.streamed.wait(), 2)
    service.release()

    with pytest.raises(ConfigValidationError) as refused:
        await asyncio.wait_for(task, 2)

    assert "did not produce a grant" in refused.value.detail
    assert OAUTH_PROVIDER not in json.loads(signing_in.config_path.read_text())["providers"]


async def test_a_provider_that_does_not_sign_in_is_refused_before_the_service_is_asked(signing_in):
    """The gate is pi's own set. A key provider sent through here would run a
    flow no vendor there implements."""
    service = signing_in.install(_FakeService([MENU_STEP]))

    with pytest.raises(ConfigValidationError) as refused:
        await model_methods.model_login({"provider": PROVIDER}, send_frame=signing_in.send_frame)

    assert "not signed in to" in refused.value.detail
    assert service.asked is None
    assert signing_in.frames == []


async def test_an_answer_or_a_cancel_for_no_login_says_so(signing_in):
    assert await model_methods.model_login_answer({"login_id": "nope", "answer": "browser"}) == {"answered": False}
    assert await model_methods.model_login_cancel({"login_id": "nope"}) == {"cancelled": False}


async def test_the_client_chooses_the_id_its_steps_carry(signing_in):
    """A picker has to know its own flow's steps from another's before the
    first one arrives, so the id is its to choose: every push and the answer
    carry the one it sent."""
    service = signing_in.install(_FakeService([MENU_STEP]))
    task = asyncio.create_task(
        model_methods.model_login(
            {"provider": OAUTH_PROVIDER, "login_id": "picker-7"}, send_frame=signing_in.send_frame
        )
    )
    await asyncio.wait_for(service.streamed.wait(), 2)

    assert signing_in.frames[0]["params"]["login_id"] == "picker-7"
    assert await model_methods.model_login_answer({"login_id": "picker-7", "answer": "browser"}) == {"answered": True}

    _store_a_grant()
    service.release()
    assert (await asyncio.wait_for(task, 2))["login_id"] == "picker-7"


async def test_a_cancel_that_arrives_before_the_service_took_the_request_still_ends_it(signing_in):
    """Esc pressed in the moment between ``model.login`` and the flow existing.

    The id is the client's, so the cancel can name the flow before the service
    has a request id for it; it is remembered and applied the moment there is
    one, rather than answered "nothing to cancel" and leaving the flow polling.
    """
    service = signing_in.install(_FakeService([MENU_STEP]))
    service.take.clear()
    task = asyncio.create_task(
        model_methods.model_login({"provider": OAUTH_PROVIDER, "login_id": "early"}, send_frame=signing_in.send_frame)
    )
    await asyncio.sleep(0)

    assert await model_methods.model_login_cancel({"login_id": "early"}) == {"cancelled": True}
    assert service.aborted == [], "nothing to abort yet"

    service.take.set()
    with pytest.raises(ConfigValidationError):
        await asyncio.wait_for(task, 2)

    assert service.aborted == [11]


async def test_a_second_sign_in_to_the_same_provider_ends_the_first(signing_in):
    """One flow per provider. The second one is the first one's client having
    gone, and two flows would race for the same slot in the store."""
    first = signing_in.install(_FakeService([MENU_STEP]))
    first_task = asyncio.create_task(
        model_methods.model_login({"provider": OAUTH_PROVIDER, "login_id": "one"}, send_frame=signing_in.send_frame)
    )
    await asyncio.wait_for(first.streamed.wait(), 2)

    second = signing_in.install(_FakeService([MENU_STEP]))
    second_task = asyncio.create_task(
        model_methods.model_login({"provider": OAUTH_PROVIDER, "login_id": "two"}, send_frame=signing_in.send_frame)
    )
    await asyncio.wait_for(second.streamed.wait(), 2)

    assert first.aborted == [11]
    with pytest.raises(ConfigValidationError):
        await asyncio.wait_for(first_task, 2)
    # And the first one's answers reach nothing, while the second's flow goes on.
    assert await model_methods.model_login_answer({"login_id": "one", "answer": "browser"}) == {"answered": False}
    assert await model_methods.model_login_answer({"login_id": "two", "answer": "browser"}) == {"answered": True}

    _store_a_grant()
    second.release()
    await asyncio.wait_for(second_task, 2)


async def test_the_request_being_cancelled_ends_the_flow_at_the_service(signing_in):
    """The client went away mid-flow -- the socket closed, the dispatcher
    cancelled the handler -- and the flow it was holding is ended rather than
    left polling a vendor for a code nobody will enter."""
    service = signing_in.install(_FakeService([MENU_STEP]))
    task = asyncio.create_task(
        model_methods.model_login({"provider": OAUTH_PROVIDER, "login_id": "gone"}, send_frame=signing_in.send_frame)
    )
    await asyncio.wait_for(service.streamed.wait(), 2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert service.aborted == [11]
    assert await model_methods.model_login_cancel({"login_id": "gone"}) == {"cancelled": False}


async def test_a_failed_sign_in_logs_its_code_and_not_the_vendors_sentence(signing_in):
    """pi's parser quotes the response it could not read, which can carry the
    tokens in it; the log gets the code and the screen gets the code."""
    from loguru import logger

    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    try:
        service = signing_in.install(
            _FakeService([MENU_STEP], failure='invalid token response: {"access_token": "sk-live-secret"}')
        )
        task = asyncio.create_task(
            model_methods.model_login({"provider": OAUTH_PROVIDER}, send_frame=signing_in.send_frame)
        )
        await asyncio.wait_for(service.streamed.wait(), 2)
        service.release()

        with pytest.raises(ConfigValidationError) as refused:
            await asyncio.wait_for(task, 2)
    finally:
        logger.remove(sink)

    assert "sk-live-secret" not in refused.value.detail
    assert "sk-live-secret" not in "".join(lines)
    assert any("login_failed" in line for line in lines)


# ---------------------------------------------------------------------------
# Signing out: pi's /logout keeps the provider, disconnect removes it
# ---------------------------------------------------------------------------


class _SignedOutService:
    """The process's model service, as far as a sign-out reaches it."""

    def __init__(self, *, fail: bool = False):
        self.forgotten: list[str] = []
        self._fail = fail

    async def logout(self, provider: str) -> dict:
        from opendde_harness.providers.model_service import ModelServiceError

        if self._fail:
            raise ModelServiceError("logout_failed", "the store could not be written")
        self.forgotten.append(provider)
        return {"provider": provider, "forgotten": True}


@pytest.fixture
def signing_out(tmp_path, monkeypatch, offline):
    """A config with a sign-in, a key and a declared endpoint, and the process's
    service faked where the sign-out reaches it (``pi_service.get_service``)."""
    from opendde_harness.providers import pi_service

    providers = {
        **_providers(),
        OAUTH_PROVIDER: {"login": "oauth"},
        "deepseek": {"apiKey": "sk-deepseek-synthetic"},
    }
    providers[PROVIDER]["apiKey"] = "relay-key"
    path = write_config(tmp_path / "config.json", providers, model=MODEL)
    monkeypatch.setattr(loader, "_current_config_path", path)
    loader._cache.clear()
    _store_a_grant()

    holder: dict[str, _SignedOutService] = {}

    async def get_service(_config):
        return holder["service"]

    monkeypatch.setattr(pi_service, "get_service", get_service)

    def install(**kwargs) -> _SignedOutService:
        holder["service"] = _SignedOutService(**kwargs)
        return holder["service"]

    def entries() -> dict:
        return json.loads(path.read_text())["providers"]

    return SimpleNamespace(install=install, entries=entries)


async def test_logout_forgets_the_grant_and_keeps_the_provider(signing_out):
    """pi's own logout: the credential goes, the entry stays."""
    service = signing_out.install()

    assert await model_methods.model_logout({"slug": OAUTH_PROVIDER}) == {"forgotten": True}

    assert service.forgotten == [OAUTH_PROVIDER]
    entry = signing_out.entries()[OAUTH_PROVIDER]
    assert entry.get("login", "") == "" and not entry.get("apiKey")


async def test_logout_blanks_a_key_and_keeps_what_the_entry_declares(signing_out):
    """A key is a credential too, and pi lists it under /logout. What the entry
    declares -- an address, a wire, its models -- is not a credential."""
    service = signing_out.install()

    assert await model_methods.model_logout({"slug": PROVIDER}) == {"forgotten": True}
    assert await model_methods.model_logout({"slug": "deepseek"}) == {"forgotten": True}

    assert service.forgotten == [], "no sign-in was involved"
    relay = signing_out.entries()[PROVIDER]
    assert relay["baseUrl"] == "http://relay/v1" and relay["models"] == ["deep-thinker"]
    assert not relay.get("apiKey")
    assert not signing_out.entries()["deepseek"].get("apiKey")
    # And a second logout has nothing left to forget.
    assert await model_methods.model_logout({"slug": "deepseek"}) == {"forgotten": False}


async def test_a_sign_out_the_service_could_not_do_is_an_error_and_the_entry_stays(signing_out):
    """A false "logged out" is a grant the config has forgotten about and the
    store has not. The entry is not touched until the store has been."""
    signing_out.install(fail=True)

    with pytest.raises(ConfigValidationError) as refused:
        await model_methods.model_logout({"slug": OAUTH_PROVIDER})
    assert "could not sign" in refused.value.detail
    assert signing_out.entries()[OAUTH_PROVIDER] == {"login": "oauth"}


async def test_a_key_replaces_a_sign_in(signing_out, count_http):
    """pi's "Sign in with an API key" for a provider that was signed in to: the
    key is written and the entry stops naming the sign-in, rather than the
    write being refused for naming one."""
    result = await model_methods.model_save_key({"slug": OAUTH_PROVIDER, "api_key": "sk-ant-synthetic"})

    entry = signing_out.entries()[OAUTH_PROVIDER]
    assert entry.get("login", "") == ""
    assert result["provider"]["auth_type"] == "key"
    assert "sk-ant-synthetic" not in json.dumps(result)
