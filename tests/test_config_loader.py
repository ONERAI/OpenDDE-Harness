import json

import pytest

from opendde_harness.config import loader
from opendde_harness.config._fields import write_json_atomic
from opendde_harness.config.loader import ConfigSchemaError, load_config, save_config
from opendde_harness.config.opendde_harness import OpenDDEHarnessConfig, load_opendde_harness_config
from opendde_harness.config.schema import Config, ProviderEntry
from opendde_harness.config.update import init_extension_block_defaults, set_skill_blocked
from opendde_harness.plugin.memory.longterm import _library
from tests._config import config_dict, declared, keyed


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(loader, "_current_config_path", path)
    monkeypatch.delenv("OPENDDE_HARNESS_LANGUAGE", raising=False)
    return path


def _write(path, data):
    write_json_atomic(path, data)


def test_loads_base_and_feature_blocks_in_either_spelling(config_path):
    _write(
        config_path,
        {
            "agents": {"defaults": {"model": "openai/gpt-4o", "max_tool_iterations": 9}},
            "skillForge": {"enabled": False, "router": {"overFetchFactor": 3}},
            "memory": {"userId": "alice"},
            "plugins": {"disabled": ["long-term-memory"]},
        },
    )
    config = load_config()

    assert config.agents.defaults.model == "openai/gpt-4o"
    assert config.agents.defaults.max_tool_iterations == 9
    assert config.skill_forge.enabled is False
    assert config.skill_forge.router.over_fetch_factor == 3
    assert config.memory.user_id == "alice"
    assert config.plugins.disabled == ["long-term-memory"]
    assert config.context.protect_first_n == 3


def test_missing_file_gives_defaults(config_path):
    config = load_config()
    assert config == Config()


def test_compat_names_are_the_single_root(config_path):
    assert OpenDDEHarnessConfig is Config
    assert load_opendde_harness_config is load_config


@pytest.mark.parametrize("ratio", [-1, 1.2, float("inf"), float("nan")])
def test_a_compaction_share_outside_zero_to_one_is_rejected(config_path, ratio):
    """Measured before this: a negative share or a NaN compares false against
    every prompt size, so ``prompt < ratio * window`` never held and the backend
    was asked to compact on every eligible turn; anything above one made the
    trigger unreachable. Both read as a working setting."""
    _write(config_path, {"context": {"serverCompactRatio": ratio}})
    with pytest.raises(ValueError, match="fails schema validation"):
        load_config()


@pytest.mark.parametrize("ratio", [0, 0.5, 1])
def test_a_compaction_share_inside_zero_to_one_loads(config_path, ratio):
    _write(config_path, {"context": {"serverCompactRatio": ratio}})
    assert load_config().context.server_compact_ratio == ratio


def test_unknown_key_is_rejected(config_path):
    _write(config_path, {"skillForge": {"noSuchKnob": 1}})
    with pytest.raises(ValueError, match="fails schema validation"):
        load_config()


@pytest.mark.parametrize(
    ("data", "key"),
    [
        ({"skillForge": {_library.EXECUTABLE: {"enabled": True}}}, f"skillForge.{_library.EXECUTABLE}"),
        (
            {"agents": {"defaults": {"model": "openai/gpt-5.5", "contextWindowTokens": 65536}}, "channels": {}},
            "channels",
        ),
    ],
)
def test_keys_from_retired_releases_are_rejected_by_name(config_path, data, key):
    """Nothing migrates an old config: the retired key is named and the remedy is onboard."""
    _write(config_path, data)
    with pytest.raises(ConfigSchemaError) as info:
        load_config()

    message = str(info.value)
    assert key in message
    assert "ddeharness onboard" in message
    assert json.loads(config_path.read_text()) == data


def test_env_override_prefix(config_path, monkeypatch):
    _write(config_path, {"agents": {"defaults": {"model": "anthropic/from-file"}}})
    monkeypatch.setenv("OPENDDE_HARNESS_LANGUAGE", "zh")
    monkeypatch.setenv("OPENDDE_HARNESS_AGENTS__DEFAULTS__MODEL", "anthropic/from-env")
    config = load_config()

    assert config.language == "zh"
    # A value the file sets wins over the environment.
    assert config.agents.defaults.model == "anthropic/from-file"


def test_cache_returns_isolated_copies_and_follows_the_file(config_path):
    _write(config_path, {"agents": {"defaults": {"model": "anthropic/one"}}})
    first = load_config()
    first.agents.defaults.workspace = "/edited"
    second = load_config()

    assert second.agents.defaults.workspace != "/edited"
    assert second == load_config()

    _write(config_path, {"agents": {"defaults": {"model": "anthropic/two"}}})
    assert load_config().agents.defaults.model == "anthropic/two"


def test_save_writes_base_blocks_only_and_round_trips(config_path):
    config = Config()
    config.agents.defaults.model = "openai/gpt-5.5"
    save_config(config)
    on_disk = json.loads(config_path.read_text())

    # `schemaVersion` rides with them: it is what tells the next build whether
    # this file is a shape it understands.
    assert set(on_disk) == {"schemaVersion", "agents", "tui", "cli", "providers", "tools", "language"}
    assert on_disk["agents"]["defaults"]["maxToolIterations"] == 40
    assert load_config() == config


def test_feature_blocks_round_trip_as_camel_case():
    config = Config.model_validate({"skillForge": {"alwaysMax": 4}, "runtime": {"checkpoint": {"policy": "never"}}})
    dumped = config.model_dump(by_alias=True)

    assert dumped["skillForge"]["alwaysMax"] == 4
    assert dumped["memory"]["memoryTopK"] == 5
    assert dumped["runtime"]["checkpoint"]["policy"] == "never"
    assert Config.model_validate(dumped) == config


def test_seeded_feature_defaults_load_and_patch_in_place(config_path):
    save_config(Config())
    init_extension_block_defaults()
    on_disk = json.loads(config_path.read_text())

    assert on_disk["memory"] == {"backend": None, "userId": "default", "agentId": "default", "memoryTopK": 5}
    assert on_disk["plugins"]["config"]["long-term-memory"] == {"base_url": "http://localhost:18791"}
    assert "memory" not in on_disk["skillForge"], "durable extraction is the backend's, configured under plugins"

    assert set_skill_blocked("Weather", True) == ["Weather"]
    assert load_config().skill_forge.blocklist == ["Weather"]


def test_the_retired_skill_selector_keys_are_refused_rather_than_silently_kept():
    """Nothing in the catalogue is decided by a model any more, so a config
    naming the retired rewriter / gate knobs is a config that no longer says
    what its author meant."""
    from opendde_harness.config.features import SkillForgeConfig

    for key in ("rewriteEnabled", "llmGateEnabled", "llmGateModel", "injectionMode"):
        with pytest.raises(ValueError, match=key):
            SkillForgeConfig(**{key: "x"})


def test_the_curators_keys_are_dropped_at_their_old_default_and_refused_otherwise(tmp_path):
    """Every config this project wrote carries a key for every setting the
    release that wrote it declared, so refusing on the key alone would fail
    every existing config over a value nobody chose. A value that *would* have
    changed behaviour is named instead -- and never quoted back."""
    from opendde_harness.config.loader import ConfigSchemaError, load_config

    path = tmp_path / "config.json"
    untouched = {
        "engine": "unified",
        "curatorModel": "",
        "curatorTimeoutSeconds": 60,
        "fastPathThreshold": 0.6,
        "relevanceDecay": 0.95,
        "relevanceReferenceBoost": 0.15,
        "archiveDir": "memory/.curator/archive",
        "protectFirstN": 3,
    }
    path.write_text(json.dumps({"context": untouched}))

    assert load_config(path).context.protect_first_n == 3, "the settings that survive still load"

    for key, held, said in (
        ("curatorModel", "anthropic/claude-haiku-4-5", "context.curatorModel"),
        ("curatorTimeoutSeconds", 300, "context.curatorTimeoutSeconds"),
        ("fastPathThreshold", 0.9, "context.fastPathThreshold"),
        ("archiveDir", "memory/mine", "context.archiveDir"),
        ("engine", "legacy", "context.engine"),
    ):
        # A file of its own per case: load_config caches by path and stamp, and
        # two of these values are the same length as the default they replace.
        one = tmp_path / f"{key}.json"
        one.write_text(json.dumps({"context": {**untouched, key: held}}))
        with pytest.raises(ConfigSchemaError, match=said):
            load_config(one)


def test_a_retired_service_key_never_quotes_a_credential(tmp_path):
    from opendde_harness.config.loader import ConfigSchemaError, load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"skillForge": {"embeddingApiKey": "sk-secret-value", "rerankerUrl": "http://x"}}))

    with pytest.raises(ConfigSchemaError) as excinfo:
        load_config(path)

    assert "sk-secret-value" not in str(excinfo.value)
    assert "skillForge.embeddingApiKey" in str(excinfo.value)


def test_the_extraction_block_onboarding_seeded_is_dropped_and_a_filled_one_is_refused(tmp_path):
    from opendde_harness.config.loader import ConfigSchemaError, load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"skillForge": {"memory": {"enabled": True}, "alwaysMax": 4}}))

    assert load_config(path).skill_forge.always_max == 4, "what onboarding wrote is asking for what it gets"

    path.write_text(json.dumps({"skillForge": {"memory": {"enabled": True, "maxSkillsTopK": 9}}}))
    with pytest.raises(ConfigSchemaError, match="skillForge.memory"):
        load_config(path)


def test_the_retired_agent_section_every_earlier_release_wrote_is_dropped_on_load(tmp_path):
    """``agent`` held two RPC preferences nothing read into a request; every
    release wrote the section, so a config carrying it -- at the defaults or
    not -- still loads, and the file is left as it is."""
    from opendde_harness.config.loader import load_config

    path = tmp_path / "config.json"
    path.write_text('{"agent": {"thinkingBudget": 4096, "temperature": 0.7}, "language": "en"}')

    config = load_config(path)

    assert not hasattr(config, "agent")
    assert config.language == "en"
    assert '"agent"' in path.read_text(), "never rewritten"


def test_the_retired_search_key_every_earlier_release_wrote_is_dropped_on_load(tmp_path):
    """``tools.web.search`` was written by default before web_search lost its
    key; a config from then must still load, and the file is left as it is."""
    from opendde_harness.config.loader import ConfigSchemaError, load_config

    path = tmp_path / "config.json"
    path.write_text('{"tools": {"web": {"jinaApiKey": "j", "search": {"apiKey": "", "maxResults": 5}}}}')

    config = load_config(path)

    assert config.tools.web.jina_api_key == "j"
    assert '"search"' in path.read_text(), "never rewritten"

    path.write_text('{"tools": {"web": {"searcher": {}}}}')
    with pytest.raises(ConfigSchemaError, match="tools.web.searcher"):
        load_config(path)


# ---------------------------------------------------------------------------
# What a providers entry may be: the two kinds, and the pairing rule
# ---------------------------------------------------------------------------


def test_a_name_pi_does_not_ship_has_to_declare_its_address_and_wire(config_path):
    """The key alone cannot say where such a provider is or what it speaks, so
    an entry carrying only a credential names nothing reachable.

    Refused where the config is read rather than left to the model service: the
    file is validated on machines with no Node at all (``ddeharness doctor``),
    and the two fields it still needs are named.
    """
    _write(config_path, config_dict({"brand-new": {"apiKey": "k"}}, model="brand-new/m1"))

    with pytest.raises(ConfigSchemaError) as refused:
        load_config()

    message = str(refused.value)
    assert "providers.brand-new" in message
    assert "baseUrl and api" in message
    assert "anthropic" in message, "the built-in ids are listed, so a typo for one is visible"


def test_a_near_miss_for_a_built_in_is_told_which_id_reaches_it():
    """A near-miss is a dead end otherwise: not a built-in, so it reads as a
    provider this config declares, and is then refused for having no address --
    a true sentence about the wrong problem.

    Asked of the schema rather than through the loader: this handful of names was
    also a section name of the previous shape, so a *file* carrying one is
    answered by the reshape refusal below first, and the suggestion is what every
    other reader of the schema gets.
    """
    with pytest.raises(ValueError, match="Did you mean 'google'"):
        Config.model_validate(config_dict({"gemini": {"apiKey": "k"}}, model="gemini/gemini-2.5-pro"))


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param({"baseUrl": "http://127.0.0.1:8000/v1"}, "baseUrl is set but api is not", id="address alone"),
        pytest.param({"api": "openai-completions"}, "baseUrl is not set", id="protocol alone"),
    ],
)
def test_an_address_and_the_protocol_it_serves_are_written_together(entry, expected):
    """Neither half is meaningful alone: an address needs the protocol it
    serves -- declared, never probed -- and naming a protocol is only meaningful
    for a provider this config declares, which is reached by address."""
    with pytest.raises(ValueError, match=expected):
        Config.model_validate(config_dict({"my-vllm": entry}, model="my-vllm/qwen3-32b"))


@pytest.mark.parametrize(
    "providers",
    [
        pytest.param(declared("my-vllm", api="telepathy"), id="on the provider"),
        pytest.param(declared("my-vllm", models=[{"id": "qwen3-32b", "api": "telepathy"}]), id="on one model"),
    ],
)
def test_a_wire_the_model_service_does_not_implement_is_refused_by_name(providers):
    """Named with the set that is implemented, because the value is a typo and
    the list is short: the alternative is a request that fails at the service."""
    with pytest.raises(ValueError, match="telepathy") as refused:
        Config.model_validate(config_dict(providers, model="my-vllm/qwen3-32b"))

    assert "openai-completions" in str(refused.value)


def test_both_kinds_load_side_by_side(config_path):
    """The shape the file actually has: pi's own carrying a credential, a
    declared one carrying its address, its wire and its own catalogue."""
    _write(
        config_path,
        config_dict(
            {
                **keyed("anthropic", key="sk-ant"),
                "openai-codex": {"login": "oauth"},
                **declared("my-vllm", models=[{"id": "qwen3-32b", "contextWindow": 131072}]),
            },
            model="my-vllm/qwen3-32b",
        ),
    )
    config = load_config()

    assert config.providers.get("anthropic").declared is False
    assert config.providers.get("openai-codex").login == "oauth"
    vllm = config.providers.get("my-vllm")
    assert vllm.declared is True
    assert vllm.model_ids == ["qwen3-32b"]
    assert vllm.row("qwen3-32b").context_window == 131072


# ---------------------------------------------------------------------------
# A providers section written in the shape this release replaced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("providers", "found"),
    [
        pytest.param(
            {"azure_openai": {"apiKey": "sk-LEAK", "apiBase": "https://r.openai.azure.com"}},
            "providers.azure_openai",
            id="a section name that is not a pi id",
        ),
        pytest.param({"anthropic": {"apiKey": "sk-LEAK", "wire": "responses"}}, "providers.anthropic.wire", id="wire"),
        pytest.param(
            {"anthropic": {"apiKey": "sk-LEAK", "extraHeaders": {"X-A": "b"}}},
            "providers.anthropic.extraHeaders",
            id="extraHeaders",
        ),
        pytest.param(
            {"anthropic": {"apiKey": "sk-LEAK", "modelOverlay": {"claude-sonnet-5": {"maxOutputTokens": 8192}}}},
            "providers.anthropic.modelOverlay",
            id="modelOverlay",
        ),
        pytest.param(
            {"openai": {"apiKey": "sk-LEAK", "apiBase": "https://relay.invalid/v1"}},
            "providers.openai.apiBase",
            id="apiBase",
        ),
    ],
)
def test_a_providers_section_in_the_previous_shape_is_refused_as_a_whole(config_path, providers, found):
    """One sentence for the section rather than a move per key.

    The block is keyed differently now (pi's own provider ids, not our section
    names) and its fields are pi's, so nothing in it maps across on its own --
    and the wizard is what writes the new one. Said by the loader rather than by
    a validator, because a validator's error quotes what it rejected, and what
    it rejected is a providers block holding the user's keys.
    """
    _write(config_path, {"agents": {"defaults": {"model": "anthropic/claude-sonnet-5"}}, "providers": providers})
    before = config_path.read_bytes()

    with pytest.raises(ConfigSchemaError) as refused:
        load_config()

    message = str(refused.value)
    assert "providers section has changed shape" in message
    assert found in message
    assert "ddeharness onboard" in message
    assert "sk-LEAK" not in message
    assert config_path.read_bytes() == before, "nothing is migrated, so nothing is rewritten"


def test_a_removed_provider_cannot_be_written_either(config_path):
    """pi still ships this provider, so dropping our own support is not enough:
    the write path has to refuse the name too, or the wizard would offer a
    provider whose config the loader then refuses to read."""
    from opendde_harness.config.update_providers import set_provider_fields

    with pytest.raises(KeyError, match="no longer supported"):
        set_provider_fields("github-copilot", {"api_key": "synthetic-test-key"})

    assert not config_path.exists(), "refused before anything was read or written"


@pytest.mark.parametrize("name", ["github_copilot", "github-copilot", "githubCopilot", "copilot"])
@pytest.mark.parametrize("entry", [{}, ProviderEntry().model_dump(), ProviderEntry().model_dump(by_alias=True)])
def test_removed_default_provider_is_dropped_only_in_memory(config_path, caplog, capsys, name, entry):
    import logging

    _write(config_path, {"providers": {name: entry, "openai": {"apiKey": "kept-key"}}})
    before = config_path.read_bytes()
    with caplog.at_level(logging.DEBUG, logger="opendde_harness.config.loader"):
        config = load_config()

    assert name not in config.providers
    assert config.providers.get("openai").api_key == "kept-key"
    assert config_path.read_bytes() == before
    assert "dropped retired default provider" in caplog.text
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)
    assert not capsys.readouterr().err


@pytest.mark.parametrize("name", ["github_copilot", "github-copilot", "githubCopilot", "copilot"])
@pytest.mark.parametrize(
    "entry",
    [
        {"apiKey": "sk-LEAK"},
        {"api_key": "sk-LEAK"},
        {"env": "sk-LEAK"},
        {"baseUrl": "https://example.invalid/sk-LEAK", "api": "openai-completions"},
        {"models": ["sk-LEAK"]},
        {"headers": {"Authorization": "sk-LEAK"}},
        {"endpoints": [{"label": "x", "apiKey": "sk-LEAK"}]},
        {"modelOverlay": {"sk-LEAK": {"contextWindowTokens": 32000}}},
        {"extraHeaders": {"Authorization": "sk-LEAK"}},
        {"endpointStrategy": "round_robin"},
        {"wire": "responses"},
        {"apiKey": "", "api_key": "sk-LEAK"},
        None,
        "sk-LEAK",
    ],
)
def test_removed_configured_provider_fails_without_exposing_values(config_path, caplog, name, entry):
    import traceback

    _write(config_path, {"providers": {name: entry}})
    before = config_path.read_bytes()

    with pytest.raises(ConfigSchemaError, match="GitHub Copilot support was removed") as exc:
        load_config()

    assert "Delete the providers.github-copilot" in str(exc.value)
    assert "unknown key" not in str(exc.value)
    assert "input_value" not in str(exc.value)
    assert "sk-LEAK" not in str(exc.value)
    assert "sk-LEAK" not in "".join(traceback.format_exception(exc.value))
    assert "sk-LEAK" not in caplog.text
    assert config_path.read_bytes() == before


def test_status_reports_removed_provider_error_without_credential(config_path, monkeypatch, capsys):
    import sys

    from opendde_harness.cli.commands import run

    _write(config_path, {"providers": {"githubCopilot": {"apiKey": "sk-LEAK"}}})
    monkeypatch.setattr(sys, "argv", ["ddeharness", "status"])
    with pytest.raises(SystemExit) as exc:
        run()
    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "GitHub Copilot support was removed" in captured.err
    assert "sk-LEAK" not in captured.out + captured.err


# ---------------------------------------------------------------------------
# Settings a release removed, in configs written before it
# ---------------------------------------------------------------------------


def _written(tmp_path, raw: dict):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return path


@pytest.mark.parametrize(
    ("label", "raw"),
    [
        pytest.param(
            "the shape every config carries",
            {
                "agents": {"defaults": {"modelOverrides": {}, "model": "openai-codex/gpt-5.6-luna"}},
                "providers": {"openai-codex": {"login": "oauth"}},
            },
            id="modelOverrides-empty",
        ),
        pytest.param(
            "the same key in the other spelling",
            {"agents": {"defaults": {"model_overrides": {}}}},
            id="modelOverrides-snake",
        ),
        pytest.param(
            "the pin every config carries, asking for what the id now says",
            {"agents": {"defaults": {"provider": "auto", "model": "openai/gpt-5.5"}}},
            id="provider-auto",
        ),
        pytest.param(
            "what a provider entry carried while the field existed",
            {"providers": {"deepseek": {"apiKey": "sk-not-real", "implementation": None}}},
            id="implementation-null",
        ),
        pytest.param(
            "asking for what every provider now does anyway",
            {"providers": {"deepseek": {"apiKey": "sk-not-real", "implementation": "native"}}},
            id="implementation-native",
        ),
        pytest.param(
            "a retired provider entry that nobody filled in, carrying both",
            {"providers": {"githubCopilot": {"implementation": None}, "deepseek": {"apiKey": "sk-not-real"}}},
            id="retired-section-untouched",
        ),
        pytest.param(
            "a retired provider entry as an older release actually wrote it",
            {
                "providers": {
                    "githubCopilot": {"apiKey": "", "endpoints": [], "endpointStrategy": "sticky"},
                    "deepseek": {"apiKey": "sk-not-real"},
                }
            },
            id="retired-section-with-old-defaults",
        ),
        pytest.param(
            "the empty endpoints list every section was written with",
            {"providers": {"deepseek": {"apiKey": "sk-not-real", "endpoints": []}}},
            id="endpoints-empty",
        ),
        pytest.param(
            "the only failover strategy there ever was",
            {"providers": {"deepseek": {"apiKey": "sk-not-real", "endpointStrategy": "sticky"}}},
            id="endpointStrategy-sticky",
        ),
        pytest.param(
            "the same key in the other spelling",
            {"providers": {"deepseek": {"apiKey": "sk-not-real", "endpoint_strategy": "sticky"}}},
            id="endpointStrategy-snake",
        ),
        pytest.param(
            "Google's key list, empty as every other section's was",
            {"providers": {"google": {"apiKey": "sk-not-real", "apiKeyList": []}}},
            id="apiKeyList-empty",
        ),
        pytest.param(
            "Azure's deployment field before anybody named a deployment",
            {
                "agents": {"defaults": {"model": "azure-openai-responses/gpt4o-prod"}},
                "providers": {
                    "azure-openai-responses": {
                        "apiKey": "k",
                        "baseUrl": "https://r.openai.azure.com",
                        "api": "azure-openai-responses",
                        "deployment": "",
                    }
                },
            },
            id="deployment-empty",
        ),
        pytest.param(
            "the api-version every Azure section was seeded with",
            {
                "providers": {
                    "azure-openai-responses": {
                        "apiKey": "k",
                        "baseUrl": "https://r.openai.azure.com",
                        "api": "azure-openai-responses",
                        "apiVersion": "2024-10-21",
                    }
                }
            },
            id="apiVersion-default",
        ),
        pytest.param(
            "the ten-minute call timeout every config carries",
            {"agents": {"defaults": {"llmCallTimeout": 600}}},
            id="llmCallTimeout-default",
        ),
        pytest.param(
            "the same key in the other spelling",
            {"agents": {"defaults": {"llm_call_timeout": 600}}},
            id="llmCallTimeout-snake",
        ),
        pytest.param(
            "the sampling temperature every config was seeded with",
            {"agents": {"defaults": {"temperature": 0.1}}},
            id="temperature-default",
        ),
    ],
)
def test_a_retired_setting_left_at_its_default_does_not_stop_the_program(tmp_path, monkeypatch, label, raw):
    """Refusing over a key nobody set is refusing over noise.

    Every config this project has written carries a key for every setting the
    release that wrote it declared. `agents.defaults.modelOverrides` is `{}` in
    the owner's file and in every file like it, so removing the field from the
    schema made the tool refuse to start at all -- the same failure the retired
    Copilot section caused, and fixed the same way.
    """
    monkeypatch.setattr(loader, "_current_config_path", _written(tmp_path, raw))
    loader._cache.clear()

    config = loader.load_config()

    assert config is not None, label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            {"agents": {"defaults": {"modelOverrides": {"kimi": {"temperature": 1.0}}}}},
            ["modelOverrides", "kimi", "providers.<provider>.models"],
            id="modelOverrides-set",
        ),
        pytest.param(
            {"agents": {"defaults": {"provider": "anthropic", "model": "openai/gpt-5.5"}}},
            ["agents.defaults.provider", "anthropic", "the field is gone"],
            id="provider-set",
        ),
        pytest.param(
            {"providers": {"deepseek": {"apiKey": "sk-not-real", "implementation": "legacy"}}},
            ["providers.deepseek.implementation", "legacy", "no longer exists"],
            id="implementation-legacy",
        ),
        pytest.param(
            {"providers": {"deepseek": {"endpoints": [{"label": "second-account", "apiKey": "sk-LEAK"}]}}},
            ["providers.deepseek.endpoints", "1 entry", "providers.deepseek.apiKey"],
            id="endpoints-set",
        ),
        pytest.param(
            {"providers": {"deepseek": {"apiKey": "sk-not-real", "endpointStrategy": "round_robin"}}},
            ["providers.deepseek.endpointStrategy", "round_robin", "one address per provider"],
            id="endpointStrategy-round-robin",
        ),
        pytest.param(
            {"providers": {"google": {"apiKeyList": ["sk-LEAK", "sk-LEAK-2"]}}},
            ["providers.google.apiKeyList", "2 entries", "providers.google.apiKey"],
            id="apiKeyList-set",
        ),
        pytest.param(
            {
                "providers": {
                    "azure-openai-responses": {
                        "apiKey": "k",
                        "baseUrl": "https://r/",
                        "api": "azure-openai-responses",
                        "deployment": "gpt4o-prod",
                    }
                }
            },
            ["providers.azure-openai-responses.deployment", "gpt4o-prod", "addresses it by the model id"],
            id="deployment-set",
        ),
        pytest.param(
            {
                "providers": {
                    "azure-openai-responses": {
                        "apiKey": "k",
                        "baseUrl": "https://r/",
                        "api": "azure-openai-responses",
                        "apiVersion": "2025-01-01",
                    }
                }
            },
            ["providers.azure-openai-responses.apiVersion", "2025-01-01", "api-version=v1"],
            id="apiVersion-set",
        ),
        pytest.param(
            {"agents": {"defaults": {"llmCallTimeout": 900}}},
            ["agents.defaults.llmCallTimeout", "900", "llmFirstTokenTimeout"],
            id="llmCallTimeout-set",
        ),
        pytest.param(
            {"agents": {"defaults": {"temperature": 0.7}}},
            ["agents.defaults.temperature", "0.7", "providers.<provider>.models[].temperature"],
            id="temperature-set",
        ),
    ],
)
def test_a_retired_setting_that_held_something_is_named_rather_than_dropped(tmp_path, monkeypatch, raw, expected):
    """Silently dropping an operator's setting is the other way to get this wrong.

    A value that would have changed behaviour is reported, saying what it held
    and where that goes now.
    """
    monkeypatch.setattr(loader, "_current_config_path", _written(tmp_path, raw))
    loader._cache.clear()

    with pytest.raises(loader.ConfigSchemaError) as refused:
        loader.load_config()

    for fragment in expected:
        assert fragment in str(refused.value), fragment


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            {"providers": {"deepseek": {"endpoints": [{"label": "x", "apiKey": "sk-LEAK"}]}}},
            id="endpoints",
        ),
        pytest.param({"providers": {"google": {"apiKeyList": ["sk-LEAK"]}}}, id="apiKeyList"),
    ],
)
def test_a_refused_retired_list_is_counted_rather_than_quoted(tmp_path, monkeypatch, caplog, raw):
    """Both retired lists held credentials, so neither may be printed.

    ``_summarise`` names a dict by its field names, which is safe; a list is
    named by its length alone, because the elements of these two *are* the
    secrets. The first version of this refusal put a live key in the traceback.
    """
    import traceback

    monkeypatch.setattr(loader, "_current_config_path", _written(tmp_path, raw))
    loader._cache.clear()

    with pytest.raises(loader.ConfigSchemaError) as refused:
        loader.load_config()

    assert "sk-LEAK" not in str(refused.value)
    assert "sk-LEAK" not in "".join(traceback.format_exception(refused.value))
    assert "sk-LEAK" not in caplog.text


def test_a_declared_endpoint_may_carry_a_name_the_old_shape_used(tmp_path):
    """The wizard declares an OpenAI-compatible endpoint under ``custom``, and
    ``custom`` was also a section name of the previous providers shape; the
    loader refused the wizard's own entry by its name alone. A declared entry
    carries ``baseUrl``, which the old sections never did, and that is the
    difference."""
    from opendde_harness.config.loader import ConfigSchemaError, load_config

    path = tmp_path / "config.json"
    path.write_text(
        '{"providers": {"custom": {"baseUrl": "https://relay.example/v1", "api": "openai-completions",'
        ' "apiKey": "sk-test-0000", "models": [{"id": "qwen3-32b"}]}},'
        ' "agents": {"defaults": {"model": "custom/qwen3-32b"}}}'
    )
    assert load_config(path).agents.defaults.model == "custom/qwen3-32b"

    path.write_text('{"providers": {"custom": {"apiBase": "https://relay.example/v1", "apiKey": "sk-test-0000"}}}')
    with pytest.raises(ConfigSchemaError, match="changed shape"):
        load_config(path)

    path.write_text('{"providers": {"vllm": {"apiKey": "sk-test-0000"}}}')
    with pytest.raises(ConfigSchemaError, match="changed shape"):
        load_config(path)


def test_an_empty_default_model_is_not_chosen_yet_rather_than_a_bare_id(tmp_path):
    """The wizard writes ``""`` when the provider that served the default is
    removed, and the setup gate parks on an unset model; the validator refused
    the empty string as a bare id, so every step after the removal failed at
    load and the wizard could not be finished."""
    from opendde_harness.config.loader import ConfigSchemaError, load_config

    path = tmp_path / "config.json"
    path.write_text('{"agents": {"defaults": {"model": ""}}}')
    assert load_config(path).agents.defaults.model == ""

    path.write_text('{"agents": {"defaults": {"model": "deepseek-chat"}}}')
    with pytest.raises(ConfigSchemaError, match="names no provider"):
        load_config(path)
