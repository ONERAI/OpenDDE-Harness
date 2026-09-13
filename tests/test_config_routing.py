"""Which provider answers a model id: the one it names, and no other.

The provider is part of the model's identity, and the id's own prefix is the
only thing that names it. A bare id names nobody: the schema refuses one as the
configured default, and routing answers nothing for one it is asked about --
never a guess from the spelling, which is how a prompt and an entry's key went
to a vendor the user never chose for that model.
"""

import pytest

from opendde_harness.config.schema import Config
from opendde_harness.providers import model_id
from opendde_harness.providers.auth import MissingCredentialsError
from tests._config import config as build_config
from tests._config import config_dict, declared, keyed

#: A provider this config declares: an address, the protocol it serves, a key.
RELAY = declared("corp-relay", base_url="https://llm.corp.internal/v1", apiKey="sk-corp")
#: A declared one with no key at all, which is what a self-hosted server wants.
LOCAL = declared("my-ollama", base_url="http://127.0.0.1:11434/v1")
#: Two of pi's own, reached with a key and nothing else.
GATEWAY = keyed("openrouter", key="sk-or")
OPENAI = keyed("openai", key="sk-openai")

#: A default this config can always serve, so a test about the model it *asks*
#: about is not also a test about the one it is configured with.
DEFAULT = "openai/gpt-5.5"


def _route(providers, model):
    """The provider id that answers ``model``, or None when nothing does."""
    return build_config(providers, model=DEFAULT)._match_provider(model)[1]


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5",  # names OpenAI by keyword, if a keyword table were consulted
        "my-gptq-model",  # a quantised local model whose name happens to contain "gpt"
        "qwen3-32b",  # would have gone to DashScope, or to the local box
        "kimi-k2.5",
        "llama-3.3-70b",  # names nobody at all
    ],
)
def test_a_bare_id_names_no_provider_however_it_is_spelled(model):
    """Every entry here is configured, and none of them is chosen."""
    assert _route({**RELAY, **LOCAL, **GATEWAY, **OPENAI}, model) is None


def test_a_bare_default_model_is_refused_rather_than_routed():
    """The same rule, said where a config file is read: a default naming no
    provider is a config error, not a model to find an owner for."""
    with pytest.raises(ValueError, match="names no provider"):
        Config.model_validate(config_dict(keyed("openai"), model="gpt-5.5"))


def test_the_prefix_names_the_entry():
    assert _route(RELAY, "corp-relay/gpt-5.5") == "corp-relay"
    assert _route(GATEWAY, "openrouter/anthropic/claude-sonnet-4.5") == "openrouter"
    assert _route(LOCAL, "my-ollama/qwen3-32b") == "my-ollama"


def test_a_vendor_without_credentials_is_not_served_by_a_gateway_that_has_some():
    """`anthropic/...` names Anthropic. A gateway holding a key is not Anthropic,
    and the id has to say `openrouter/anthropic/...` to mean the gateway."""
    assert _route({**RELAY, **GATEWAY}, "anthropic/claude-sonnet-5") is None
    assert _route({**RELAY, **GATEWAY}, "openrouter/anthropic/claude-sonnet-5") == "openrouter"


def test_a_declared_endpoint_answers_only_when_it_is_named():
    assert _route(RELAY, "anthropic/claude-sonnet-5") is None
    assert _route(LOCAL, "anthropic/claude-sonnet-5") is None


def test_the_vendor_itself_answers_when_it_has_credentials():
    providers = {**RELAY, **GATEWAY, **keyed("anthropic", key="sk-ant")}
    assert _route(providers, "anthropic/claude-sonnet-5") == "anthropic"


def test_one_of_pis_own_is_reached_by_its_own_entry():
    """pi carries the address, the wire and the catalogue, so an entry under the
    vendor's own id holding a key is the whole route."""
    assert _route(keyed("mistral", key="sk-mistral"), "mistral/mistral-large-latest") == "mistral"
    assert _route(RELAY, "mistral/mistral-large-latest") is None


def test_the_ids_this_projects_own_onboarding_writes_all_route():
    """What the wizard stores is always provider-qualified; every shape it
    writes must land on the entry the user just configured."""
    cases = [
        (GATEWAY, "openrouter", "anthropic/claude-sonnet-4.5"),
        (RELAY, "corp-relay", "deepseek-chat"),
        (LOCAL, "my-ollama", "qwen3:32b"),
        (OPENAI, "openai", "gpt-5.5"),
        ({"openai-codex": {"login": "oauth"}}, "openai-codex", "gpt-5.6-luna"),
    ]
    for providers, provider, typed in cases:
        stored = model_id.join(provider, typed)
        assert stored == f"{provider}/{typed}", stored
        assert _route(providers, stored) == provider, (provider, stored)


def test_a_refused_bare_id_is_told_how_to_name_a_provider():
    config = build_config({**RELAY, **GATEWAY}, model=DEFAULT)
    why = config.explain_unrouted("gpt-5.5")

    assert "names no provider" in why
    assert "openrouter/gpt-5.5" in why
    assert "provider use <provider>/gpt-5.5" in why


def test_startup_refuses_a_bare_id_rather_than_guessing():
    """The gate is the second place that must not guess. A file cannot hold a
    bare default -- the schema refuses it -- so the id is put there directly,
    which is also what an in-session model switch can do."""
    from opendde_harness.cli._helpers import check_provider_credentials

    config = build_config({**RELAY, **GATEWAY, **OPENAI}, model=DEFAULT)
    config.agents.defaults.model = "my-gptq-model"

    with pytest.raises(MissingCredentialsError) as caught:
        check_provider_credentials(config)
    assert "names no provider" in str(caught.value)
    assert "openrouter/my-gptq-model" in str(caught.value)


def test_a_prefixed_id_whose_provider_is_not_configured_reports_that_entry():
    config = build_config(RELAY, model=DEFAULT)
    assert "providers.anthropic" in config.explain_unrouted("anthropic/claude-sonnet-5")
