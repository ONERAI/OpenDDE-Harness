"""This project's config and credentials, as the model service's ``configure``.

Everything here is synthetic and lands in ``tmp_path``: no real credential
directory is read, and no environment key is allowed to answer for one.
"""

from __future__ import annotations

import os

import pytest

from opendde_harness.config.schema import Config
from opendde_harness.providers.pi_auth import configure_payload
from tests._config import config as build_config
from tests._config import declared, keyed


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch):
    """No vendor key in the environment answers for an entry in these tests.

    pi resolves the vendor's own variable itself, and this suite is about what
    the *config* says. A developer's exported key would otherwise put a real
    secret in an asserted payload.
    """
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_AUTH_TOKEN")) or name.startswith("CHATGPT_"):
            monkeypatch.delenv(name, raising=False)


def config() -> Config:
    """One of pi's own, one declared OpenAI-compatible, one on another wire."""
    return build_config(
        {
            **keyed("anthropic", key="sk-ant-synthetic"),
            **declared(
                "my-vllm",
                base_url="http://127.0.0.1:8000/v1",
                api="openai-completions",
                apiKey="sk-vllm-synthetic",
                headers={"X-Tenant": "lab"},
                models=[
                    {"id": "qwen3-32b", "name": "Qwen 32B", "contextWindow": 40960, "maxTokens": 8192},
                    "deepseek-v3",
                ],
            ),
            **declared(
                "plan-relay",
                base_url="https://relay.invalid/anthropic/v1",
                api="anthropic-messages",
                apiKey="sk-plan-synthetic",
                models=["MiniMax-M3"],
            ),
        },
        model="my-vllm/qwen3-32b",
    )


def test_one_of_pis_own_becomes_a_key_and_a_declared_one_becomes_a_provider(tmp_path):
    payload = configure_payload(config(), tmp_path / "auth.json")

    assert payload["credentials"] == str(tmp_path / "auth.json")
    # Anthropic is pi's own provider: it takes the key and nothing else.
    assert payload["apiKeys"] == {"anthropic": "sk-ant-synthetic"}
    assert not any(provider["id"] == "anthropic" for provider in payload["providers"])

    vllm = next(p for p in payload["providers"] if p["id"] == "my-vllm")
    assert vllm["api"] == "openai-completions"
    assert vllm["baseUrl"] == "http://127.0.0.1:8000/v1"
    assert vllm["apiKey"] == "sk-vllm-synthetic"
    assert vllm["headers"] == {"X-Tenant": "lab"}
    # Every model the entry declares, written either way round.
    assert [model["id"] for model in vllm["models"]] == ["qwen3-32b", "deepseek-v3"]


def test_a_row_states_the_limits_no_catalogue_carries(tmp_path):
    vllm = next(p for p in configure_payload(config(), tmp_path / "auth.json")["providers"] if p["id"] == "my-vllm")
    row = next(m for m in vllm["models"] if m["id"] == "qwen3-32b")

    assert row["contextWindow"] == 40960
    assert row["maxTokens"] == 8192
    assert row["name"] == "Qwen 32B"
    # And a model with neither a row nor a catalogue lookup carries no limits at
    # all. pi reads a missing window as "do not trim" and a missing ceiling as
    # "send none", which are the honest answers; a number here would be a claim
    # about a deployment only its operator has measured.
    unsized = next(m for m in vllm["models"] if m["id"] == "deepseek-v3")
    assert "maxTokens" not in unsized and "contextWindow" not in unsized


def test_a_declared_model_is_sized_from_pis_row_for_the_same_id(tmp_path):
    """A relay serving ``deepseek-v3`` is serving the vendor's model, and the
    vendor's row is the same upstream figure a table of ours carried.

    The lookup is handed in rather than made here: building the payload is
    synchronous and the answer is a request to the service, so the caller
    gathers the rows first (``pi_service._declared_catalog``).
    """
    rows = {
        "deepseek-v3": [
            {"contextWindow": 128_000, "input": ["text"], "maxTokens": 8_192, "reasoning": True},
            # A second provider resells it under a shorter window; the smaller
            # limit is the only one true of both.
            {"contextWindow": 64_000, "input": ["text", "image"], "maxTokens": 8_192, "reasoning": False},
        ]
    }
    payload = configure_payload(config(), tmp_path / "auth.json", catalog=rows.get)
    vllm = next(p for p in payload["providers"] if p["id"] == "my-vllm")
    sized = next(m for m in vllm["models"] if m["id"] == "deepseek-v3")

    assert sized["contextWindow"] == 64_000
    assert sized["maxTokens"] == 8_192
    assert sized["reasoning"] is True
    assert sized["input"] == ["text", "image"]
    # The row still wins for the model that has one: the operator is the
    # authority on their own deployment.
    declared_row = next(m for m in vllm["models"] if m["id"] == "qwen3-32b")
    assert (declared_row["contextWindow"], declared_row["maxTokens"]) == (40960, 8192)


def test_every_declared_model_id_is_offered_for_lookup(tmp_path):
    """What ``_declared_catalog`` asks about: the bare id each declared entry
    serves. pi's own providers are skipped -- it already has their rows."""
    from opendde_harness.providers.pi_auth import declared_model_ids

    assert declared_model_ids(config()) == ["qwen3-32b", "deepseek-v3", "MiniMax-M3"]


def test_a_declared_entry_says_which_wire_it_is_served_on(tmp_path):
    """``api`` travels as declared. There is no per-vendor table saying which
    wire a name speaks and nothing is probed, so the entry is the only source."""
    payload = configure_payload(config(), tmp_path / "auth.json")
    relay = next(p for p in payload["providers"] if p["id"] == "plan-relay")

    assert relay["apiKey"] == "sk-plan-synthetic"
    assert relay["api"] == "anthropic-messages"
    assert relay["baseUrl"] == "https://relay.invalid/anthropic/v1"
    assert [model["id"] for model in relay["models"]] == ["MiniMax-M3"]


def test_a_full_row_arrives_with_pis_own_field_names(tmp_path):
    """The section is pi's ``models.json`` shape, so an entry is passed through.

    Asserted field by field because the payload is the wire: a row's name, its
    two limits, its modalities, whether it thinks and what it costs all have to
    arrive under the names pi's ``ConfigureModel`` reads, and the three fields
    that are this project's own (``reasoningEffort``, ``temperature``,
    ``catalogModel``) must not travel at all -- pi has nowhere to put them.
    """
    config = build_config(
        {
            **keyed("google", key="sk-google-synthetic"),
            **declared(
                "lab",
                base_url="https://lab.invalid/v1",
                api="openai-responses",
                apiKey="sk-lab-synthetic",
                headers={"X-Tenant": "lab"},
                models=[
                    {
                        "id": "lab-1",
                        "name": "Lab One",
                        "description": "the box in the corner",
                        "contextWindow": 131072,
                        "maxTokens": 32768,
                        "reasoning": True,
                        "input": ["text", "image"],
                        "cost": {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 3.75},
                        "reasoningEffort": "high",
                        "temperature": 0.7,
                        "catalogModel": "openai/gpt-4o",
                    }
                ],
            ),
        },
        model="lab/lab-1",
    )

    payload = configure_payload(config, tmp_path / "auth.json")

    # The built-in contributes its key and nothing else.
    assert payload["apiKeys"] == {"google": "sk-google-synthetic"}
    lab = payload["providers"][0]
    assert lab["id"] == "lab"
    assert lab["apiKey"] == "sk-lab-synthetic"
    assert lab["headers"] == {"X-Tenant": "lab"}
    assert lab["models"] == [
        {
            "id": "lab-1",
            "contextWindow": 131072,
            "maxTokens": 32768,
            "reasoning": True,
            "input": ["text", "image"],
            "name": "Lab One",
            "cost": {"cacheRead": 0.3, "cacheWrite": 3.75, "input": 3, "output": 15},
        }
    ]


def test_a_declared_entry_with_no_models_travels_so_the_endpoint_can_be_asked(tmp_path):
    """What a declared endpoint serves is read from the endpoint itself (its
    ``/models``), the way pi treats OpenRouter, so an entry naming no models is
    one whose list has not been fetched yet -- it is declared, with its key and
    the compat block every model it publishes travels with."""
    empty = build_config(
        declared("my-vllm", models=[], apiKey="sk-vllm-synthetic"),
        model="my-vllm/solo",
    )
    payload = configure_payload(empty, tmp_path / "auth.json")

    assert payload["apiKeys"] == {}
    [provider] = payload["providers"]
    assert provider["id"] == "my-vllm"
    assert provider["models"] == []
    assert provider["apiKey"] == "sk-vllm-synthetic"
    # The wire's default, the same block a declared row without one gets.
    assert provider["compat"] == {"supportsDeveloperRole": False}


# ---------------------------------------------------------------------------
# Compatibility overrides
# ---------------------------------------------------------------------------


def test_a_declared_chat_completions_provider_is_sent_the_standard_system_role(tmp_path):
    """pi's own detection is a guess for a host it has no rule for, and on this
    wire the guess is OpenAI: a reasoning model's system prompt would go out as
    ``role: "developer"``, which a relay taking only the standard roles answers
    with an error and no ``finish_reason``. So every row of a declared
    ``openai-completions`` entry carries the flag, whether the row is a full one
    or a bare id."""
    payload = configure_payload(config(), tmp_path / "auth.json")
    vllm = next(p for p in payload["providers"] if p["id"] == "my-vllm")

    assert [row.get("compat") for row in vllm["models"]] == [
        {"supportsDeveloperRole": False},
        {"supportsDeveloperRole": False},
    ]
    # Only that wire. pi's Anthropic adapter has no such role to get wrong, and
    # a flag it does not read would be a claim we cannot support.
    relay = next(p for p in payload["providers"] if p["id"] == "plan-relay")
    assert all("compat" not in row for row in relay["models"])


def test_a_declared_compat_block_travels_as_written(tmp_path):
    """The block is pi's own shape, so it is passed through -- and an entry that
    says ``supportsDeveloperRole`` itself is an operator describing their own
    relay, which outranks the wire's default."""
    written = build_config(
        declared(
            "my-vllm",
            api="openai-completions",
            compat={"supportsDeveloperRole": True, "maxTokensField": "max_tokens"},
            models=["qwen3-32b", {"id": "glm-5", "compat": {"thinkingFormat": "zai"}}],
        ),
        model="my-vllm/qwen3-32b",
    )
    rows = configure_payload(written, tmp_path / "auth.json")["providers"][0]["models"]

    # pi keeps compat per model, so the provider's block is fanned out to every
    # row that carries none of its own.
    assert rows[0]["compat"] == {"supportsDeveloperRole": True, "maxTokensField": "max_tokens"}
    # A row's own block replaces the provider's rather than merging with it, and
    # the wire's default then fills in only what nobody stated.
    assert rows[1]["compat"] == {"thinkingFormat": "zai", "supportsDeveloperRole": False}


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param(
            declared("my-vllm", api="openai-completions", compat={"supportsDevloperRole": False}),
            ["'supportsDevloperRole'", "openai-completions", "supportsDeveloperRole"],
            id="misspelled-key",
        ),
        pytest.param(
            declared("my-vllm", api="openai-completions", compat={"supportsTemperature": False}),
            ["'supportsTemperature'", "openai-completions"],
            id="another-wires-key",
        ),
        pytest.param(
            declared("my-vllm", api="openai-completions", models=[{"id": "m", "compat": {"nope": True}}]),
            ["'nope'", "openai-completions"],
            id="on-a-row",
        ),
        pytest.param(
            declared("relay", api="google-generative-ai", base_url="https://r.invalid", compat={"supportsStore": True}),
            ["reads none on google-generative-ai"],
            id="a-wire-with-no-compat",
        ),
        pytest.param(
            {"anthropic": {"apiKey": "sk-ant-synthetic", "compat": {"supportsTemperature": False}}},
            ["declares no baseUrl", "detects the compatibility of its own providers"],
            id="one-of-pis-own",
        ),
    ],
)
def test_a_compat_field_pi_does_not_read_is_refused_by_name(entry, expected):
    """The block travels to pi untouched, so a field pi has no slot for would be
    sent and ignored -- a typo that costs a turn and says nothing. Refused where
    the config is read, against the wire it was written under."""
    import pydantic

    provider = next(iter(entry))
    with pytest.raises(pydantic.ValidationError) as refused:
        build_config(entry, model=f"{provider}/m")

    for fragment in expected:
        assert fragment in str(refused.value), fragment
