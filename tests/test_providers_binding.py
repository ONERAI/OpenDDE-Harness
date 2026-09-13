"""A model and its credential are one pair, scoped to the turn that entered it.

``ModelBinding`` is what a turn runs on; ``ProviderPool`` is where a model id
becomes a binding with the credential that serves it, built once per pair.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from opendde_harness.config.schema import Config
from opendde_harness.providers import model_id
from opendde_harness.providers.binding import (
    ModelBinding,
    active_binding,
    active_window,
    resolve,
    set_active_window,
    use_binding,
)
from opendde_harness.providers.pool import ProviderPool
from tests._config import config as build_config


class _Stub:
    def __init__(self, name: str = "stub") -> None:
        self.name = name

    def get_default_model(self) -> str:
        return f"{self.name}/default"


# ---------------------------------------------------------------------------
# ModelBinding and the turn context
# ---------------------------------------------------------------------------


def test_a_binding_needs_a_model() -> None:
    with pytest.raises(ValueError):
        ModelBinding(_Stub(), "")


def test_nothing_is_bound_outside_a_turn() -> None:
    assert active_binding() is None
    assert active_window(123) == 123


def test_a_binding_and_its_window_are_visible_for_the_block_and_no_longer() -> None:
    binding = ModelBinding(_Stub(), "vendor/model")
    with use_binding(binding, window=64_000):
        assert active_binding() is binding
        assert active_window(123) == 64_000
    assert active_binding() is None
    assert active_window(123) == 123


def test_an_unknown_turn_window_is_unknown_not_the_fallback() -> None:
    """None means "no table lists this model" and must not borrow the
    out-of-turn number, which belongs to another model."""
    with use_binding(ModelBinding(_Stub(), "vendor/model"), window=None):
        assert active_window(123) is None


def test_the_window_can_be_corrected_inside_the_turn_only() -> None:
    set_active_window(50)  # outside a turn: nothing to correct
    assert active_window(123) == 123
    with use_binding(ModelBinding(_Stub(), "vendor/model"), window=None):
        set_active_window(200_000)
        assert active_window(123) == 200_000
    assert active_window(123) == 123


def test_a_binding_is_restored_when_the_block_raises() -> None:
    with pytest.raises(RuntimeError):
        with use_binding(ModelBinding(_Stub(), "vendor/model"), window=1):
            raise RuntimeError("turn failed")
    assert active_binding() is None
    assert active_window(None) is None


def test_bindings_nest() -> None:
    outer = ModelBinding(_Stub("outer"), "vendor/outer")
    inner = ModelBinding(_Stub("inner"), "vendor/inner")
    with use_binding(outer, window=1):
        with use_binding(inner, window=2):
            assert active_binding() is inner and active_window(None) == 2
        assert active_binding() is outer and active_window(None) == 1


def test_resolve_prefers_a_pin_then_the_turn_then_the_fallback() -> None:
    pin = ModelBinding(_Stub("pin"), "vendor/pin")
    turn = ModelBinding(_Stub("turn"), "vendor/turn")
    fallback = ModelBinding(_Stub("boot"), "vendor/boot")

    assert resolve(None, fallback) is fallback
    assert resolve(pin, fallback) is pin
    with use_binding(turn):
        assert resolve(None, fallback) is turn
        assert resolve(pin, fallback) is pin


async def test_a_detached_task_keeps_the_binding_it_was_created_under() -> None:
    """``asyncio.create_task`` copies the context, which is what lets a
    subagent finish on the model its conversation was on when it asked."""
    first = ModelBinding(_Stub("first"), "vendor/first")
    seen: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def detached() -> None:
        started.set()
        await release.wait()
        seen.append(active_binding().model)

    with use_binding(first):
        task = asyncio.create_task(detached())
    await started.wait()
    with use_binding(ModelBinding(_Stub("second"), "vendor/second"), window=None):
        release.set()
        await task
    assert seen == ["vendor/first"]


# ---------------------------------------------------------------------------
# ProviderPool
# ---------------------------------------------------------------------------


def _config(model: str = "anthropic/claude-sonnet-4-5", **keys: str) -> Config:
    """A config whose default model is ``model``, with a key per named provider.

    The providers map is the whole route: there is no field pinning a provider
    beside the model any more, so a test says which entries exist and the id
    says which of them answers.
    """
    return build_config({name: {"apiKey": key} for name, key in keys.items()}, model=model)


@pytest.fixture
def built(monkeypatch) -> list[tuple[str, str]]:
    """Every provider the pool built, as the (provider, model) it was asked for.

    The provider is read back off the model id, because that is where the pool
    puts it: the config it hands ``make_provider`` carries one qualified id and
    nothing else that names a vendor.
    """
    calls: list[tuple[str, str]] = []

    def fake_make_provider(cfg: Config):
        pair = (model_id.provider_of(cfg.agents.defaults.model), cfg.agents.defaults.model)
        calls.append(pair)
        return SimpleNamespace(pair=pair, effort=cfg.agents.defaults.reasoning_effort)

    monkeypatch.setattr("opendde_harness.cli._helpers.make_provider", fake_make_provider)
    return calls


def test_a_model_gets_its_own_vendors_key(built) -> None:
    pool = ProviderPool(_config(anthropic="sk-ant", openai="sk-oa"))

    binding = pool.bind("openai/gpt-5-mini")

    assert binding.model == "openai/gpt-5-mini"
    assert binding.provider.pair == ("openai", "openai/gpt-5-mini")
    assert binding.provider_name == "openai"


def test_a_gateway_is_reached_by_being_named_in_the_id(built) -> None:
    """``anthropic/...`` names Anthropic. A gateway holding a key is not
    Anthropic, and reaching it means saying so in the id -- which leaves the
    gateway's own prefix in front of the vendor's id, because that remainder is
    the model OpenRouter itself serves."""
    pool = ProviderPool(_config(openrouter="sk-or"))

    binding = pool.bind("openrouter/anthropic/claude-sonnet-4-5")

    assert binding.provider.pair == ("openrouter", "openrouter/anthropic/claude-sonnet-4-5")
    assert binding.provider_name == "openrouter"
    assert pool.bind("openrouter/anthropic/claude-sonnet-4-5") is binding


def test_a_model_no_configured_provider_serves_raises_as_a_direct_build_would() -> None:
    """The real builder, because the credential gate is the thing under test."""
    from opendde_harness.providers.auth import MissingCredentialsError

    pool = ProviderPool(_config(openrouter="sk-or"))
    with pytest.raises(MissingCredentialsError):
        pool.bind("nobody/knows-this")
    # A vendor whose id names it is never served by a gateway that holds a key.
    with pytest.raises(MissingCredentialsError):
        pool.bind("anthropic/claude-sonnet-4-5")


def test_edited_generation_defaults_rebuild_the_provider(built) -> None:
    """``make_provider`` copies the effort, the retries and the two stream
    deadlines onto the provider, so a cached binding built with old values is
    stale."""
    current = _config(anthropic="sk-ant")
    pool = ProviderPool(lambda: current)
    first = pool.bind("anthropic/claude-sonnet-4-5")

    current = _config(anthropic="sk-ant")
    current.agents.defaults.reasoning_effort = "high"
    assert pool.bind("anthropic/claude-sonnet-4-5") is not first


def test_a_config_edit_landing_mid_bind_cannot_file_an_old_build_under_the_new_fingerprint(built) -> None:
    """The route, the fingerprint and the build come from one read of the
    supplier: read twice, an edit between the reads built on the old config
    and cached it under the new fingerprint, for every later bind to reuse."""
    old = _config(anthropic="sk-ant")
    new = _config(anthropic="sk-ant")
    new.agents.defaults.reasoning_effort = "high"
    reads = iter([old, new, new, new])
    pool = ProviderPool(lambda: next(reads))

    first = pool.bind("anthropic/claude-sonnet-4-5")
    second = pool.bind("anthropic/claude-sonnet-4-5")

    assert first.provider.effort == old.agents.defaults.reasoning_effort
    assert second is not first and second.provider.effort == "high"
    assert pool.bind("anthropic/claude-sonnet-4-5") is second, "the current config's build is what is reused"


def test_the_same_pair_is_built_once(built) -> None:
    pool = ProviderPool(_config(anthropic="sk-ant"))
    a = pool.bind("anthropic/claude-sonnet-4-5")
    b = pool.bind("anthropic/claude-sonnet-4-5", "anthropic")
    assert a is b
    assert built == [("anthropic", "anthropic/claude-sonnet-4-5")]


def test_two_models_of_one_vendor_are_two_bindings(built) -> None:
    pool = ProviderPool(_config(anthropic="sk-ant"))
    assert pool.bind("anthropic/claude-sonnet-4-5") is not pool.bind("anthropic/claude-opus-4-5")
    assert len(built) == 2


def test_an_explicit_provider_name_wins_over_derivation(built) -> None:
    """A gateway reselling another vendor's model bills the gateway, so the id
    it is bound under says the gateway -- the vendor's own id stays inside it as
    the model the gateway serves."""
    pool = ProviderPool(_config(anthropic="sk-ant", openrouter="sk-or"))
    binding = pool.bind("anthropic/claude-sonnet-4-5", "openrouter")
    assert binding.provider.pair == ("openrouter", "openrouter/anthropic/claude-sonnet-4-5")


def test_a_pin_with_credentials_becomes_a_pair(built) -> None:
    pool = ProviderPool(_config(anthropic="sk-ant", openai="sk-oa"))
    pin = pool.bind_pin("openai/gpt-5-mini")
    assert pin is not None and pin.provider.pair == ("openai", "openai/gpt-5-mini")


def test_a_pin_without_credentials_is_not_a_pair() -> None:
    """The real builder: an unconfigured provider is refused where every other
    caller is refused, and the pin reports that as "no pair" rather than
    raising -- it is asked while the context engine is being constructed."""
    pool = ProviderPool(_config(anthropic="sk-ant"))
    assert pool.bind_pin("openai/gpt-5-mini") is None


def test_an_unset_pin_is_not_a_pair(built) -> None:
    pool = ProviderPool(_config(anthropic="sk-ant"))
    assert pool.bind_pin(None) is None
    assert pool.bind_pin("") is None
    assert built == []


def test_a_bare_pin_is_refused_by_name_rather_than_followed(built) -> None:
    """An unresolvable choice is not an unset one. Silently treating it as "use
    the conversation's model" is how a pinned subsystem looked configured while
    never running, so the field to fix is named instead."""
    pool = ProviderPool(_config(anthropic="sk-ant"))
    with pytest.raises(ValueError, match="context.curatorModel"):
        pool.bind_pin("claude-haiku-4-5", field="context.curatorModel")
    assert built == []


def test_a_pin_whose_vendor_cannot_be_derived_is_not_a_pair() -> None:
    pool = ProviderPool(_config(anthropic="sk-ant"))
    assert pool.bind_pin("nobody/knows-this") is None


def test_a_configured_provider_beats_the_vendor_the_id_names(built) -> None:
    pool = ProviderPool(_config(anthropic="sk-ant", openrouter="sk-or"))
    pin = pool.bind_pin("anthropic/claude-haiku-4-5", "openrouter")
    assert pin is not None and pin.provider.pair == ("openrouter", "openrouter/anthropic/claude-haiku-4-5")


def test_a_configured_provider_without_credentials_is_not_a_pair(built) -> None:
    """Explicitly named and unusable is a config error, not a silent fallback
    onto whatever the id names."""
    pool = ProviderPool(_config(anthropic="sk-ant"))
    assert pool.bind_pin("anthropic/claude-haiku-4-5", "openrouter") is None
    assert built == []


def test_an_unbuildable_pin_is_reported_not_raised(monkeypatch) -> None:
    """Called at context-engine construction, so a bad pin must not stop the agent."""

    def boom(cfg: Config):
        raise RuntimeError("vendor module missing")

    monkeypatch.setattr("opendde_harness.cli._helpers.make_provider", boom)
    pool = ProviderPool(_config(anthropic="sk-ant"))
    assert pool.bind_pin("anthropic/claude-haiku-4-5") is None


def test_a_config_supplier_is_re_read_and_drops_stale_bindings(built) -> None:
    """A credential edited after start must serve the next switch, so bindings
    built from the old config are dropped."""
    current = _config(anthropic="sk-ant")
    pool = ProviderPool(lambda: current)
    first = pool.bind("anthropic/claude-sonnet-4-5")

    current = _config(anthropic="sk-ant-rotated")
    second = pool.bind("anthropic/claude-sonnet-4-5")

    assert second is not first
    assert len(built) == 2


def test_a_re_reading_supplier_still_reuses_bindings(built) -> None:
    """A supplier that returns a fresh object each call (a file re-read) must
    not defeat the cache: the fingerprint, not identity, decides staleness."""
    pool = ProviderPool(lambda: _config(anthropic="sk-ant"))
    assert pool.bind("anthropic/claude-sonnet-4-5") is pool.bind("anthropic/claude-sonnet-4-5")
    assert len(built) == 1


def test_a_credential_added_after_boot_is_visible_without_a_restart() -> None:
    current = _config(anthropic="sk-ant")
    pool = ProviderPool(lambda: current)
    assert pool.bind_pin("openai/gpt-5-mini") is None

    current = _config(anthropic="sk-ant", openai="sk-oa")
    assert pool.bind_pin("openai/gpt-5-mini") is not None
