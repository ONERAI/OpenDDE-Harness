"""Builds a :class:`ModelBinding` for a model id, and caches it.

One place answers "which credential serves this model", so a per-session
model, a subsystem pin and the configured default all resolve the same way
instead of each guessing. Caching matters because a provider learns its
model's facts -- the window, the output ceiling, the modalities, the price --
from the model service's first reply, and a session switching back and forth
would throw that away and ask again every turn.

Resolution is the model id's own prefix and nothing else: a qualified id names
its provider, and a bare one names nobody and is refused rather than routed by
its spelling. What is left to do here is build on a config copy carrying the
model being bound, because that is what ``make_provider`` reads.

Ported from Raven's ``providers/pool.py`` (#284).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from opendde_harness.providers.binding import ModelBinding

if TYPE_CHECKING:
    from opendde_harness.config.schema import Config


class ProviderPool:
    """Resolve and cache one provider per (provider name, model) pair."""

    def __init__(self, config: "Config | Callable[[], Config]") -> None:
        # A supplier, not a snapshot: a credential fixed after start (an OAuth
        # re-login, an edited config file) has to be visible without a
        # restart, which the old per-switch config reload gave for free.
        # Cached bindings are dropped when the config that produced them is no
        # longer the current one.
        self._supplier = config if callable(config) else (lambda: config)
        self._cache: dict[tuple[str, str], ModelBinding] = {}
        self._cache_key: str | None = None

    @property
    def config(self) -> "Config":
        return self._supplier()

    def _live_cache(self, config: "Config") -> dict[tuple[str, str], ModelBinding]:
        """Drop cached bindings when anything they were built from changed.

        Not identity on the config object: a supplier that re-reads the file
        returns a new object every call, which would clear the cache every
        time and defeat the pool. A fingerprint of what a binding is actually
        built from is the thing that has to match -- taken from the same
        snapshot the binding is built from, or an edit landing between the two
        reads would file a provider built on the old credential under the new
        fingerprint, and every later bind would reuse it.
        """
        fingerprint = self._cache_fingerprint(config)
        if self._cache_key != fingerprint:
            self._cache_key = fingerprint
            self._cache = {}
        return self._cache

    @staticmethod
    def _cache_fingerprint(config: "Config") -> str:
        """Everything a cached binding was built from: the providers map
        (credentials, addresses, every declared model row) and the generation
        defaults ``make_provider`` copies onto the provider (effort, the
        retries, the two stream deadlines). Not the default model: it is per
        bind, and the route is in the cache key."""
        try:
            material = {
                "providers": config.providers.model_dump(exclude_none=True),
                "defaults": config.agents.defaults.model_dump(exclude_none=True, exclude={"model"}),
            }
        except Exception:
            return ""
        return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _route(config: "Config", model: str, provider_name: str | None) -> "tuple[Config, str | None]":
        """A config that builds ``model`` on the right credential, and its provider.

        The id's prefix is the answer; an explicit ``provider_name`` qualifies a
        bare id the caller already knows the provider of (a picker selection, a
        session record). A bare id nobody named is left bare, and the build then
        refuses it rather than picking a vendor from its spelling.
        """
        from opendde_harness.providers import model_id

        cfg = config.model_copy(deep=True)
        if provider_name:
            model = model_id.join(provider_name, model)
        cfg.agents.defaults.model = model
        return cfg, model_id.provider_of(model) or None

    def bind(self, model: str, provider_name: str | None = None) -> ModelBinding:
        """Build (or reuse) the provider that serves ``model``.

        ``provider_name`` is what the caller already knows -- the picker sends
        one, a session record holds one. Absent, the id's own prefix decides
        (see ``_route``). A model nothing can serve raises the same
        ``MissingCredentialsError`` a direct build would.
        """
        return self._bind(self.config, model, provider_name)

    def _bind(self, config: "Config", model: str, provider_name: str | None) -> ModelBinding:
        """``bind`` on one snapshot of the config: routed, fingerprinted and built from the same read."""
        from opendde_harness.providers import pi_ids

        cfg, route = self._route(config, model, provider_name)
        # The one identity every reader of this binding sees: qualified by the
        # provider that serves it, exactly as the config spells its default.
        model = cfg.agents.defaults.model
        # Before anything is built. A removed provider has no entry left, so
        # the credential check further down would have reported it as merely
        # unconfigured and told the user to set a key for a provider that no
        # longer exists.
        pi_ids.refuse_removed(provider_name, route)
        cache = self._live_cache(config)
        key = (route or "auto", model)
        cached = cache.get(key)
        if cached is not None:
            return cached

        from opendde_harness.cli._helpers import make_provider

        binding = ModelBinding(make_provider(cfg), model, provider_name=route)
        cache[key] = binding
        return binding

    def bind_pin(
        self,
        model: str | None,
        provider_name: str | None = None,
        *,
        field: str = "the pinned model",
    ) -> ModelBinding | None:
        """A subsystem's own model, on its own credential -- or None.

        This is what lets a pinned subsystem run off the session's model. None
        means the pin is unusable for want of credentials, and the caller
        should fall back to the session's binding rather than send one
        vendor's key to another.

        A pin that names no provider at all is a different thing: not an unset
        choice but an unresolvable one, and it is refused here, naming the
        field to fix, rather than quietly turned into "follow the
        conversation". ``field`` is the configuration key the refusal names.

        ``provider_name`` is the configured half of the pair, and when present
        nothing is derived: an id alone cannot say whether ``anthropic`` or a
        gateway reselling it is meant, and those are different credentials and
        different bills. An explicitly named provider without credentials is a
        config error and is reported; a pin the routing cannot place is simply
        not a pair.
        """
        if not model:
            return None
        config = self.config
        if "/" not in model and not provider_name:
            raise ValueError(
                f"{field} = {model!r} names no provider. Write {field}: <provider>/{model} "
                f"(for example deepseek/{model}) -- the id is what names the provider."
            )
        _, route = self._route(config, model, provider_name)
        if provider_name:
            if route is None or not self._has_credentials(config, route):
                # Explicitly configured and still unusable: a config error the
                # user can fix, and silence here is what let a pinned
                # subsystem look configured while never running.
                logger.warning(
                    "pinned model {!r} names provider {!r}, which has no usable credentials; "
                    "the subsystem follows the conversation's model instead",
                    model,
                    provider_name,
                )
                return None
        elif route is None:
            return None
        try:
            return self._bind(config, model, route)
        except Exception as exc:
            # Called from the context-engine factory at construction, so this
            # must leave the subsystem following the conversation rather than
            # stop the agent from starting. Deliberately broad: building a
            # provider imports a vendor module, so the failure modes are not
            # only the credential ones.
            logger.warning("cannot build a provider for pinned model {!r}: {}", model, exc)
            return None

    @staticmethod
    def _has_credentials(config: "Config", provider_name: str) -> bool:
        """Can this provider be used, or is its entry missing or unfinished?

        The same check the credential preflight uses, so a pin cannot be called
        usable here and refused there.
        """
        from opendde_harness.providers.auth import credential_status

        entry = config.providers.get(provider_name)
        return entry is not None and credential_status(provider_name, entry).ok
