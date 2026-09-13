"""How a provider is connected to: what material it needs, and whether it is there.

Three shapes, and the ``providers`` entry says which without anything being
inferred:

* ``login: "oauth"`` -- a sign-in. The grant lives in the model service's
  credential store, never in ``config.json``, so the config carries nothing and
  the store is what is asked;
* a key under a **built-in** id. The config's own ``apiKey`` answers, and so do
  the two places pi resolves a key from by itself: the vendor's environment
  variable and a key already in the store. All three are reported as configured,
  because all three reach the vendor;
* an address under a **declared** id. The schema already requires ``baseUrl``
  and ``api`` of one, so a declared provider is reachable by construction; its
  key is optional, because a self-hosted server usually wants none.

Callers ask :func:`credential_status`. What they must not do is re-derive the
answer from the entry's fields at the call site, which is what produced the
divergence this module was written to end: Azure with a key and no address was
once accepted by routing and display and rejected at startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opendde_harness.providers import pi_ids

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: A credential arrives as a sign-in rather than a config field.
KIND_DEVICE_FLOW = "device_flow"
#: The environment already holds it (the AWS credential chain, Google ADC).
KIND_AMBIENT = "ambient"
#: An address is all that is needed; there is no credential.
KIND_NONE = "none"
#: A key: in the config, in the vendor's environment variable, or in the store.
KIND_API_KEY = "api_key"

#: The three shapes a caller chooses a set-up flow by. Derived from the entry
#: and from whether pi ships the id; nothing is stored per provider.
CRED_OAUTH = "oauth"  # a sign-in, run through the model service
CRED_ENDPOINT = "endpoint"  # a provider this config declares: an address, and a key only if it wants one
CRED_KEY = "key"  # one of pi's own, reached with a key


class MissingCredentialsError(Exception):
    """A provider cannot be used because its credentials are absent.

    Raised where the gate is decided, not where it is reported. The check runs
    behind three entry points -- the CLI, the gateway, and the TUI -- and used to
    end in ``console.print`` plus ``typer.Exit``, which is one of them speaking.
    Through the other two the message went to a log nobody was reading and the
    user got ``internal_error`` with ``exception_message: "1"``: the exit code,
    stringified.
    """

    def __init__(self, summary: str, *, provider: str = "", remedy: str = ""):
        super().__init__(summary)
        self.summary = summary
        self.provider = provider
        #: The command that fixes it, when there is one to name.
        self.remedy = remedy


@dataclass(frozen=True)
class Requirement:
    """One thing that must be present, named the way the user has to fix it."""

    label: str
    hint: str = ""


@dataclass(frozen=True)
class CredentialStatus:
    """The single answer to "can this provider be used right now"."""

    provider: str
    ok: bool
    kind: str
    missing: tuple[Requirement, ...] = ()
    #: Where the credential came from, for a status line that has to say.
    #: "config", "environment", "store", "declared", or "" when nothing did.
    source: str = field(default="", compare=False)

    @property
    def summary(self) -> str:
        """What to tell the user, naming the field rather than "No API key"."""
        if self.ok:
            return f"{self.provider} is configured"
        if not self.missing:
            return f"{self.provider} is not configured"
        return f"{self.provider} needs {', '.join(m.hint or m.label for m in self.missing)}"


def _value(entry: Any, *names: str) -> str:
    """One field of an entry, whether it arrived as a model or as raw JSON.

    Entries reach here as both: the schema object on the routing path, the
    file's own mapping on the display path, and the mapping carries the
    camelCase the file is written in.
    """
    if entry is None:
        return ""
    for name in names:
        value = entry.get(name) if isinstance(entry, dict) else getattr(entry, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def configured_key(entry: Any) -> str:
    """The key this entry supplies, in every spelling it may use."""
    return _value(entry, "api_key", "apiKey")


def configured_login(entry: Any) -> str:
    """The sign-in this entry names, or "" when it names none."""
    return _value(entry, "login")


def env_key_name(provider: str) -> str:
    """The environment variable pi would read this provider's key from, if set.

    Reported only -- the key itself never travels through this project (see
    :data:`pi_ids.ENV_KEYS`). Empty when none is set, so the caller can say
    "no key" rather than naming a variable that is not there.
    """
    for name in pi_ids.ENV_KEYS.get(provider, ()):
        if (os.environ.get(name) or "").strip():
            return name
    return ""


def has_credentials(provider: str) -> bool:
    """Is a sign-in stored for this provider right now?

    Asked of pi's credential store, which is where a sign-in lands: the login
    runs through the model service, and the service is what refreshes the token
    afterwards. Expiry is not part of the question -- an expired access token
    with a refresh token beside it is a working sign-in -- and neither is
    validity at the vendor, which only a request can answer.
    """
    from opendde_harness.providers.pi_credentials import stored_oauth

    return stored_oauth(provider)


def _stored_key(provider: str) -> bool:
    from opendde_harness.providers.pi_credentials import stored_kind

    return stored_kind(provider) == "api_key"


def credential_kind(provider: str, entry: Any = None) -> str:
    """Which of the three shapes this provider is set up as.

    Every decision about how a provider is configured follows from this:
    whether to run a sign-in, to ask for an address, or to ask for a key. It is
    derived from the entry and from whether pi ships the id, so there is no
    table to keep in step with pi's.
    """
    if configured_login(entry) == "oauth":
        return CRED_OAUTH
    if _value(entry, "base_url", "baseUrl"):
        # Asked before pi's catalogue is consulted, because an address is what
        # makes an entry declared -- a built-in id carrying an Azure resource
        # URL is reached by that address and not by pi's own.
        return CRED_ENDPOINT
    if provider in pi_ids.OAUTH and provider not in pi_ids.ENV_KEYS:
        # pi offers this one no key at all, so an entry that names no `login`
        # is still a sign-in -- asking for a key would be asking for something
        # the vendor does not issue. Every other OAuth provider also takes a
        # key, and there the entry is what says which of the two was chosen.
        return CRED_OAUTH
    if pi_ids.is_builtin(provider):
        return CRED_KEY
    return CRED_ENDPOINT


_SIGN_IN = "a sign-in"
_A_KEY = "an API key"
_AN_ADDRESS = "an address"


def credential_status(
    provider: str,
    entry: Any,
    *,
    include_external: bool = False,
) -> CredentialStatus:
    """Can this provider be used, and if not, what exactly is missing?

    ``entry`` is the providers entry -- the schema object or the raw mapping,
    either way; ``None`` means there is no entry at all, which is its own answer.

    ``include_external`` decides whether material held outside the config is
    looked at: the credential store and the vendor's environment variable.
    Routing asks without it -- it runs on every call, and reading a file there
    would put disk I/O on the hot path and make the choice depend on a sign-in
    that startup is the right place to require. Display and startup ask with it,
    because both report on what is true right now.
    """
    if entry is None:
        return CredentialStatus(provider, False, KIND_API_KEY, (Requirement("an entry", _entry_hint(provider)),))

    kind = credential_kind(provider, entry)

    if kind == CRED_OAUTH:
        if not include_external:
            # A sign-in is legitimately invisible from the config. Routing has
            # to let it through and let startup report it.
            return CredentialStatus(provider, True, KIND_DEVICE_FLOW, (), "store")
        if has_credentials(provider):
            return CredentialStatus(provider, True, KIND_DEVICE_FLOW, (), "store")
        public = provider.replace("_", "-")
        return CredentialStatus(
            provider,
            False,
            KIND_DEVICE_FLOW,
            (Requirement(_SIGN_IN, f"{_SIGN_IN} -- run `ddeharness provider login {public}`"),),
        )

    if kind == CRED_ENDPOINT:
        # The schema requires baseUrl and api of a declared provider, so the
        # only way to be here without one is a raw mapping that never went
        # through it. Say so rather than trusting the shape.
        if _value(entry, "base_url", "baseUrl"):
            return CredentialStatus(provider, True, KIND_NONE, (), "declared")
        return CredentialStatus(
            provider,
            False,
            KIND_NONE,
            (Requirement(_AN_ADDRESS, f"{_AN_ADDRESS} -- run `ddeharness provider set {provider} --base-url <url>`"),),
        )

    if configured_key(entry):
        return CredentialStatus(provider, True, KIND_API_KEY, (), "config")
    if provider in pi_ids.AMBIENT:
        # The credential is a whole environment chain; asking for a key would
        # be asking for something the user does not have in this form.
        return CredentialStatus(provider, True, KIND_AMBIENT, (), "environment")
    if include_external:
        if env_key_name(provider):
            return CredentialStatus(provider, True, KIND_API_KEY, (), "environment")
        if _stored_key(provider) or (provider in pi_ids.OAUTH and has_credentials(provider)):
            return CredentialStatus(provider, True, KIND_API_KEY, (), "store")
    return CredentialStatus(
        provider,
        False,
        KIND_API_KEY,
        (Requirement(_A_KEY, f"{_A_KEY} -- run `ddeharness provider set {provider} --api-key <key>`"),),
    )


def _entry_hint(provider: str) -> str:
    """What to tell somebody whose config has no entry for this provider."""
    meant = pi_ids.suggestion(provider)
    if meant:
        return f"a providers.{meant} entry -- {provider!r} is not a pi provider id, {meant!r} is"
    if pi_ids.is_builtin(provider):
        return f"a providers.{provider} entry -- run `ddeharness provider set {provider} --api-key <key>`"
    return (
        f"a providers.{provider} entry declaring its baseUrl and api -- {provider!r} is not one of pi's "
        f"built-in providers, so this config has to declare it"
    )


def key_refusal(provider: str) -> str | None:
    """Why a key cannot configure this name, or ``None`` if it can.

    Checked before a provider entry is written for a name the config has never
    seen. Every reason is the same one now: the name is a near-miss for a pi
    provider id, and writing it would declare a provider nobody serves.
    """
    meant = pi_ids.suggestion(provider)
    if meant is None:
        return None
    return (
        f"{provider!r} is not a pi provider id. {meant!r} is the one that reaches it: "
        f"run `ddeharness provider set {meant} --api-key <key>`, or pick it from `ddeharness onboard`."
    )


__all__ = [
    "CRED_ENDPOINT",
    "CRED_KEY",
    "CRED_OAUTH",
    "KIND_AMBIENT",
    "KIND_API_KEY",
    "KIND_DEVICE_FLOW",
    "KIND_NONE",
    "CredentialStatus",
    "MissingCredentialsError",
    "Requirement",
    "configured_key",
    "configured_login",
    "credential_kind",
    "credential_status",
    "env_key_name",
    "has_credentials",
    "key_refusal",
]
