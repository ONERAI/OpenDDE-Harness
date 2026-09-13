"""Model identity: "<pi provider id>/<model id>", split and joined.

A model is chosen in one place and read about in another, so the spelling has
to be decided once. It is decided by pi: a provider id is pi's own id, it has
exactly one spelling, and a model id is whatever the endpoint serves. So there
is nothing to canonicalize here and nothing to compare loosely -- the whole
module is :func:`split` and :func:`join`, which is what is left of a file that
used to reconcile a vendor's own name, a snake_case field name and a camelCase
key for every provider we carried.

The split is on the FIRST separator only. A gateway's ids carry the upstream
vendor in their own name ("openrouter/anthropic/claude-opus-4-5"), and that
remainder is the model id the gateway serves, not a second prefix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opendde_harness.config.schema import ModelEntry, ProviderEntry, ProvidersConfig


def split(model: str | None) -> tuple[str, str]:
    """``("<provider>", "<model>")``, or ``("", model)`` for an id naming nobody.

    A bare id is a different case from an unknown provider: it makes no claim
    about who serves it, and every caller refuses it rather than guessing.
    """
    head, sep, rest = (model or "").partition("/")
    if not sep:
        return "", model or ""
    return head, rest


def join(provider: str | None, model: str) -> str:
    """``"<provider>/<model>"``, leaving an id that already names ``provider`` alone.

    Idempotent on the provider, not on the separator. A gateway's model ids
    carry the upstream vendor in their own name -- OpenRouter serves
    "anthropic/claude-opus-4.8" -- so "has a slash" cannot mean "already
    qualified": reading it that way stored that id bare, and a bare
    "anthropic/..." names Anthropic direct, on Anthropic's key.
    """
    if not model or not provider or provider_of(model) == provider:
        return model
    return f"{provider}/{model}"


def provider_of(model: str | None) -> str:
    """The provider this id names, or "" if it names none."""
    return split(model)[0]


def bare(model: str | None) -> str:
    """The id the endpoint is asked for: the remainder after the provider."""
    return split(model)[1]


def display(provider: str, model: str | None) -> str:
    """``model`` with a prefix naming ``provider`` dropped, for a scoped list.

    What is stored stays qualified -- routing depends on it -- but a picker
    that has already named the provider repeats it on every row for nothing.
    """
    head, rest = split(model)
    return rest if head and head == provider else (model or "")


def row_for(providers: "ProvidersConfig | None", model: str | None) -> "ModelEntry | None":
    """The declared row for this qualified id, or None if there is none.

    An exact lookup: the prefix is a key of ``providers`` and the remainder is
    a row's ``id``. Nothing is matched loosely, so a row declared for a
    self-hosted model can never answer for a hosted one of the same name.

    Bare string entries carry no facts, so they answer None -- the row exists
    for what the user knows beyond the id.
    """
    if not providers or not model:
        return None
    provider, wanted = split(model)
    entry = providers.get(provider) if provider else None
    if entry is None or not wanted:
        return None
    return entry.row(wanted)


def rows_of(entry: "ProviderEntry | None") -> "list[ModelEntry]":
    """Every declared model of an entry as a row, bare strings included."""
    return list(entry.rows()) if entry is not None else []


__all__ = ["bare", "display", "join", "provider_of", "row_for", "rows_of", "split"]
