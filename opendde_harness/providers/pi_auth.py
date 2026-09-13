"""This project's configuration, as the model service's ``configure`` request.

The ``providers`` section is already pi's ``models.json`` shape, so there is no
translation left to do here: an entry is passed through. What remains is the
one split the service's protocol makes, which the config makes too --

* an entry with no address is one of pi's own. It contributes its key to
  ``apiKeys``, under the same id it is filed under, and nothing else: pi has the
  address, the wire and the catalogue, and a model it does not carry is not
  reachable this way. An entry with no key at all still contributes nothing and
  is not an error -- pi reads the vendor's own environment variable itself, and
  the stored credential from a sign-in outranks both;
* an entry with an address is **declared** and is sent whole, as
  ``createProvider`` input: id, address, wire, headers, key, and its models as
  rows. Under a built-in's own id that replaces pi's built-in, which is what an
  Azure resource with its own deployment names needs.

Two things this module computes rather than passes through: the sizing of a
declared model, and the compatibility flags it is sent with -- pi attaches those
per model, so a provider-level block is fanned out to its rows and the wire's
default for a declared provider fills in what nobody stated
(:func:`_row_compat`, :data:`_DECLARED_COMPAT`).

As for the sizing: a limit comes from the row the user wrote, then from pi's own
row for the same model id (the service's ``catalog`` request). A relay serving
``deepseek-v4-flash`` is serving the vendor's model, and the vendor's row is the
same upstream figure a table of ours would have carried. A limit nothing knows
is left out rather than guessed -- a number invented here would be a claim about
a deployment only its operator has measured.

Secrets travel as plain strings in the payload, because that is what goes out on
the pipe; nothing here logs, formats or returns them anywhere else.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from opendde_harness.providers import pi_ids

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opendde_harness.config.schema import Config, ModelEntry, ProviderEntry

#: What a **declared** provider is sent on a wire, unless its own entry says
#: otherwise. pi reads its compatibility flags off the base URL, and for a host
#: it has no rule for it assumes the server the protocol is named after: on
#: ``openai-completions`` that means a reasoning model's system prompt goes out
#: as ``role: "developer"`` (``convertMessages``, pi's
#: ``api/openai-completions.js``). Almost nothing that serves Chat Completions is
#: OpenAI, and a relay taking only the four standard roles answers such a
#: request with an error -- so a provider this config declares is sent the
#: standard role, and pi's guess is left to pi's own providers, where it is not a
#: guess. An entry that knows better says so in its own ``compat``.
_DECLARED_COMPAT: dict[str, dict[str, Any]] = {"openai-completions": {"supportsDeveloperRole": False}}


def declared_model_ids(config: "Config") -> list[str]:
    """Every model id a declared provider serves, as :func:`_model_row` looks it up.

    The lookup key rather than the declared id: a row's ``catalogModel`` names
    the model to read instead of a deployment name, which is not evidence of
    anything. pi's own providers are skipped -- it already has their rows.

    Exists so a caller can gather the ``catalog`` answers before building the
    payload, which is synchronous.
    """
    out: list[str] = []
    for entry in config.providers.values():
        if not entry.declared:
            continue
        for row in entry.rows():
            key = catalog_key(row)
            if key and key not in out:
                out.append(key)
    return out


def catalog_key(row: "ModelEntry") -> str:
    """The bare model id pi's catalogue is asked for on this row's behalf.

    ``catalogModel`` when the row names one -- an id that names a deployment
    rather than a model is not evidence of anything -- else the row's own id.

    A leading provider id is dropped, because a catalogue lookup is by model and
    people write the qualified form they see everywhere else. Only a leading
    *pi* id: a relay fronting OpenRouter names its models "anthropic/claude-x",
    and stripping that would ask about a model nobody serves.
    """
    key = row.catalog_model or row.id
    head, rest = key.partition("/")[0], key.partition("/")[2]
    return rest if rest and pi_ids.is_builtin(head) else key


def merged_catalog_row(rows: "Sequence[Mapping[str, Any]]") -> dict[str, Any]:
    """One row's worth of facts from every built-in row for the same model id.

    Several of pi's providers serve one id -- the vendor and every relay that
    resells it -- so a ``catalog`` lookup answers with a list. The rows agree
    about the model and differ about the price, which is not read here.

    Where they do disagree about a limit the smallest positive value wins: a
    limit is something a request has to respect, and the smallest is the only
    one true of every row. ``reasoning`` is true if any row says so and the
    modalities are unioned, because over-claiming either is refused by the
    endpoint while under-claiming loses a picture or a thinking budget in
    silence. Empty for no rows at all.
    """
    limits: dict[str, int] = {}
    reasoning = False
    modalities: list[str] = []
    for row in rows:
        for field in ("contextWindow", "maxTokens"):
            value = row.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                limits[field] = min(limits.get(field, value), value)
        reasoning = reasoning or row.get("reasoning") is True
        for kind in row.get("input") or ():
            if kind in {"text", "image"} and kind not in modalities:
                modalities.append(kind)
    merged: dict[str, Any] = dict(limits)
    if rows:
        merged["reasoning"] = reasoning
    if modalities:
        merged["input"] = modalities
    return merged


def _row_compat(row: "ModelEntry", entry: "ProviderEntry") -> dict[str, Any]:
    """The compatibility overrides this row travels with; empty for none.

    pi attaches them per model (``Model.compat``), so a block the provider wrote
    for all of its models is fanned out here to every row that carries none of
    its own -- and a row's own block replaces it rather than merging with it, so
    what one row says about itself is all that is said about it.

    Then the wire's own default for a declared provider fills in what nobody
    stated (:data:`_DECLARED_COMPAT`). Only what nobody stated: an entry saying
    ``supportsDeveloperRole: true`` is an operator telling us about their own
    relay, and it wins.
    """
    compat = dict(row.compat or entry.compat or {})
    wire = row.api or entry.api or pi_ids.DEFAULT_API
    for key, value in _DECLARED_COMPAT.get(wire, {}).items():
        compat.setdefault(key, value)
    return compat


def _model_row(
    row: "ModelEntry",
    provider: "ProviderEntry",
    catalog: "Callable[[str], Sequence[Mapping[str, Any]]] | None" = None,
) -> dict[str, Any]:
    """One declared model as the service's ``ConfigureModel``.

    The row the user wrote, then pi's own row for the same model id where the
    user left a limit out. ``catalog`` is the lookup that answers the second
    tier; without one a declared model knows only what its row says.

    A limit nothing knows is left out. pi reads a missing ``contextWindow`` as
    "do not trim", which is the honest answer. A missing ``maxTokens`` it does
    not omit, and what it does instead depends on the wire:

    * window unknown too: ``clampMaxTokensToContext`` floors the ceiling at one
      token, on every wire. A one-token reply;
    * window known, ``anthropic-messages`` or ``google-generative-ai``: the zero
      travels as the ceiling (``max_tokens: 0``, ``maxOutputTokens: 0``), which
      the vendor refuses;
    * window known, one of the three OpenAI-shaped wires: the falsy zero drops
      the field, and only here is "send no ceiling" what actually happens.

    So for a model whose ceiling resolves to nothing and whose request names
    none, the model service (``ceiling.ts``) streams the OpenAI-shaped wires
    under an unbounded window, which makes the zero drop out and the server's
    own default apply -- the request every OpenAI-compatible client sends for a
    model it has not sized -- and refuses the other wires (``no_max_tokens``),
    naming ``ddeharness provider set`` as the fix.
    """
    entry: dict[str, Any] = {"id": row.id}
    known = merged_catalog_row(catalog(catalog_key(row)) or ()) if catalog is not None else {}

    window = row.context_window or known.get("contextWindow")
    if window:
        entry["contextWindow"] = window
    ceiling = row.max_tokens or known.get("maxTokens")
    if ceiling:
        entry["maxTokens"] = ceiling
    reasoning = row.reasoning if row.reasoning is not None else known.get("reasoning")
    if reasoning is not None:
        entry["reasoning"] = bool(reasoning)
    modalities = row.input or known.get("input")
    if modalities:
        entry["input"] = list(modalities)
    if row.name:
        entry["name"] = row.name
    if row.api:
        entry["api"] = row.api
    if row.cost is not None:
        entry["cost"] = {
            "cacheRead": row.cost.cache_read,
            "cacheWrite": row.cost.cache_write,
            "input": row.cost.input,
            "output": row.cost.output,
        }
    compat = _row_compat(row, provider)
    if compat:
        entry["compat"] = compat
    return entry


def _provider_compat(entry: "ProviderEntry") -> dict[str, Any]:
    """The block every model the endpoint *publishes* travels with.

    The same rule as :func:`_row_compat` for a row that says nothing of its
    own: the entry's block, then the wire's default. A discovered model has no
    row to say anything, so this is all that is said about it.
    """
    compat = dict(entry.compat or {})
    for key, value in _DECLARED_COMPAT.get(entry.api or pi_ids.DEFAULT_API, {}).items():
        compat.setdefault(key, value)
    return compat


def _declared_provider(
    provider: str,
    entry: "ProviderEntry",
    catalog: "Callable[[str], Sequence[Mapping[str, Any]]] | None",
) -> dict[str, Any]:
    """One declared provider as the service's ``ConfigureProvider``.

    With or without declared rows: what the endpoint serves is read from the
    endpoint itself (its ``/models``), the way pi treats OpenRouter, so an entry
    naming no models is an endpoint whose list has not been fetched yet rather
    than one that serves nothing. The rows the operator declared travel as the
    baseline the fetched list overlays.
    """
    rows = [_model_row(row, entry, catalog) for row in entry.rows()]
    payload: dict[str, Any] = {
        "api": entry.api or pi_ids.DEFAULT_API,
        "baseUrl": entry.base_url,
        "id": provider,
        "models": rows,
        "name": pi_ids.display_name(provider, entry.name),
    }
    compat = _provider_compat(entry)
    if compat:
        payload["compat"] = compat
    if entry.api_key:
        payload["apiKey"] = entry.api_key
    if entry.headers:
        payload["headers"] = dict(entry.headers)
    return payload


def configure_payload(
    config: "Config",
    credentials_path: Path,
    *,
    catalog: "Callable[[str], Sequence[Mapping[str, Any]]] | None" = None,
) -> dict[str, Any]:
    """The ``configure`` params for this configuration.

    Built once and sent whole: a later call replaces the provider set rather
    than adding to it, so an entry removed from the config disappears from the
    service on the next configure.

    ``catalog`` looks a bare model id up in pi's own built-in rows, and is how a
    declared provider's models get sized (:func:`_model_row`). It is a parameter
    rather than a call from inside because this function is synchronous and the
    lookup is a request to the service: the caller gathers the rows first.
    ``pi_service._configure`` supplies it; without one, declared models carry
    only what their row declares.
    """
    api_keys: dict[str, str] = {}
    providers: list[dict[str, Any]] = []

    for provider, entry in config.providers.items():
        if not entry.declared:
            # pi's own provider: it has the address, the wire and the models. A
            # sign-in brings no key, and neither does an entry that leaves the
            # key to the vendor's environment variable -- pi resolves both.
            if entry.api_key:
                api_keys[provider] = entry.api_key
            continue
        providers.append(_declared_provider(provider, entry, catalog))

    return {"apiKeys": api_keys, "credentials": str(credentials_path), "providers": providers}


__all__ = ["catalog_key", "configure_payload", "declared_model_ids", "merged_catalog_row"]
