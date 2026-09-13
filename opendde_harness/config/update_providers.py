"""Atomic operations for the ``providers`` section of ``config.json``.

This module is the ONLY write path for provider configuration. All entry points
(CLI commands, the wizard, the TUI's RPC) must call functions defined here.
Direct ``load_config`` / ``save_config`` on the providers section is forbidden.

The section is pi's ``models.json`` shape: a map from pi provider id to a
provider declaration, and one entry is there because somebody wrote it. That
makes every read here an exact dictionary lookup -- there are no aliases, no
camelCase-versus-snake_case spellings and no always-present empty blocks, which
is what the reconciliation this file used to carry was for.

A provider whose credential is a sign-in has a separate auth path --
``provider login``, which runs pi-ai's own flow through the model service -- and
its grant lives in the service's credential store, not in ``config.json``.
``set_provider_fields`` refuses to write ``apiKey`` for those unless the same
write turns the sign-in off, which is how a key replaces a sign-in.
``reset_provider`` handles both: it signs the provider out when the entry named
a sign-in, then deletes the entry.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from opendde_harness.config._fields import (
    coerce_value,
    flatten_fields,
    flatten_instance,
    set_nested,
    walk_nested_path,
    write_json_atomic,
)
from opendde_harness.config.loader import get_config_path, read_raw_or_raise
from opendde_harness.config.schema import ModelEntry, ProviderEntry, ProvidersConfig
from opendde_harness.providers import model_id, pi_ids

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def configured_provider_names(data: dict[str, Any]) -> list[str]:
    """Every provider this config holds an entry for, in the file's own order.

    There is no set of "declared" providers to list beside them any more: pi's
    built-in ids are what the wizard offers (:data:`pi_ids.BUILTIN`), and the
    config holds the ones that were chosen.
    """
    stored = data.get("providers")
    if not isinstance(stored, dict):
        return []
    return [name for name, entry in stored.items() if isinstance(entry, dict)]


def _refuse_unwritable(provider: str) -> None:
    """Stop before an entry is written for a name that can never be one.

    A provider this project removed is gone whatever is written for it, and a
    string that is not a provider id at all cannot be a key. A near-miss for a
    pi id is deliberately *not* refused here: it is a perfectly good name for a
    provider this config declares, and the one case where it is a mistake --
    naming it with no address -- is refused by :func:`_commit` with the
    schema's own sentence, which already says which id was probably meant.
    """
    reason = pi_ids.removed_message(provider)
    if reason:
        raise KeyError(f"'{provider}' is no longer supported. {reason}")
    if not provider or "/" in provider or provider != provider.strip():
        raise KeyError(f"'{provider}' is not a provider id: a pi provider id carries no slash and no spaces.")


def _entry_dict(data: dict[str, Any], provider: str) -> dict[str, Any]:
    """This provider's entry as raw JSON, or ``{}`` when there is none."""
    stored = data.get("providers")
    entry = stored.get(provider) if isinstance(stored, dict) else None
    return dict(entry) if isinstance(entry, dict) else {}


def _entry(data: dict[str, Any], provider: str) -> ProviderEntry:
    """This provider's entry, parsed; a fresh one when it is absent or broken."""
    try:
        return ProviderEntry.model_validate(_entry_dict(data, provider))
    except ValidationError:
        return ProviderEntry()


def _dump(entry: ProviderEntry) -> dict[str, Any]:
    """One entry as the file holds it: pi's spelling, and only what was said.

    ``exclude_defaults`` because this is the shape the user reads and edits.
    Writing every field a fresh entry carries put ``"apiKey": ""`` and
    ``"models": []`` under a provider that signs in, which is the always-present
    empty block the previous shape was criticised for, one level down.
    """
    return entry.model_dump(by_alias=True, exclude_defaults=True)


def _commit(data: dict[str, Any], provider: str, entry: ProviderEntry, path: Path) -> None:
    """Write one entry, having checked the whole section still loads.

    The section, not just the entry: a provider pi does not ship needs an
    address and a protocol, and that is a rule about the key an entry is filed
    under, which only :class:`ProvidersConfig` can see. Validating the entry
    alone let a write succeed and the next ``load_config`` refuse the file --
    the one failure a single write path exists to make impossible.
    """
    providers = data.setdefault("providers", {})
    providers[provider] = _dump(entry)
    try:
        ProvidersConfig.model_validate(providers)
    except ValidationError as exc:
        providers.pop(provider, None)
        raise ValueError(_first_message(exc)) from exc
    write_json_atomic(path, data)


def _first_message(exc: ValidationError) -> str:
    """The first error's own sentence, without the input it was raised about.

    Pydantic's string form quotes the rejected value, and the rejected value
    here is a providers section holding the user's keys.
    """
    for error in exc.errors():
        message = str(error.get("msg") or "")
        return message.removeprefix("Value error, ")
    return "the providers section is not valid"


def _redact(value: Any) -> Any:
    """Redact a single value, list of values, or dict of values (per value,
    keys left visible -- see ``_redact_headers``)."""
    if value in (None, "", [], {}):
        return "(empty)"
    if isinstance(value, list):
        return ["****set****" for _ in value]
    if isinstance(value, dict):
        return _redact_headers(value)
    return "****set****"


def _redact_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Redact each header's value, keeping the key names visible.

    ``headers`` can carry a secret (an auth header some gateways need alongside
    the key) -- masking the whole dict as one ``****set****`` string would also
    hide which headers are configured, so each value is redacted on its own, the
    same rule every other secret field follows.
    """
    if headers is None:
        return None
    return {key: _redact(value) for key, value in headers.items()}


def provider_field_specs(provider: str = "") -> dict[str, dict[str, Any]]:
    """Reflect the provider entry schema into a flat ``path -> spec`` map.

    Each entry has keys: ``type``, ``default``, ``is_secret``, ``description``.
    Used by CLI parsers, the ``provider show`` command, and
    :func:`get_provider_config` to know which fields exist and which to redact.
    One schema for every provider now, so the name is accepted and ignored.
    """
    del provider
    return flatten_fields(ProviderEntry)


def unread_fields(provider: str) -> dict[str, str]:
    """Fields an entry may carry that this provider's route never reads.

    One schema serves every provider, so an entry offers fields that mean
    nothing for the way this particular one is reached. Reflecting the schema
    alone offered ``--api-key``, ``--base-url`` and ``--api`` for a provider that
    signs in -- which refuses the first, ignores the second by design so a
    configured address cannot redirect an account's token, and refuses the third
    because pi owns its protocol. For a built-in the last two are pi's to state,
    not ours.

    Keyed by field, valued by why.
    """
    entry = ProviderEntry()
    try:
        entry = _entry(read_raw_or_raise(get_config_path()), provider)
    except Exception:  # noqa: BLE001 - a report path; an unreadable config just has no entry
        pass
    from opendde_harness.providers.auth import CRED_OAUTH, credential_kind

    if credential_kind(provider, entry) == CRED_OAUTH:
        sign_in = f"ddeharness provider login {provider}"
        return {
            "api_key": f"this provider signs in -- run `{sign_in}`",
            "base_url": "this route's address is fixed by the grant",
            "api": "this route speaks its own protocol",
        }
    if pi_ids.is_builtin(provider):
        return {
            "base_url": f"pi carries {provider}'s address",
            "api": f"pi carries {provider}'s protocol",
        }
    return {}


# ---------------------------------------------------------------------------
# Public API: read
# ---------------------------------------------------------------------------


def list_providers(*, config_path: Path | None = None) -> list[dict[str, Any]]:
    """Every provider this config configures, with its current status.

    Returns one dict per entry:

    - ``name``               the pi provider id it is filed under
    - ``display_name``       the entry's own name, else pi's, else the id
    - ``is_oauth``           the entry names a sign-in
    - ``is_builtin``         pi ships this provider
    - ``configured``         :func:`providers.auth.credential_status` says yes
    - ``credential_state``   what pi's store holds for a sign-in ("oauth", or
                             "invalid" when the file is present and unreadable),
                             empty for every other kind
    - ``api_key_redacted``   ``****set****`` / ``(empty)`` / ``OAuth token`` / ...
    - ``base_url``           the declared address, or None
    """
    from opendde_harness.providers.auth import credential_status
    from opendde_harness.providers.pi_credentials import KIND_INVALID, stored_kind

    path = config_path or get_config_path()
    data = read_raw_or_raise(path)

    out: list[dict[str, Any]] = []
    for provider in configured_provider_names(data):
        entry = _entry(data, provider)
        is_oauth = entry.login == "oauth"
        status = credential_status(provider, entry, include_external=True)

        # "Signed out" and "the file is there and unreadable" are different
        # answers, and reporting the second as the first told someone whose
        # store was damaged that they had simply never signed in.
        credential_state = stored_kind(provider) if is_oauth else ""
        if is_oauth:
            if status.ok:
                api_key_redacted = "OAuth token"
            else:
                api_key_redacted = "(unreadable)" if credential_state == KIND_INVALID else "(empty)"
        elif entry.api_key:
            api_key_redacted = "****set****"
        elif status.ok and status.source == "environment":
            api_key_redacted = "(from the environment)"
        elif status.ok and status.source == "store":
            api_key_redacted = "(in the credential store)"
        elif status.ok:
            api_key_redacted = "(not needed)"
        else:
            api_key_redacted = "(empty)"

        out.append(
            {
                "name": provider,
                "display_name": pi_ids.display_name(provider, entry.name),
                "is_oauth": is_oauth,
                "is_builtin": pi_ids.is_builtin(provider),
                "configured": status.ok,
                "credential_state": credential_state,
                "api_key_redacted": api_key_redacted,
                "base_url": entry.base_url or None,
            }
        )
    return out


def get_provider_config(
    provider: str,
    *,
    redact_secrets: bool = True,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Return one provider's entry as a flat ``path -> value`` dict.

    Secret fields render as ``'****set****'`` / ``'(empty)'`` by default. Pass
    ``redact_secrets=False`` for plaintext.
    """
    _refuse_unwritable(provider)
    path = config_path or get_config_path()
    entry = _entry(read_raw_or_raise(path), provider)

    specs = provider_field_specs()
    flat = flatten_instance(entry)
    out: dict[str, Any] = {}
    for key, spec in specs.items():
        value = flat.get(key)
        if key == "models":
            # Rows are objects; a flat map is for display and for the CLI's
            # `--field value` surface, so name the models rather than dumping
            # their rows. The rows themselves are read through `model_rows`.
            value = entry.model_ids
        out[key] = _redact(value) if redact_secrets and spec["is_secret"] else value
    return out


def model_rows(provider: str, *, config_path: Path | None = None) -> list[dict[str, Any]]:
    """Every declared model of this provider, as the rows the config holds.

    A model written as a bare id comes back as a row of just its id, so callers
    reading facts about a model do not have to know which of the two ways it was
    written in.
    """
    path = config_path or get_config_path()
    entry = _entry(read_raw_or_raise(path), provider)
    return [row.model_dump(by_alias=True, exclude_none=True) for row in entry.rows()]


# ---------------------------------------------------------------------------
# Public API: write
# ---------------------------------------------------------------------------


def set_provider_fields(
    provider: str,
    fields: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Patch specific fields on a provider entry. Returns ``{path: previous_value}``.

    Creates the entry when there is none: writing a field for a provider is how
    one is configured, and there is no always-present empty block to patch.

    Raises:
        KeyError: a name no entry can be written for, or an unknown field path.
        RuntimeError: setting ``api_key`` on an entry that names a sign-in
            without turning the sign-in off in the same write.
        ValidationError: a value the entry schema rejects, including a declared
            provider left without ``baseUrl`` or ``api``.
    """
    _refuse_unwritable(provider)
    if not fields:
        return {}

    field_specs = provider_field_specs()
    unknown = [k for k in fields if k not in field_specs]
    if unknown:
        raise KeyError(f"Unknown field(s) {unknown} for provider '{provider}'. Available fields: {sorted(field_specs)}")

    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    current = _entry(data, provider)

    # Judged on the entry as it will stand: a key written together with
    # ``login: None`` is a key replacing a sign-in, which is how pi's own "Sign
    # in with an API key" lands on a provider that was signed in to.
    if fields.get("login", current.login) == "oauth" and fields.get("api_key"):
        raise RuntimeError(
            f"Provider '{provider}' signs in -- cannot set api_key directly. Run: ddeharness provider login {provider}"
        )

    working = current.model_dump()
    prev: dict[str, Any] = {}
    for key, raw in fields.items():
        leaf_cls, leaf_field = walk_nested_path(ProviderEntry, key)
        coerced = coerce_value(raw, leaf_cls.model_fields[leaf_field].annotation)
        if key == "models" and isinstance(coerced, list):
            # pi's ``models`` holds the ids the endpoint serves, with no
            # provider prefix: the prefix is the key this entry is filed under.
            # A caller holding a qualified id (the picker, the wizard) hands it
            # over as it has it, and this is the one place that strips it -- the
            # third write path is how two spellings of one model used to land in
            # one list.
            coerced = [_model_value(provider, m) for m in coerced]
        prev[key] = set_nested(key, coerced, working)

    _commit(data, provider, ProviderEntry.model_validate(working), path)
    return prev


def _model_value(provider: str, model: Any) -> Any:
    """One ``models`` element, with a prefix naming this provider stripped."""
    if isinstance(model, dict):
        row = dict(model)
        row["id"] = model_id.display(provider, str(row.get("id") or ""))
        return row
    return model_id.display(provider, str(model))


def serves_default_model(provider: str, *, config_path: Path | None = None) -> bool:
    """Is this provider the one that answers the configured default model?

    Removing its entry leaves the model id behind, so the config still names a
    provider that can no longer answer -- and the startup gate reads that as
    "not set up" and runs the wizard. Both doors that reset a provider ask this
    so they can say so before it happens.
    """
    from opendde_harness.config.loader import load_config

    try:
        config = load_config(config_path) if config_path else load_config()
        return model_id.provider_of(config.agents.defaults.model) == provider
    except Exception:  # noqa: BLE001 - a warning path; never the reason a reset fails
        return False


def reset_provider(
    provider: str,
    *,
    config_path: Path | None = None,
) -> None:
    """Sign a provider out when its entry named a sign-in, then remove the entry.

    Removal rather than a rewrite to schema defaults: an entry exists because
    somebody wrote it, and an entry emptied of its address and key is not a
    provider with nothing configured -- for a declared one it is not even a
    valid entry. So "reset" is "un-configure", which is what every caller meant.

    The sign-out comes first and is not caught: an entry removed while the grant
    stays stored is a sign-in the config has forgotten about and the store has
    not, and the caller would have been told it had signed out.

    For synchronous callers -- the CLI, the wizard -- which run their own loop
    for the sign-out. Inside a running loop (the TUI's RPC), await
    :func:`provider_logout` and then call :func:`remove_provider_entry`.
    """
    path = config_path or get_config_path()
    if _entry(read_raw_or_raise(path), provider).login == "oauth":
        from opendde_harness.providers.pi_service import run_then_shutdown

        run_then_shutdown(provider_logout(provider))
    remove_provider_entry(provider, config_path=path)


def remove_provider_entry(provider: str, *, config_path: Path | None = None) -> None:
    """Remove a provider's entry from the section. The write half of a reset."""
    _refuse_unwritable(provider)
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    _entry(data, provider)

    stored = data.get("providers")
    if isinstance(stored, dict):
        stored.pop(provider, None)
    write_json_atomic(path, data)
    logger.info("update_providers: {} removed from the providers section", provider)


async def provider_logout(provider: str) -> bool:
    """Forget this provider's stored sign-in. True when there was one.

    The delete runs inside the process's model service, which owns pi's
    credential store: it is the process that refreshes a rotated token, so it
    is the one whose writes this has to be serialised with. The process's own
    service, on the loop that owns it -- a second child on another loop would
    write the same file from its own snapshot, and ``get_service`` refuses to
    take over a service a running loop is streaming through.
    """
    from opendde_harness.config.loader import load_config
    from opendde_harness.providers.pi_service import get_service

    service = await get_service(load_config())
    reply = await service.logout(provider)
    return bool(reply.get("forgotten"))


def add_provider_model(
    provider: str,
    model: str,
    *,
    config_path: Path | None = None,
) -> list[str]:
    """Append ``model`` to a provider's ``models`` list (idempotent).

    Returns the new list of model ids. A prefix naming this provider is stripped
    first: pi's ``models`` holds the ids the endpoint serves.
    """
    _refuse_unwritable(provider)
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    entry = _entry(data, provider)

    wanted = model_id.display(provider, model)
    if wanted and wanted not in entry.model_ids:
        working = entry.model_dump()
        working["models"] = [*(working.get("models") or []), wanted]
        validated = ProviderEntry.model_validate(working)
        _commit(data, provider, validated, path)
        return validated.model_ids
    return entry.model_ids


def remove_provider_model(
    provider: str,
    model: str,
    *,
    config_path: Path | None = None,
) -> list[str]:
    """Remove ``model`` from a provider's ``models`` list (no-op if absent).

    Returns the new list of model ids. Accepts the id either way round -- bare
    as the list holds it, or qualified as the picker and the config's default
    spell it.
    """
    _refuse_unwritable(provider)
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    entry = _entry(data, provider)

    wanted = model_id.display(provider, model)
    if wanted in entry.model_ids:
        working = entry.model_dump()
        working["models"] = [m for m in entry.models if (m if isinstance(m, str) else m.id) != wanted]
        validated = ProviderEntry.model_validate(working)
        _commit(data, provider, validated, path)
        return validated.model_ids
    return entry.model_ids


def set_model_row(
    provider: str,
    model: str,
    fields: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Patch one model's row inside a provider's ``models`` list.

    A model listed as a bare id becomes a row; a model not listed at all is
    added as one. Returns the row as stored. Raises KeyError for an unknown
    provider or field, ValidationError for a value the row rejects.
    """
    _refuse_unwritable(provider)
    unknown = [k for k in fields if k not in ModelEntry.model_fields or k == "id"]
    if unknown:
        available = sorted(k for k in ModelEntry.model_fields if k != "id")
        raise KeyError(f"Unknown model field(s) {unknown}. Available: {available}")

    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    # An entry that no longer validates is left alone: falling back to the
    # defaults and writing them back erased the key, the address and the models
    # to change one model's row.
    entry = ProviderEntry.model_validate(_entry_dict(data, provider))

    wanted = model_id.display(provider, model)
    coerced = {k: coerce_value(v, ModelEntry.model_fields[k].annotation) for k, v in fields.items()}

    rows: list[Any] = []
    written: ModelEntry | None = None
    for element in entry.models:
        current = ModelEntry(id=element) if isinstance(element, str) else element
        if current.id != wanted:
            rows.append(element)
            continue
        written = ModelEntry.model_validate({**current.model_dump(), **coerced})
        rows.append(written)
    if written is None:
        written = ModelEntry.model_validate({"id": wanted, **coerced})
        rows.append(written)

    working = entry.model_dump()
    working["models"] = [r if isinstance(r, str) else r.model_dump() for r in rows]
    _commit(data, provider, ProviderEntry.model_validate(working), path)
    return written.model_dump(by_alias=True, exclude_none=True)


# ---------------------------------------------------------------------------
# Public API: credential health check
# ---------------------------------------------------------------------------


#: What the check asks the model for. Short on purpose: this is a reachability
#: question, and a long prompt only makes it a more expensive one.
_TEST_PROMPT = "Reply with the single word OK."

#: The ceiling the check names. A model with no declared ceiling is refused by
#: the service unless the request carries one, so this is not optional.
_TEST_MAX_TOKENS = 16


def test_provider(
    provider: str,
    *,
    timeout_s: int = 60,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Is this provider reachable, and does the credential work?

    Two questions, both asked of the model service, which is the thing that will
    be doing the sending:

    1. ``auth`` -- does pi resolve a credential for this provider at all, and
       from where. An entry the service does not even carry (no address, no
       models) answers here.
    2. one ``stream`` of a few tokens on the provider's own default model. What
       is reported is the first event: anything that is not an error proves the
       vendor accepted the credential and answered, and the stream is dropped
       there rather than run to the end.

    Returns a dict and never raises:

    - ``ok``            the model answered
    - ``status``        ``valid``, or why not (see the CLI's hint table)
    - ``elapsed_ms``    how long the two requests took
    - ``model``         the model that was asked
    - ``models_count`` / ``model_ids``  what the service serves for this provider
    - ``error``         one line of detail, or None
    """
    import time

    start = time.monotonic()

    try:
        _refuse_unwritable(provider)
    except (KeyError, RuntimeError) as exc:
        return _test_result(False, "unknown_provider", start, error=str(exc))

    try:
        from opendde_harness.config.loader import load_config

        config = load_config(config_path) if config_path else load_config()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return _test_result(False, "config_unreadable", start, error=str(exc))

    try:
        from opendde_harness.providers.pi_service import run_then_shutdown

        return run_then_shutdown(_check_through_the_service(provider, config, start, timeout_s))
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        from opendde_harness.providers.model_service import ModelServiceError

        if isinstance(exc, ModelServiceError):
            return _test_result(False, exc.code, start, error=str(exc))
        logger.debug("provider test: {} failed", provider, exc_info=True)
        return _test_result(False, "service_unavailable", start, error=str(exc))


def _test_result(
    ok: bool,
    status: str,
    start: float,
    *,
    error: str | None = None,
    model: str = "",
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    import time

    return {
        "ok": ok,
        "status": status,
        "elapsed_ms": int((time.monotonic() - start) * 1000),
        "model": model,
        "models_count": len(model_ids) if model_ids is not None else None,
        "model_ids": model_ids,
        "error": error,
    }


async def _check_through_the_service(provider: str, config: Any, start: float, timeout_s: int) -> dict[str, Any]:
    from opendde_harness.providers.pi_service import get_service

    service = await get_service(config)

    report = await service.auth()
    entry = next((p for p in report.get("providers") or () if p.get("id") == provider), None)
    if entry is None:
        # Not in the service's provider set: the entry supplies nothing it could
        # be declared from. `provider show` lists what it still needs.
        return _test_result(
            False,
            "not_served",
            start,
            error=f"{provider} is not configured well enough to be declared -- "
            f"run `ddeharness provider show {provider}`",
        )
    if not entry.get("configured"):
        return _test_result(False, "not_configured", start, error=entry.get("error") or "no credential resolved")

    served = [m for m in await service.models() if m.get("provider") == provider]
    model_ids = [str(m.get("id")) for m in served]
    model = _model_to_check(provider, config, model_ids)
    if not model:
        return _test_result(
            False,
            "no_model",
            start,
            error=f"{provider} serves no model -- add one with `ddeharness provider set {provider} --models <id>`",
            model_ids=model_ids,
        )

    context = {"messages": [{"role": "user", "content": _TEST_PROMPT, "timestamp": 0}]}
    options = {"maxTokens": _TEST_MAX_TOKENS}
    events = service.stream(provider, model, context, options)
    try:
        async with asyncio.timeout(timeout_s):
            async for event in events:
                kind = event.get("type")
                if kind == "start":
                    continue
                if kind == "error":
                    failure = event.get("error") or {}
                    return _test_result(
                        False,
                        str(event.get("code") or "error"),
                        start,
                        error=str(failure.get("errorMessage") or "the provider refused the request"),
                        model=model,
                        model_ids=model_ids,
                    )
                return _test_result(True, "valid", start, model=model, model_ids=model_ids)
    except TimeoutError:
        return _test_result(
            False,
            "timeout",
            start,
            error=f"no answer within {timeout_s}s",
            model=model,
            model_ids=model_ids,
        )
    finally:
        await events.aclose()
    return _test_result(
        False, "no_answer", start, error="the stream ended without an event", model=model, model_ids=model_ids
    )


def _model_to_check(provider: str, config: Any, served: list[str]) -> str:
    """Which of this provider's models the check asks: the configured default.

    The default model first, when this provider is the one serving it -- that is
    the model a turn would run -- then the curated shortlist's first entry, then
    whatever the service lists first. Matched on the bare id, because the service
    is asked for the id the endpoint serves.
    """
    from opendde_harness.providers.common_models import common_models_for

    if not served:
        return ""
    candidates: list[str] = []
    try:
        if model_id.provider_of(config.agents.defaults.model) == provider:
            candidates.append(str(config.agents.defaults.model))
    except Exception:  # noqa: BLE001 - a config that cannot answer just has no preference
        pass
    candidates.extend(common_models_for(provider))
    for candidate in candidates:
        bare = model_id.display(provider, candidate)
        if bare in served:
            return bare
    return served[0]


__all__ = [
    "add_provider_model",
    "configured_provider_names",
    "get_provider_config",
    "list_providers",
    "model_rows",
    "provider_field_specs",
    "provider_logout",
    "remove_provider_entry",
    "remove_provider_model",
    "reset_provider",
    "serves_default_model",
    "set_model_row",
    "set_provider_fields",
    "test_provider",
    "unread_fields",
]
