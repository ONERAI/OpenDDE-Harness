"""``model.*`` RPC handlers — backend for the TUI ``/model`` picker.

Ten methods drive the picker:

* ``model.options`` — the model this conversation is on, plus one row per
  provider: the ones this config configures, then the ones it could.
* ``model.save_key`` — write a provider's credential: a key for one of pi's
  own, or the address and the wire a provider this config declares is reached
  by.
* ``model.declare_provider`` — write the whole entry for an OpenAI-compatible
  endpoint this config does not hold yet: its id, its address and a key if it
  wants one; what it serves is read from its ``/models``.
* ``model.scope`` — pi's scoped models, read or saved.
* ``model.overlay`` — one field of one model's declared row.
* ``model.login`` / ``model.login_answer`` / ``model.login_cancel`` — pi's own
  OAuth flow, run inside the model service and shown step by step: every step it
  reports is pushed as a ``login.step`` notification carrying the login's id, and
  a step it is waiting on is answered by that id. The flow is pi's; nothing here
  parses a vendor's page or holds a token.

Two spellings meet here and the boundary is this module. The model service
reports the id the endpoint serves, bare; this project's config, and every
stored selection, spell a model ``"<provider>/<model>"`` because the prefix is
the only thing that names the provider serving it. So every id the picker
offers is qualified (``providers.model_id.join``) and every id shown inside one
provider's list is shortened again (``providers.model_id.display``), which is
why a row carries a label for each id rather than letting the client print the
id it was given.

What a provider is offered as serving comes from the model service, which is
the catalogue: one ``models`` request answers for every row, on the loop rather
than in a thread, because it is a line each way to a child this session already
holds. The curated shortlist in ``providers.common_models`` is what answers
when that request cannot be made at all.

All write helpers live in ``opendde_harness.config.update_providers`` (the
single write path for provider config); the handlers wrap the synchronous calls
in ``asyncio.to_thread`` so the event loop is not blocked on disk IO. An entry
that names a sign-in refuses a key from the picker — the sign-in is how it is
reached, and ``model.login`` is how one is run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from uuid import uuid4

from loguru import logger
from pydantic import ValidationError

from opendde_harness.config.update import set_scoped_models
from opendde_harness.config.update_providers import (
    provider_logout,
    remove_provider_entry,
    set_model_row,
    set_provider_fields,
)
from opendde_harness.providers import auth, model_id, pi_ids
from opendde_harness.providers.common_models import (
    common_models_for,
    provider_auth,
    refresh_models,
    rows_for_provider,
    service_rows,
)
from opendde_harness.tui_rpc.errors import (
    ConfigValidationError,
    NotSupportedInV01Error,
)
from opendde_harness.tui_rpc.models import (
    ModelDeclareProviderParams,
    ModelLoginAnswerParams,
    ModelLoginCancelParams,
    ModelLoginParams,
    ModelLogoutParams,
    ModelOptionsParams,
    ModelOverlayParams,
    ModelSaveKeyParams,
    ModelScopeParams,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from opendde_harness.config.schema import Config, ModelEntry, ProviderEntry, ProvidersConfig
    from opendde_harness.providers.model_service import ModelService
    from opendde_harness.tui_rpc.dispatcher import Dispatcher
    from opendde_harness.tui_rpc.methods.session import AgentLoopFactory

    SendFrame = Callable[[dict[str, Any]], Awaitable[None]]


def _parse(model_cls: type, params: dict) -> Any:
    try:
        return model_cls.model_validate(params)
    except ValidationError as exc:
        raise ConfigValidationError(
            f"invalid params for {model_cls.__name__}",
            data={"errors": exc.errors(include_url=False)},
        ) from exc


# ---------------------------------------------------------------------------
# Picker rows
# ---------------------------------------------------------------------------


def _offered_models(
    provider: str,
    entry: "ProviderEntry | None",
    *,
    configured: bool,
    served: list[dict[str, Any]] | None,
) -> list[str]:
    """The ids to offer for this provider: its own list, then what it serves.

    What the user declared comes first -- a self-hosted deployment is named
    nowhere else -- and then every row the model service reports for this
    provider, which for a declared endpoint is the list it publishes. The service is the catalogue: pi's
    own rows for a provider it carries, the entry's declared list for one this
    config declares, so the picker and the wizard cannot disagree about what a
    provider offers.

    Only for a *configured* provider. The service is handed every entry whether
    or not a credential resolves for it, so listing an unconfigured one would
    offer models whose every request fails on a missing key. Such a row shows
    what it configured and nothing else.

    ``served`` is ``None`` only when the service could not answer, and that is
    the one case the curated shortlist stands in for -- see
    ``providers.common_models``. A service that answered with no rows means this
    provider serves nothing, which is said rather than papered over.

    Every id comes back qualified. The entry stores its models bare, as pi's
    ``models`` holds them, and the service reports the id it was configured
    with; but the picker's ids go straight into ``config.set``, and a bare one
    names no provider.
    """
    curated = [model_id.join(provider, model) for model in (entry.model_ids if entry is not None else ())]
    if served is None:
        # The shortlist is written qualified already; join is idempotent on a
        # prefix that names this provider, so it is the one rule either way.
        offered = [model_id.join(provider, model) for model in common_models_for(provider)]
    elif configured:
        offered = [model_id.join(provider, str(row.get("id") or "")) for row in served]
    else:
        offered = []

    out: list[str] = []
    seen: set[str] = set()
    for candidate in (*curated, *offered):
        # A qualified id has exactly one spelling, so plain equality is the
        # whole identity check -- this used to need a per-provider merge key.
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _model_labels(
    provider: str,
    entry: "ProviderEntry | None",
    models: list[str],
    served: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """What to show for each offered id: a name, and a one-liner where there is one.

    Three sources, in this order: the row the user declared for the model, which
    is them describing their own deployment; the name on the model service's row
    for the same id; and last the id with this provider's prefix dropped, which
    is all anyone knows about a model nothing describes and is shorter than the
    qualified id the row is keyed by.

    Both other sources are keyed by the bare id -- the row as the entry stores
    it, the service's row as the endpoint serves it -- so each offered id is
    shortened before it is looked up.
    """
    names = {str(row.get("id") or ""): str(row.get("name") or "") for row in served or ()}
    out: dict[str, dict[str, Any]] = {}
    for model in models:
        bare = model_id.bare(model)
        row = entry.row(bare) if entry is not None else None
        item: dict[str, Any] = {
            "label": (row.name if row else "") or names.get(bare, "") or model_id.display(provider, model)
        }
        if row and row.description:
            item["description"] = row.description
        out[model] = item
    return out


#: pi's own auth method names, in this row's vocabulary. pi spells the key one
#: ``api_key``; every ``auth_type`` this project reports spells it ``key``, and
#: one spelling per field is the rule -- so the translation happens here, once.
_PI_METHODS = {"oauth": auth.CRED_OAUTH, "api_key": auth.CRED_KEY}


def _auth_methods(provider: str, declared: dict[str, Any] | None) -> list[str]:
    """Which ways in pi offers for this provider, in pi's own order.

    Read from the model service, which reads pi's provider objects: a provider
    with both a subscription and a key is the case a picker has to offer a
    choice for, and pi is the only thing that knows which those are.

    A provider pi does not carry -- one this config declares -- is reached by its
    address and a key, which is the one way in it has.
    """
    if declared is None:
        return [auth.CRED_KEY] if not pi_ids.is_builtin(provider) else []
    return [_PI_METHODS[method] for method in declared.get("methods") or () if method in _PI_METHODS]


def _auth_type(provider: str, entry: "ProviderEntry | None") -> str:
    """How this provider is reached, for a picker that has to choose one flow.

    ``providers.auth`` answers for an entry that exists. For one that does not --
    every row under "Add provider…" -- it answers "a key", which is right for
    the built-ins whose credential is a key and wrong for the ones that have no
    key to paste. pi reads no environment variable for a provider it only signs
    in to, so a provider that offers a sign-in and names no key variable is
    offered the sign-in; anything else is offered the key it does have.
    """
    kind = auth.credential_kind(provider, entry)
    if entry is None and kind == auth.CRED_KEY and provider in pi_ids.OAUTH and provider not in pi_ids.ENV_KEYS:
        return auth.CRED_OAUTH
    return kind


def _warning(provider: str, entry: "ProviderEntry | None", status: Any, kind: str) -> str:
    """What this provider still needs, for the row and the model list's subtitle.

    Only for an entry that exists and cannot be used. A provider with no entry
    at all is already marked "not connected", and repeating the whole credential
    sentence on every row of "Add provider…" would say nothing the header does
    not.
    """
    if entry is None or status.ok:
        return ""
    if kind == auth.CRED_OAUTH:
        # Not the CLI command any more: the picker runs the sign-in itself
        # (``model.login``), and naming a command to leave and type would be
        # advice against the key the user is about to press.
        return "not signed in yet — choose it to sign in"
    return status.summary


def _build_provider_entry(
    provider: str,
    entry: "ProviderEntry | None",
    *,
    current_provider: str | None,
    current_model: str | None = None,
    include_catalog: bool = True,
    served: list[dict[str, Any]] | None = None,
    declared_auth: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One picker row. ``served`` is every row the model service reported.

    The whole answer rather than this provider's share of it, because one
    ``models`` request answers for every row the picker builds. ``None`` means
    the service could not be asked, which is the only thing that puts the
    curated shortlist in front of a user.

    ``entry`` is this provider's ``providers`` entry, or ``None`` when the config
    has none -- which is what tells a configured provider from one that could be
    added, and is asked of ``providers.auth`` rather than derived here.
    """
    status = auth.credential_status(provider, entry, include_external=True)
    configured = status.ok
    kind = _auth_type(provider, entry)
    # Reported, never sent: pi resolves the environment itself. Its value is a
    # key, so only the variable's name travels, and only when one is set --
    # "this provider already has a key in the environment" is what the key form
    # has to say before it asks for one.
    key_env = auth.env_key_name(provider)

    rows = None if served is None else rows_for_provider(served, provider)
    models = _offered_models(provider, entry, configured=configured, served=rows)
    # pi's own description of how this provider is reached: which methods it
    # declares, and the sentence it shows for each. Absent when the service could
    # not be asked, and then the picker offers no choice rather than a guessed one.
    declared = (declared_auth or {}).get(provider)
    if provider == current_provider and current_model:
        # The model in use heads the list whatever the sources know of it: a
        # relay lists hundreds, and a picker that cannot show the current choice
        # reads as broken.
        models = [current_model, *[model for model in models if model != current_model]]
    return {
        # Names and one-liners for the ids above, so the picker shows what a
        # model is rather than the qualified id it is stored under.
        "model_labels": _model_labels(provider, entry, models, rows),
        "slug": provider,
        "name": pi_ids.display_name(provider, entry.name if entry is not None else ""),
        "authenticated": configured,
        "is_current": provider == current_provider,
        "auth_type": kind,
        "key_env": key_env or None,
        "models": models,
        "models_loaded": include_catalog,
        "total_models": len(models),
        # What the credential form must ask for, decided by the gate that will
        # judge the submission rather than by the form. A key already in the
        # environment is not one to ask for: the entry itself may hold nothing.
        # Nor is one for a provider whose credential is a whole environment
        # chain (:data:`pi_ids.AMBIENT`): the gate accepts an empty key there,
        # and a form that demanded one demanded a string nobody has.
        "needs_api_key": kind == auth.CRED_KEY and not key_env and provider not in pi_ids.AMBIENT,
        # A provider this config declares is reached by address, and an address
        # needs the protocol it serves -- both, or neither, and the schema
        # refuses one without the other.
        "needs_base_url": kind == auth.CRED_ENDPOINT,
        "base_url": (entry.base_url or None) if entry is not None else None,
        "api": (entry.api or None) if entry is not None else None,
        # Every way in pi offers, so a provider that has two can be offered the
        # choice between them -- pi's own step, with pi's own labels.
        "auth_methods": _auth_methods(provider, declared),
        "login_label": (declared or {}).get("loginLabel") or None,
        "key_label": (declared or {}).get("keyName") or None,
        "warning": _warning(provider, entry, status, kind),
    }


def _listed_providers(config: "Config | None") -> "list[tuple[str, ProviderEntry | None]]":
    """Every provider the picker lists: the configured ones, then the addable ones.

    The configured entries in the order the file declares them -- somebody
    reading their own config reads it in the order they wrote it -- and then
    pi's built-ins, which are the providers a picker can add with nothing but a
    credential.

    A provider this config *declares* is never in that second half. It needs an
    address and a wire, neither of which is a thing to guess at in a picker, so
    ``ddeharness provider set`` is where one is written; once written it is a
    configured entry like any other and appears in the first half. The one
    built-in that is reached only that way (:data:`pi_ids.DECLARED_BUILTINS`)
    is left out of the second half for the same reason: a key-only row under
    its name could not be finished.
    """
    entries = list(config.providers.items()) if config is not None else []
    known = {provider for provider, _ in entries}
    return [
        *entries,
        *[
            (provider, None)
            for provider in pi_ids.BUILTIN
            if provider not in known and provider not in pi_ids.DECLARED_BUILTINS
        ],
    ]


def _loaded_config() -> "Config | None":
    """The config, or ``None`` when it cannot be read.

    A config the picker cannot parse is not a reason to answer nothing: the rows
    still say which providers could be added and what each one needs, which is
    exactly what a user with a broken file is looking at the picker to fix. What
    such an answer cannot include is the configured half, because nothing can
    read it.
    """
    from opendde_harness.config.loader import load_config

    try:
        return load_config()
    except Exception:  # noqa: BLE001 - reported as "nothing configured", not raised at the picker
        return None


async def _service_rows(config: "Config | None") -> list[dict[str, Any]] | None:
    """Every model the service serves, or ``None`` when it could not answer.

    On the event loop, not in a thread: the service is a child process this
    session already holds and one ``models`` request is a line each way, which
    is faster than the hop that used to be here. That hop existed because the
    chain parsed a vendored catalogue on first read; the catalogue went with the
    Python routes.

    A failure here is reported as ``None`` rather than raised. No Node, no built
    bundle, a configuration the service refused -- none of them are a reason for
    the picker to be unopenable, and the curated shortlist is what the rows fall
    back to.
    """
    if config is None:
        return None
    try:
        return await service_rows(config)
    except Exception as exc:  # noqa: BLE001 - the picker has an offline answer
        logger.debug("model picker: the model service could not list models ({})", exc)
        return None


async def _declared_auth(config: "Config | None") -> dict[str, dict[str, Any]]:
    """pi's own auth description per provider, keyed by id, or empty.

    Empty is the honest answer when the service cannot be asked: a picker that
    does not know which methods a provider offers offers no choice, which is what
    it did before pi was asked at all.
    """
    if config is None:
        return {}
    try:
        return {str(row.get("id") or ""): row for row in await provider_auth(config)}
    except Exception as exc:  # noqa: BLE001 - the picker opens without pi's labels
        logger.debug("model picker: the model service could not describe provider auth ({})", exc)
        return {}


async def _entry(provider: str, current_provider: str | None, current_model: str | None = None) -> dict[str, Any]:
    """One picker row, for the handlers that return the provider they wrote to."""
    config = _loaded_config()
    served = await _service_rows(config)
    entry = config.providers.get(provider) if config is not None else None
    return _build_provider_entry(
        provider,
        entry,
        current_provider=current_provider,
        current_model=current_model,
        served=served,
        declared_auth=await _declared_auth(config),
    )


async def _entries(
    current_provider: str | None,
    current_model: str | None = None,
    *,
    slug: str | None = None,
    include_catalog: bool = True,
) -> list[dict[str, Any]]:
    """Every picker row, from one config read and one ``models`` request.

    All three are hoisted out of the per-row build -- the config, the models the
    service serves, and pi's own description of how each provider signs in:
    asking per row would be forty-odd requests for one answer, and re-reading the
    config per row would read the same file as many times.

    ``slug`` is matched exactly. A pi provider id has one spelling, so there is
    nothing to canonicalize and a name that matches nothing lists nothing.
    """
    config = _loaded_config()
    served = await _service_rows(config)
    declared_auth = await _declared_auth(config)
    return [
        _build_provider_entry(
            provider,
            entry,
            current_provider=current_provider,
            current_model=current_model,
            include_catalog=include_catalog,
            served=served,
            declared_auth=declared_auth,
        )
        for provider, entry in _listed_providers(config)
        if slug is None or provider == slug
    ]


def _current_selection() -> tuple[str, str | None]:
    """The configured default model, and the provider its id names.

    One read, because there is one fact: the id carries its provider, so there
    is no second field to reconcile it with and no spelling to canonicalize.
    """
    from opendde_harness.cli._helpers import load_runtime_config

    model = load_runtime_config(None, None).agents.defaults.model
    return model, model_id.provider_of(model) or None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def model_options(params: dict, agent_loop_factory: "AgentLoopFactory | None" = None) -> dict:
    """Which models exist, and which one *this conversation* is on.

    The session matters: the model is per conversation now, so answering from
    ``agents.defaults`` would star the wrong row for every session that has
    switched -- the picker would disagree with the status bar it sits under.
    """
    parsed = _parse(ModelOptionsParams, params)
    current_model, current_provider = await _selection_for(agent_loop_factory, parsed.session_id)
    refresh_errors: dict[str, str] = {}
    if parsed.refresh:
        refresh_errors = await _refresh_models()
    entries = await _entries(current_provider, current_model, slug=parsed.slug, include_catalog=parsed.include_catalog)
    config = _loaded_config()
    return {
        "model": current_model,
        "provider": current_provider or "",
        "default_model": str(config.agents.defaults.model or "") if config is not None else "",
        "providers": entries,
        "refresh_errors": refresh_errors,
    }


async def _refresh_models(providers: list[str] | None = None, *, force: bool = False) -> dict[str, str]:
    """Ask the declared endpoints what they serve, now. Failures by provider.

    A service that cannot be asked at all is reported under ``*`` rather than
    raised: the picker still opens on the last known list, the way pi's does
    ("showing cached models").
    """
    config = _loaded_config()
    if config is None:
        return {}
    try:
        return await refresh_models(config, providers, force=force)
    except Exception as exc:  # noqa: BLE001 - the picker has the last list to show
        logger.debug("model picker: the model service could not refresh catalogs ({})", exc)
        return {"*": str(exc)}


async def model_scope(params: dict) -> dict:
    """pi's scoped models, as ``agents.scopedModels`` holds them.

    ``write`` saves the list (``None`` for all, pi's "all enabled"); without it
    the saved list is read back. The session-only scope pi keeps between saves
    lives in the client, which is where pi keeps it too.
    """
    parsed = _parse(ModelScopeParams, params)
    if parsed.write:
        try:
            await asyncio.to_thread(set_scoped_models, parsed.models)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ConfigValidationError(str(exc)) from exc
        return {"models": parsed.models}
    config = _loaded_config()
    scoped = config.agents.scoped_models if config is not None else None
    return {"models": list(scoped) if scoped is not None else None}


#: Which input a refused submission names, keyed by the kind of credential the
#: gate was judging. ``providers.auth`` reports the requirement in the user's
#: words; the form needs the field to put the cursor in.
_MISSING_FIELD = {auth.KIND_NONE: "base_url", auth.KIND_DEVICE_FLOW: "login"}


async def model_save_key(params: dict) -> dict:
    """Write a provider's credential: a key, or the address and wire of a declared one.

    The submission is judged by ``providers.auth`` -- the same gate routing and
    startup ask -- rather than by a rule of this handler's own, which is how the
    picker came to refuse a submission the gate would have accepted. What the
    schema alone can judge (an address without its protocol, an ``api`` the
    service does not implement) is judged by the write path, and its sentence is
    passed through as it stands.
    """
    parsed = _parse(ModelSaveKeyParams, params)
    config = _loaded_config()
    entry = config.providers.get(parsed.slug) if config is not None else None
    label = pi_ids.display_name(parsed.slug, entry.name if entry is not None else "")

    # A name that is a near-miss for a pi provider id is not one, and writing it
    # would declare a provider nobody serves. Said here because the picker can
    # be asked about a name it never listed.
    refusal = auth.key_refusal(parsed.slug)
    if refusal:
        raise ConfigValidationError(refusal, data={"slug": parsed.slug, "field": "slug"})

    # The entry as it would stand after this write, so the gate judges the
    # result rather than the difference: a key submitted for an entry that
    # already declares its address is complete, and the address is not resent.
    # A key is one way in and a sign-in is the other, so submitting a key turns
    # the sign-in off -- pi's "Sign in with an API key" chosen for a provider
    # that was signed in to means "use this key now", and pi's own store keeps
    # one credential per provider.
    submitted = {
        "login": None,
        "api_key": parsed.api_key,
        "base_url": parsed.base_url or (entry.base_url if entry is not None else ""),
        "api": parsed.api or (entry.api if entry is not None else None),
    }
    status = auth.credential_status(parsed.slug, submitted)
    if not status.ok:
        raise ConfigValidationError(
            status.summary or f"{label} is missing credentials",
            data={"slug": parsed.slug, "field": _MISSING_FIELD.get(status.kind, "api_key")},
        )

    # The key is written even when it is empty, and said rather than omitted:
    # leaving the field alone kept whatever was there, so an entry that once
    # held a key would still be sending it to an endpoint reached without one.
    fields: dict[str, Any] = {"login": None, "api_key": parsed.api_key}
    if parsed.base_url:
        fields["base_url"] = parsed.base_url
    if parsed.api:
        fields["api"] = parsed.api

    try:
        await asyncio.to_thread(set_provider_fields, parsed.slug, fields)
    except RuntimeError as exc:
        raise NotSupportedInV01Error(str(exc), data={"slug": parsed.slug}) from exc
    except KeyError as exc:
        raise ConfigValidationError(str(exc), data={"slug": parsed.slug}) from exc
    except ValueError as exc:
        # Every schema refusal lands here: ``ValidationError`` is a
        # ``ValueError``, and the section check raises the schema's own sentence
        # as one. Both name the field to fix, so neither is rewritten.
        raise ConfigValidationError(str(exc), data={"slug": parsed.slug}) from exc

    _, current_provider = _current_selection()
    return {
        "provider": await _entry(parsed.slug, current_provider),
    }


#: What this door declares a provider as speaking. Chat Completions is what a
#: relay or a self-hosted server implements, and it is the whole reason the form
#: asks for four fields rather than five: a provider on another wire is written
#: by ``ddeharness provider set``, which takes every field the entry has.
_DECLARED_API = pi_ids.DEFAULT_API


def _dialable(base_url: str) -> bool:
    """An address the model service can dial: http(s), and a host to reach.

    Judged here rather than by the entry schema, which takes any string: the
    schema's rule is that an address names its wire, and this is the narrower
    question of whether *this* form's address is one at all. A scheme it does
    not recognise is refused rather than prefixed, because guessing between
    http and https is guessing at a credential's transport.
    """
    parsed = urlsplit(base_url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


async def model_declare_provider(params: dict) -> dict:
    """Declare an OpenAI-compatible endpoint: id, address, key, first model.

    The picker's one door to what ``ddeharness provider set`` writes, and the
    whole entry in one submission because a half-written one is refused by the
    section check -- an address needs its wire, and a declared provider with no
    model serves nothing.

    Three refusals are this door's own rather than the write path's, because
    each is a question about the door. A pi provider id is refused: pi carries
    that provider's address and its catalogue, the provider list already offers
    it a key, and a declaration filed under its id would replace both with four
    typed fields. An address has to be dialable. A model has to be named,
    because pi has no catalogue for a provider it does not ship.

    Every refusal here travels as a detail and names no field. The frame carries
    one of the two -- a ``data`` of its own replaces the sentence the dispatcher
    would have wrapped -- and a form of four fields has to show the sentence,
    while the field name is something no client reads.

    The key is written and never read back: the returned row is the picker's
    own, which carries no credential.
    """
    parsed = _parse(ModelDeclareProviderParams, params)
    provider = parsed.provider.strip()
    base_url = parsed.base_url.strip()
    models = [part.strip() for part in parsed.model.split(",") if part.strip()]

    removed = pi_ids.removed_message(provider)
    if removed:
        raise ConfigValidationError(removed)
    if pi_ids.is_builtin(provider):
        raise ConfigValidationError(
            f"pi ships {pi_ids.display_name(provider)} itself, with its own address and catalogue: pick "
            f"{provider} from the provider list and give it a credential instead of declaring it here."
        )
    # A near-miss for a pi id is not a name to declare a provider under: the
    # gate says which id was meant, and it is one of the rows above this form.
    refusal = auth.key_refusal(provider)
    if refusal:
        raise ConfigValidationError(refusal)
    if not _dialable(base_url):
        raise ConfigValidationError(
            "the base URL has to be an http:// or https:// address: it is dialled as it is written and never "
            "guessed at."
        )
    # One write, so the entry never exists in the half state the schema refuses.
    # The key goes in even when it is empty, as ``model.save_key`` sends it: a
    # redeclared endpoint reached without one must stop sending the old key.
    fields: dict[str, Any] = {
        "base_url": base_url,
        "api": _DECLARED_API,
        "api_key": parsed.api_key,
        "models": models,
    }
    try:
        await asyncio.to_thread(set_provider_fields, provider, fields)
    except RuntimeError as exc:
        raise NotSupportedInV01Error(str(exc)) from exc
    except KeyError as exc:
        # A name no entry can be filed under -- a slash, a space, a provider
        # this project removed. The write path's sentence already says which.
        raise ConfigValidationError(str(exc)) from exc
    except ValueError as exc:
        # Every schema refusal, including the section check's own sentence.
        raise ConfigValidationError(str(exc)) from exc

    # The endpoint is asked what it serves as soon as it is declared -- the way
    # pi treats OpenRouter, the address and the key are what gets typed and the
    # list is published. Asking reconfigures a running service with the new
    # entry first. An endpoint that publishes nothing and was given no ids is
    # refused rather than declared empty: it would offer nothing to pick.
    errors = await _refresh_models([provider], force=True)
    _, current_provider = _current_selection()
    row = await _entry(provider, current_provider)
    discovered = max(0, len(row["models"]) - len(models))
    if not row["models"]:
        why = errors.get(provider) or errors.get("*") or "it publishes no models at GET /models"
        try:
            await asyncio.to_thread(remove_provider_entry, provider)
        except KeyError:
            pass
        raise ConfigValidationError(
            f"{provider} lists no models ({why}). Name the ids it serves, comma-separated, and declare it again."
        )
    return {"provider": row, "discovered": discovered}


async def _forget_sign_in(slug: str) -> bool:
    """Delete the stored grant, on this loop, through the process's service.

    Not on a worker thread: the service belongs to the loop it was started on,
    and a second loop cannot use it or end it (``pi_service.get_service``).
    A sign-out the service could not do is an error, not a false "done".
    """
    from opendde_harness.providers.model_service import ModelServiceError

    try:
        return await provider_logout(slug)
    except (ModelServiceError, OSError, RuntimeError) as exc:
        raise ConfigValidationError(
            f"could not sign {pi_ids.display_name(slug)} out: {exc}", data={"slug": slug}
        ) from exc


async def model_logout(params: dict) -> dict:
    """pi's ``/logout``: forget the credential and keep the provider.

    A stored sign-in is deleted from the store and the entry stops naming it; a
    key is blanked. What the entry declares -- an address, a wire, its models --
    stays, the way pi's own logout removes what its auth store holds and touches
    nothing in ``models.json``. ``forgotten`` is False when there was nothing of
    either kind to forget: a provider reached through an environment variable is
    not signed out by this, and the row says which variable.
    """
    parsed = _parse(ModelLogoutParams, params)
    config = _loaded_config()
    entry = config.providers.get(parsed.slug) if config is not None else None
    if entry is None:
        return {"forgotten": False}

    if auth.configured_login(entry) == "oauth":
        forgotten = await _forget_sign_in(parsed.slug)
        fields: dict[str, Any] = {"login": None, "api_key": ""}
    elif entry.api_key:
        forgotten = True
        fields = {"api_key": ""}
    else:
        return {"forgotten": False}

    try:
        await asyncio.to_thread(set_provider_fields, parsed.slug, fields)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ConfigValidationError(str(exc), data={"slug": parsed.slug}) from exc
    return {"forgotten": forgotten}


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------


@dataclass
class _Login:
    """One sign-in in flight, and how to reach the flow running it."""

    provider: str
    service: "ModelService"
    #: The model service request the flow is running as: what an answer and a
    #: cancellation are addressed to. None until the service has taken the
    #: request, which is a moment a cancel can arrive in.
    request_id: int | None = None
    #: A cancel arrived before there was a request to abort.
    cancelled: bool = False


#: The sign-ins running right now, by the id their pushed steps carry. The id is
#: the client's when it sent one -- minted before the first step can arrive, so
#: the client can tell its own flow's steps from another's -- and minted here
#: otherwise. Never the service's own request id: that travels to a client and
#: back, and a service that restarted would count from one again.
_LOGINS: dict[str, _Login] = {}


async def _abort_login(login: _Login) -> None:
    """End the flow at the service, if the service has one to end yet."""
    if login.request_id is not None:
        await login.service.abort(login.request_id)


async def _login_service() -> "ModelService":
    """The process's model service, configured from this config.

    The seam the login handlers are tested against. Configured is the part that
    matters: it is ``configure`` that hands the service the credential store a
    grant has to land in, so a login run against an unconfigured service would
    succeed and store nothing.
    """
    from opendde_harness.providers.pi_service import get_service

    config = _loaded_config()
    if config is None:
        raise ConfigValidationError(
            "the config cannot be read, so there is nowhere to record a sign-in: fix config.json first"
        )
    return await get_service(config)


async def _push_login_step(send_frame: "SendFrame | None", login_id: str, provider: str, event: dict) -> None:
    """One step of pi's flow, pushed as it arrives.

    pi's own object travels: the device code and its URL, the sign-in link, the
    login-method menu with its options. Converting it here would mean a second
    vocabulary to keep in step with pi's, and the client shows what pi wrote.
    """
    if send_frame is None:
        return
    step = event.get("ask") if isinstance(event.get("ask"), dict) else event.get("notify")
    if not isinstance(step, dict):
        return
    await send_frame(
        {
            "jsonrpc": "2.0",
            "method": "login.step",
            "params": {"login_id": login_id, "provider": provider, "step": step},
        }
    )


async def model_login(params: dict, *, send_frame: "SendFrame | None" = None) -> dict:
    """Run one provider's sign-in, and answer when a credential is stored.

    The flow is pi's, run inside the model service (``ModelService.login_with_id``)
    so the grant lands in the store the service itself refreshes from. This
    handler is the two things around it: every step pi reports is pushed to the
    client as it arrives, and what the client sends back is routed to the flow
    that is waiting (``model.login_answer``).

    The request is answered once the flow has finished, which is minutes for a
    device code. Nothing else is held up by that: the server dispatches each
    frame as its own task.

    Success means a credential was stored, not that a model answered -- asking a
    model for one token to prove a sign-in conflates holding a credential with
    being entitled to that model, and bills the user to find out. The entry is
    then written the way the wizard writes it: ``login: "oauth"`` is the whole of
    what the config says about a sign-in, and it is what makes the picker read
    the store rather than look for a key.
    """
    from opendde_harness.providers.model_service import ModelServiceError

    parsed = _parse(ModelLoginParams, params)
    provider = parsed.provider.strip()
    if provider not in pi_ids.OAUTH:
        signs_in = ", ".join(sorted(pi_ids.OAUTH))
        raise ConfigValidationError(
            f"{provider} is not signed in to. The ones that are: {signs_in}. Everything else takes a key.",
            data={"slug": provider, "field": "slug"},
        )

    login_id = parsed.login_id or uuid4().hex
    if login_id in _LOGINS:
        raise ConfigValidationError(f"a sign-in with id {login_id} is already running", data={"slug": provider})

    # One sign-in per provider: a second one for the same provider is the first
    # one's client having gone (Esc before a step arrived, a closed picker), and
    # two flows storing into the same slot would race for it.
    for stale_id, stale in list(_LOGINS.items()):
        if stale.provider == provider:
            _LOGINS.pop(stale_id, None)
            await _abort_login(stale)

    service = await _login_service()
    login = _LOGINS[login_id] = _Login(provider=provider, service=service)
    result: dict[str, Any] = {}
    try:
        # No mode: pi's login-method menu travels as a step like any other,
        # which is the whole point of running the flow here rather than
        # pre-answering it.
        login.request_id, steps = await service.login_with_id(provider)
        if login.cancelled:
            # The cancel came while the service was taking the request. Ended
            # the way a later cancel ends it: the flow fails, and so does this.
            await _abort_login(login)
        async for step in steps:
            if step.get("type") == "login_prompt":
                await _push_login_step(send_frame, login_id, provider, step)
            else:
                result = step
    except ModelServiceError as exc:
        # The code only: pi's or the vendor's sentence can quote the exchange
        # that failed, up to the tokens in a response it could not parse, and a
        # log line is not the place for those.
        logger.debug("model.login: {} failed ({})", provider, exc.code)
        raise ConfigValidationError(
            f"the sign-in did not finish ({exc.code}). Try again, or run `ddeharness provider login {provider}`.",
            data={"slug": provider},
        ) from exc
    except asyncio.CancelledError:
        # The request's own task was cancelled -- the client is gone -- so the
        # flow it was holding is ended rather than left polling a vendor.
        await _abort_login(login)
        raise
    finally:
        _LOGINS.pop(login_id, None)

    if result.get("type") != "oauth":
        raise ConfigValidationError(
            f"the sign-in did not produce a grant for {pi_ids.display_name(provider)}",
            data={"slug": provider},
        )

    try:
        await asyncio.to_thread(set_provider_fields, provider, {"login": "oauth", "api_key": ""})
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ConfigValidationError(str(exc), data={"slug": provider}) from exc

    _, current_provider = _current_selection()
    # Built fresh, so the row is read after the write: the entry now names a
    # sign-in, the store holds the grant, and reading it reconfigures a service
    # that is already running.
    return {"login_id": login_id, "provider": await _entry(provider, current_provider)}


async def model_login_answer(params: dict) -> dict:
    """Hand a waiting sign-in what it asked for: an option id, or a pasted code."""
    parsed = _parse(ModelLoginAnswerParams, params)
    login = _LOGINS.get(parsed.login_id)
    if login is None:
        return {"answered": False}
    reply = await login.service.login_answer(login.request_id, parsed.answer)
    return {"answered": bool(reply.get("answered"))}


async def model_login_cancel(params: dict) -> dict:
    """Abort a sign-in nobody is waiting for any more.

    The service's own abort, which is what ends pi's flow: a device-code poll
    stops, a callback server closes, and the pending ``model.login`` fails rather
    than sitting on a question with nobody left to answer it.
    """
    parsed = _parse(ModelLoginCancelParams, params)
    login = _LOGINS.get(parsed.login_id)
    if login is None:
        return {"cancelled": False}
    login.cancelled = True
    await _abort_login(login)
    return {"cancelled": True}


#: The fields of a model's row a session may set, and how a value typed at the
#: prompt is read for each. These are ``ModelEntry``'s own fields: what the user
#: knows about their own model beyond its id, declared per model because one
#: number or one thinking level cannot be right for every model a session
#: switches between. ``id`` is not among them -- that is which row this is.
_ROW_FIELDS: dict[str, str] = {
    "api": "api",
    "catalog_model": "text",
    "context_window": "tokens",
    "description": "text",
    "max_tokens": "tokens",
    "name": "text",
    "reasoning": "flag",
    "reasoning_effort": "level",
    "temperature": "number",
}

#: pi's thinking levels, as ``config.schema.ReasoningEffort`` spells them.
_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

#: The two fields that size a request, so a change has to reach the builders
#: that sized themselves against the window outside a turn.
_WINDOW_FIELDS = ("context_window", "max_tokens")

#: What clears a field. Not the empty string: an empty value is a plausible
#: thing to type for a name, and "unset this" has to be unambiguous.
_CLEARED = "default"


def _row_value(field: str, raw: str) -> Any:
    """One typed value, read for the field it is being written to.

    Everything arrives as a string -- the prompt has nothing else to give -- and
    is turned into what the row declares here rather than in the write path, so
    the refusal can name the vocabulary the field accepts.
    """
    kind = _ROW_FIELDS[field]
    value = raw.strip()
    cleared = value.lower() == _CLEARED
    if kind == "text":
        # ``name`` and ``description`` are plain strings whose unset value is
        # the empty one; ``catalog_model`` is absent when it names nothing.
        if cleared:
            return "" if field in ("name", "description") else None
        return value
    if cleared:
        return None
    if kind == "level":
        if value.lower() not in _THINKING_LEVELS:
            raise ConfigValidationError(
                f"unknown thinking level {raw!r}; use one of {', '.join(_THINKING_LEVELS)} or {_CLEARED}",
                data={"field": "value", "got": raw},
            )
        return value.lower()
    if kind == "api":
        if value not in pi_ids.APIS:
            raise ConfigValidationError(
                f"unknown api {raw!r}; use one of {', '.join(pi_ids.APIS)} or {_CLEARED}",
                data={"field": "value", "got": raw},
            )
        return value
    if kind == "flag":
        lowered = value.lower()
        if lowered in ("on", "true", "yes", "1"):
            return True
        if lowered in ("off", "false", "no", "0"):
            return False
        raise ConfigValidationError(
            f"{field} takes on or off, or {_CLEARED} to leave it to the catalogue",
            data={"field": "value", "got": raw},
        )
    if kind == "number":
        try:
            number = float(value)
        except ValueError:
            number = -1.0
        if number < 0:
            raise ConfigValidationError(
                f"{field} takes a number of zero or more, or {_CLEARED}",
                data={"field": "value", "got": raw},
            )
        return number
    # tokens: a count, with the "128k" shorthand people write windows in.
    digits = value.lower().replace(",", "").replace("_", "")
    scale = 1000 if digits.endswith("k") else 1
    digits = digits.rstrip("k")
    if not digits.isdigit() or int(digits) <= 0:
        raise ConfigValidationError(
            f"{field} takes a positive token count (e.g. 128000 or 128k) or {_CLEARED}",
            data={"field": "value", "got": raw},
        )
    return int(digits) * scale


def _live_providers_maps(loop: Any) -> list[Any]:
    """Every providers map a turn could read a row from, once each.

    Rows are read live, off the map the provider was built with, so a row
    written to disk has to be written into the objects already in memory too.
    There is more than one: the loop's settings hold the map the boot provider
    was built from, and a session that switched runs on a provider built from
    its own deep copy of the config -- the session declaring the row is on that
    copy, not on the default's.

    Read out of ``__dict__`` rather than with ``getattr``, because a decorator
    wrapping a provider forwards unknown attributes to the one it wraps, and
    that would report the inner map as the wrapper's own. Deduplicated by
    identity, since the common case is one object shared by all of them.
    """
    maps: list[Any] = []
    seen: set[int] = set()

    def remember(candidate: Any) -> None:
        if candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            maps.append(candidate)

    remember(getattr(getattr(loop, "settings", None), "providers", None))
    stack = list(loop.live_providers())
    visited: set[int] = set()
    while stack:
        provider = stack.pop()
        if provider is None or id(provider) in visited:
            continue
        visited.add(id(provider))
        remember(provider.__dict__.get("providers") if hasattr(provider, "__dict__") else None)
        stack.append(getattr(provider, "_provider", None))
        stack.extend(getattr(provider, "_inners", None) or [])
    return maps


def _write_row_live(providers: "ProvidersConfig | None", provider: str, row: "ModelEntry") -> None:
    """Put one model's row into a live providers map, in place.

    Nothing is added for a provider the map has no entry for: a model in use is
    served by an entry, because routing refuses an id whose provider is not
    configured, so an absent entry means this map does not describe that
    provider at all.
    """
    entry = providers.get(provider) if providers is not None else None
    if entry is None:
        return
    for index, element in enumerate(entry.models):
        if (element if isinstance(element, str) else element.id) == row.id:
            entry.models[index] = row
            return
    entry.models.append(row)


async def model_overlay(params: dict, agent_loop_factory: "AgentLoopFactory | None" = None) -> dict:
    """Set one field of the current model's declared row, on disk and in the loop.

    The method is still called ``model.overlay`` because the TUI calls it that;
    what it writes is a row in ``providers.<id>.models``, which is where a model's
    declared facts live. Per model, as pi keeps its thinking levels: what one
    model needs (a level, a window a relay caps, an output ceiling) must not
    become a setting for every model the session switches to. ``default`` clears
    the field.
    """
    from opendde_harness.config.schema import ModelEntry

    parsed = _parse(ModelOverlayParams, params)
    field = parsed.field.strip()
    if field not in _ROW_FIELDS:
        raise ConfigValidationError(
            f"unknown model field {parsed.field!r}; use one of {', '.join(sorted(_ROW_FIELDS))}",
            data={"field": "field", "got": parsed.field},
        )
    value = _row_value(field, parsed.value)

    # The session's model when it switched: a row describes one model, and the
    # model this conversation is on is the one the user is describing.
    model, provider = await _selection_for(agent_loop_factory, parsed.session_id)
    if not model or not provider:
        raise ConfigValidationError("no current model to describe", data={"model": model or ""})
    try:
        stored = await asyncio.to_thread(set_model_row, provider, model, {field: value})
    except KeyError as exc:
        raise ConfigValidationError(str(exc), data={"model": model}) from exc
    except ValueError as exc:
        # ``ValidationError`` is a ``ValueError``, and so is the section check's
        # own refusal; both already name the field and the vocabulary.
        raise ConfigValidationError(str(exc), data={"model": model}) from exc

    result: dict[str, Any] = {"model": model, "field": field, "value": value}
    loop = agent_loop_factory() if agent_loop_factory is not None else None
    if loop is not None:
        row = ModelEntry.model_validate(stored)
        for providers in _live_providers_maps(loop):
            _write_row_live(providers, provider, row)
        if field in _WINDOW_FIELDS:
            # Budgets follow the new figure from the next turn on: turns re-walk
            # the ladder at entry, the out-of-turn fallback here.
            await asyncio.to_thread(loop.refresh_context_window)
            result["context_window"] = loop.resolve_window(model).tokens
    return result


def _session_selection(
    agent_loop_factory: "AgentLoopFactory | None", session_id: str | None
) -> tuple[str, str | None] | None:
    """This session's own (model, provider), or None when it never switched."""
    if not agent_loop_factory or not session_id:
        return None
    try:
        loop = agent_loop_factory()
    except Exception:
        return None
    # ``session_model`` falls back to the default, so it never answers None --
    # asking it alone would report a switch for every session that never made
    # one.
    if loop is None or not loop.has_session_binding(session_id):
        return None
    # One binding, both halves: read separately, a switch landing between the
    # two reads pairs one model with another binding's provider.
    binding = loop.binding_for_session(session_id)
    return binding.model, binding.provider_name


async def _selection_for(
    agent_loop_factory: "AgentLoopFactory | None", session_id: str | None
) -> tuple[str, str | None]:
    """The (model, provider) this conversation is on.

    The configured default unless the session switched. On the event loop, like
    every other read of the loop's bindings: a first ask for a session that
    switched before a restart reads its record and builds the provider, and a
    worker thread doing that raced the turn and the switch it can interleave
    with (a restore installed after a newer choice overwrote it).

    The provider is the one the binding was built on, which is the prefix of its
    own model id; a binding that recorded none answers from the id, which every
    switch writes qualified.
    """
    model, provider = _current_selection()
    own = _session_selection(agent_loop_factory, session_id)
    if own is None:
        return model, provider
    session_model, route = own
    return session_model, (route or model_id.provider_of(session_model) or provider)


def register_model_methods(
    dispatcher: "Dispatcher",
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
    send_frame: "SendFrame | None" = None,
) -> None:
    """Register the ``model.*`` handlers on a dispatcher instance.

    ``send_frame`` gates the sign-in group the way the brokers are gated: the
    steps of a login are pushed, so a gateway with no notification sink would
    register a method that shows a user nothing and waits for an answer to a
    question they never saw.
    """

    async def _options(params: dict) -> dict:
        return await model_options(params, agent_loop_factory)

    async def _overlay(params: dict) -> dict:
        return await model_overlay(params, agent_loop_factory)

    async def _login(params: dict) -> dict:
        return await model_login(params, send_frame=send_frame)

    dispatcher.register("model.options", _options)
    dispatcher.register("model.save_key", model_save_key)
    dispatcher.register("model.declare_provider", model_declare_provider)
    dispatcher.register("model.logout", model_logout)
    dispatcher.register("model.scope", model_scope)
    dispatcher.register("model.overlay", _overlay)
    if send_frame is not None:
        dispatcher.register("model.login", _login)
        dispatcher.register("model.login_answer", model_login_answer)
        dispatcher.register("model.login_cancel", model_login_cancel)


__all__ = [
    "model_options",
    "model_save_key",
    "model_declare_provider",
    "model_logout",
    "model_scope",
    "model_overlay",
    "model_login",
    "model_login_answer",
    "model_login_cancel",
    "register_model_methods",
    "_build_provider_entry",
]
