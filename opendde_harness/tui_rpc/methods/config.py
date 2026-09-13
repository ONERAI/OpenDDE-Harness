"""``config.get`` / ``config.set`` RPC handlers (specs §3.6).

Contract source: ``docs/openspec/changes/tui-ipc-bridge/specs/tui-ipc.md §3.6``.

The surface is the ``tui.*`` preference fields, derived from
the schema section itself; any other write target raises
:class:`ConfigFieldReadonlyError` (-32010). Values are stored
in ``~/.opendde_harness/config.json`` using dotted-path nesting (``tui.theme`` →
``{"tui": {"theme": "..."}}``) so that the same file is loadable by the legacy
``opendde_harness.config.loader`` without any schema gymnastics.

Validation
----------

Per-key validators reject:

* ``tui.theme``: must be a non-empty string matching ``[A-Za-z0-9_-]+``.
* ``tui.show_token_usage``: must be a boolean.

Anything else → :class:`ConfigValidationError` (-32011).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import ValidationError
from pydantic.alias_generators import to_camel

from opendde_harness.cli._helpers import load_runtime_config, make_provider
from opendde_harness.config.schema import TuiConfig
from opendde_harness.providers import model_id
from opendde_harness.providers.auth import MissingCredentialsError
from opendde_harness.tui_rpc.errors import (
    ConfigFieldReadonlyError,
    ConfigValidationError,
    ModelNotAvailableError,
)

if TYPE_CHECKING:
    from opendde_harness.tui_rpc.dispatcher import Dispatcher
    from opendde_harness.tui_rpc.methods.session import AgentLoopFactory


_CONFIG_DIR_NAME = ".opendde_harness"
_CONFIG_FILENAME = "config.json"

# The persisted section defines the writable fields, defaults and validation.
_HOT_FIELDS = {
    f"{section}.{name}": (schema, name) for section, schema in (("tui", TuiConfig),) for name in schema.model_fields
}
_DEFAULTS = {key: schema.model_fields[name].default for key, (schema, name) in _HOT_FIELDS.items()}
CONFIG_WRITABLE_KEYS: tuple[str, ...] = tuple(_HOT_FIELDS)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    return Path.home() / _CONFIG_DIR_NAME / _CONFIG_FILENAME


def _load_config() -> dict[str, Any]:
    """Load ``config.json`` for a read-modify-write (get/set/_set_model).

    Absent / empty -> ``{}`` (safe to create fresh). A present-but-unparseable
    file raises ConfigValidationError rather than the old empty-dict fallback:
    returning ``{}`` here and then ``_save_config`` would overwrite the user's
    whole config with just the changed key (data loss). The on-disk file is the
    source of truth; downstream loaders read the same file independently.
    """
    from opendde_harness.config.loader import ConfigReadError, read_raw_or_raise

    try:
        return read_raw_or_raise(_config_path())
    except ConfigReadError as exc:
        raise ConfigValidationError(str(exc)) from exc


def _save_config(payload: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _get_nested(payload: dict[str, Any], dotted_key: str) -> Any | None:
    """Return the value at the dotted path, or None if absent."""
    parts = dotted_key.split(".")
    cur: Any = payload
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(to_camel(part), cur.get(part))
    return cur


def _set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: dict[str, Any] = payload
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    # save_config emits camelCase. Replace that spelling instead of leaving two
    # versions of a field, which the strict schema rejects as an extra input.
    cur.pop(to_camel(parts[-1]), None)
    cur[parts[-1]] = value


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def config_get(params: dict) -> dict:
    """Return values for whitelisted keys.

    Spec §3.6: unknown keys are silently omitted (NOT an error).
    """
    requested_raw = params.get("keys") if isinstance(params, dict) else None
    if requested_raw is None:
        requested: list[str] = list(CONFIG_WRITABLE_KEYS)
    else:
        if not isinstance(requested_raw, list) or not all(isinstance(k, str) for k in requested_raw):
            raise ConfigValidationError(
                "config.get params.keys must be a list[str] if provided",
                data={"field": "keys", "got": repr(requested_raw)},
            )
        requested = requested_raw

    payload = _load_config()
    out: dict[str, Any] = {}
    for key in requested:
        if key not in _HOT_FIELDS:
            # Unknown / non-whitelisted key — silently omit per spec.
            continue
        value = _get_nested(payload, key)
        out[key] = value if value is not None else _DEFAULTS[key]
    return {"config": out}


async def config_set(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """Write a single whitelisted key. Returns ``{applied, previous}``.

    The special key ``"model"`` switches the model this session runs on, or
    the default new ones start on (returns ``{applied, previous, value, scope,
    ...}``); see :func:`_set_model`.

    Raises:
        ConfigValidationError (-32011): params shape or value invalid.
        ConfigFieldReadonlyError (-32010): key not in writable whitelist.
    """
    if not isinstance(params, dict):
        raise ConfigValidationError(
            "config.set params must be an object",
            data={"got": type(params).__name__},
        )

    key = params.get("key")
    if not isinstance(key, str) or not key:
        raise ConfigValidationError(
            "config.set params.key is required and must be a non-empty string",
            data={"field": "key", "got": repr(key)},
        )
    if "value" not in params:
        raise ConfigValidationError(
            "config.set params.value is required",
            data={"field": "value"},
        )
    raw_value = params["value"]

    if key == "model":
        return _set_model(params, raw_value, agent_loop_factory)

    if key not in _HOT_FIELDS:
        raise ConfigFieldReadonlyError(
            f"key '{key}' is not in the v0.1 hot-changeable whitelist",
            data={"field": key, "writable": list(CONFIG_WRITABLE_KEYS)},
        )

    schema, field = _HOT_FIELDS[key]
    try:
        validated = getattr(schema.model_validate({field: raw_value}), field)
    except ValidationError as exc:
        raise ConfigValidationError(
            f"{key}: {exc.errors()[0]['msg']}",
            data={"field": key, "got": repr(raw_value)},
        ) from exc

    payload = _load_config()
    previous = _get_nested(payload, key)
    _set_nested(payload, key, validated)
    _save_config(payload)

    return {"applied": True, "previous": previous}


def _set_model(
    params: dict,
    raw_value: Any,
    agent_loop_factory: "AgentLoopFactory | None",
) -> dict:
    """Switch the model this session runs on, or the default new ones start on.

    Two scopes, because they answer different questions. With a
    ``session_id`` (what the picker sends) the switch is scoped to that
    session: no other session moves, and ``agents.defaults`` is left alone so
    a new session still starts on the configured default. Pass
    ``scope="default"``, or omit ``session_id``, to change that default
    instead; sessions that already switched keep their own model.

    Either way the provider is built before anything is persisted or applied,
    so a rebuild failure aborts with the on-disk model untouched.

    A switch during a turn is not refused. The running turn holds the binding
    it started on for its whole tree, so the new model takes effect on the
    session's next turn -- which is what a user asking mid-answer means.
    """
    if not isinstance(raw_value, str) or not raw_value:
        raise ConfigValidationError(
            "config.set model value must be a non-empty string",
            data={"field": "value", "got": repr(raw_value)},
        )
    # The id names its provider or it names nobody. There is no provider field
    # beside it any more and nothing derives one from the spelling: a bare
    # `/model gpt-5.5` is refused here rather than sent to whichever vendor a
    # table guessed, which is how one vendor's model reached another vendor's
    # key. The sentence is the schema's own, so the refusal the TUI shows and
    # the one `ddeharness` prints for the same input are the same sentence.
    if not model_id.provider_of(raw_value):
        from opendde_harness.config.schema import Config

        # ``model_construct`` because this is a refusal path: it skips every
        # validator, so an environment that itself names a bare model cannot
        # turn "qualify that id" into an internal error, and the method reads
        # nothing off the instance when it is given the id to explain.
        raise ConfigValidationError(
            Config.model_construct().explain_unrouted(raw_value),
            data={"field": "value", "got": raw_value},
        )

    session_id = params.get("session_id")
    scope = params.get("scope")
    if scope not in (None, "session", "default"):
        raise ConfigValidationError(
            "config.set model scope must be 'session' or 'default'",
            data={"field": "scope", "got": repr(scope)},
        )
    has_session = isinstance(session_id, str) and bool(session_id)
    if scope == "session" and not has_session:
        # Never widen a scope the caller narrowed: falling through to the
        # default branch here would write agents.defaults and move every
        # session that never switched. The TUI sends a session_id that is null
        # until the first session.create resolves, so this is reachable.
        raise ConfigValidationError(
            "config.set model scope 'session' needs a session_id",
            data={"field": "session_id", "got": repr(session_id)},
        )
    session_scoped = scope != "default" and has_session

    loop = agent_loop_factory() if agent_loop_factory is not None else None
    binding = None
    if loop is not None:
        try:
            binding = _build_binding(loop, raw_value)
        except MissingCredentialsError as exc:
            # Carried through as the sentence the user needs. `typer.Exit`
            # subclasses RuntimeError, so this used to land in the branch below
            # and `str(exc)` was the exit code -- the picker said
            # `cannot build provider ... error: "1"`.
            raise ModelNotAvailableError(
                exc.summary,
                data={"model": raw_value, "provider": exc.provider, "remedy": exc.remedy},
            ) from exc
        except (SystemExit, RuntimeError, ValueError) as exc:
            raise ModelNotAvailableError(
                f"cannot build provider for model {raw_value!r}",
                data={"model": raw_value, "error": str(exc)},
            ) from exc

    if session_scoped:
        if loop is None:
            # Nothing was built, so nothing was validated -- do not report a
            # switch that did not happen.
            return {"applied": False, "previous": None, "value": raw_value, "scope": "session"}
        previous = loop.session_model(session_id)
        loop.set_session_binding(session_id, binding)
        _remember_session_model(loop, session_id, raw_value)
        return {
            "applied": True,
            "previous": previous,
            "value": raw_value,
            "scope": "session",
            "session_id": session_id,
            "applies_to_session": True,
        }

    # A default-scoped switch still moves the asking conversation when that
    # conversation never chose a model of its own, because it reads the
    # default. Answered here rather than inferred from the scope: the client
    # cannot see which sessions have their own binding.
    follows_default = None
    if loop is not None and has_session:
        follows_default = not loop.has_session_binding(session_id)

    payload = _load_config()
    previous = _get_nested(payload, "agents.defaults.model")
    # The model id is the whole switch. `agents.defaults.provider` used to be
    # written beside it and is gone from the schema: two fields naming one
    # provider disagreed, and the field won over the id.
    _set_nested(payload, "agents.defaults.model", raw_value)
    _save_config(payload)

    if loop is not None:
        # Not a two-attribute assignment: the subagent manager and the context
        # engine each hold a fallback for work that runs outside a turn, and
        # this is what re-points them.
        loop.set_default_binding(binding)

    result = {"applied": True, "previous": previous, "value": raw_value, "scope": "default"}
    if follows_default is not None:
        # Absent rather than null when there is no session to answer for: the
        # wire contract types it as a boolean.
        result["applies_to_session"] = follows_default
    return result


def _remember_session_model(loop: Any, session_key: str, model: str) -> None:
    """Persist the choice on the session, so a restart does not undo it.

    Stored on the session record rather than in ``agents.defaults``: it is
    this conversation's model, and a new conversation must still start on the
    configured default.

    The model id is all that is stored. The provider used to be written beside
    it, for a restore to rebuild the binding with; the id names it now, and the
    restore reads the provider from the id it already has.

    Written in memory unconditionally, saved only for a session that already
    has a file. ``session.create`` is lazy -- it mints a key and writes
    nothing until the session's first real save -- so saving here would
    manufacture a record with zero messages for anyone who runs ``/model``
    before saying anything. The choice still reaches disk: it rides the
    session's first real save. ``session.title`` guards the same case the
    same way.
    """
    sessions = loop.sessions
    try:
        session = sessions.get_or_create(session_key)
        session.metadata["model"] = model
        if sessions.exists(session_key):
            sessions.save(session)
    except Exception:
        logger.warning("could not persist the model on session {!r}", session_key, exc_info=True)


def _build_binding(loop: Any, model: str) -> Any:
    """One provider per (provider, model), reused across sessions and switches.

    Building one resolves a credential and starts a model service, so a session
    flipping between two models must not pay twice. The pool is the loop's;
    without one (a one-shot wiring, a test) build directly from the prospective
    config.

    Only the model id goes in. It is qualified by the time it gets here, and the
    pool routes on its prefix, so there is nothing else to hand over -- naming
    the provider a second time is what let a caller redirect a model to a
    credential that does not serve it.
    """
    from opendde_harness.providers.binding import ModelBinding

    pool = loop.provider_pool
    if pool is not None:
        return pool.bind(model)
    runtime = load_runtime_config(None, None)
    runtime.agents.defaults.model = model
    return ModelBinding(make_provider(runtime), model)


def register_config_methods(
    dispatcher: "Dispatcher",
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> None:
    """Register ``config.get`` / ``config.set`` on a dispatcher instance."""

    async def _set(params: dict) -> dict:
        return await config_set(params, agent_loop_factory=agent_loop_factory)

    dispatcher.register("config.get", config_get)
    dispatcher.register("config.set", _set)


__all__ = [
    "config_get",
    "config_set",
    "register_config_methods",
    "CONFIG_WRITABLE_KEYS",
]
