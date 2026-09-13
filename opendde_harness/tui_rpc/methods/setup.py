"""Configuration status reported to the TUI (``setup.status``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from opendde_harness.plugin.active import active_registry

if TYPE_CHECKING:
    from opendde_harness.tui_rpc.dispatcher import Dispatcher


def _config_path() -> Path:
    from opendde_harness.config.loader import get_config_path

    return get_config_path()


def _detect_provider_configured(payload: dict) -> bool:
    """Return True iff the loaded config payload indicates a usable provider.

    The onboarding gate's criterion ("required config complete"): at least one
    provider is usable AND ``agents.defaults.model`` is set. Either alone can't
    drive a turn, so the UI must still park on the setup panel.

    "Usable" is ``providers.auth``'s one question, asked of every entry --
    including the sign-in ones, whose grant it reads from the model service's
    credential store. This used to carry two extra branches for the MiniMax
    plans, which were reached by a sign-in of this project's own; they are
    reached by an API key now, so the ordinary walk answers for them.
    """
    if not isinstance(payload, dict):
        return False

    agents = payload.get("agents")
    defaults = agents.get("defaults") if isinstance(agents, dict) else None
    defaults = defaults if isinstance(defaults, dict) else {}

    model = defaults.get("model")
    if not (isinstance(model, str) and model):
        return False

    # A model id names its provider, and that is the only place a provider is
    # named: `agents.defaults.provider` is gone from the schema, so there is no
    # second field to wave the gate through on. Whether the provider the id
    # names holds anything is asked below, of every entry.
    providers = payload.get("providers")
    if isinstance(providers, dict):
        # `providers.auth`, like every other gate. Reading `apiKey` off the raw
        # payload made this the seventh rule and it disagreed with the other six
        # -- on the exact configuration this module's rewrite was filed to fix:
        # Azure with a key and no address was waved through into a chat that
        # then could not run.
        from opendde_harness.config.schema import ProvidersConfig
        from opendde_harness.providers.auth import credential_status

        try:
            entries = ProvidersConfig.model_validate(providers)
        except Exception:
            # A section the schema refuses configures nothing. Reporting it as
            # "no provider" parks the UI on the setup panel, which is where
            # somebody with an unloadable providers section has to start.
            return False
        for provider, entry in entries.items():
            if credential_status(provider, entry, include_external=True).ok:
                return True

    return False


async def setup_status(params: dict) -> dict:
    """Report missing or invalid configuration without claiming readiness."""
    path = _config_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("setup.status: configuration missing at {}", path)
        return {"provider_configured": False, "error": "Configuration is missing. Run ddeharness onboard."}
    except OSError as exc:
        logger.warning("setup.status: read failed for {}: {}", path, exc)
        return {"provider_configured": False, "error": "Configuration cannot be read. Run ddeharness doctor."}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("setup.status: invalid JSON in {}: {}", path, exc)
        return {
            "provider_configured": False,
            "error": "Configuration contains invalid JSON. Repair it before onboarding.",
        }

    if not isinstance(payload, dict):
        return {"provider_configured": False, "error": "Configuration must be a JSON object."}
    plugins = payload.get("plugins") or {}
    configs = plugins.get("config") or {} if isinstance(plugins, dict) else {}
    if not isinstance(configs, dict):
        configs = {}
    registry = active_registry()
    compute_configured = True
    for plugin_id, ready in registry.readiness_checks():
        plugin_config = configs.get(plugin_id) or {}
        if not isinstance(plugin_config, dict):
            name = registry.manifest_for(plugin_id).display_name or plugin_id
            return {"provider_configured": False, "error": f"{name} configuration must be a JSON object."}
        compute_configured = compute_configured and ready(plugin_config)
    return {
        "provider_configured": _detect_provider_configured(payload),
        "compute_configured": compute_configured,
    }


def register_setup_methods(dispatcher: "Dispatcher") -> None:
    dispatcher.register("setup.status", setup_status)


__all__ = ["setup_status", "register_setup_methods"]
