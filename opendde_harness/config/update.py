"""Minimal in-place updates for ~/.opendde_harness/config.json.

Unlike ``save_config`` which re-serializes the entire Pydantic model (and
would bake every runtime default back into the file), these helpers read
the raw JSON, patch a small set of fields, and atomically rewrite via
temp-file + rename. Used by the onboarding wizard and the ``skill`` /
``provider`` commands so a change persists across restarts without
touching unrelated fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from opendde_harness.config._fields import write_json_atomic
from opendde_harness.config.loader import get_config_path, read_raw_or_raise

# Default long-term memory server endpoint, seeded into a fresh config's
# plugins.config["long-term-memory"]. Kept in sync with
# opendde_harness.plugin.memory.longterm._server.DEFAULT_MEMORY_BASE_URL.
_DEFAULT_MEMORY_BASE_URL = "http://localhost:18791"


def set_skill_blocked(
    name: str,
    blocked: bool,
    *,
    config_path: Path | None = None,
) -> list[str]:
    """Add/remove a skill name on ``skillForge.blocklist``; returns the new
    list. Matching is case-insensitive; adding an already-listed name or
    removing an absent one is a no-op (the file is still not rewritten).

    The blocklist is read at process start (AgentLoop / context engine
    construction), so a change takes effect on the next agent start, not
    on a running process.
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    section = data.setdefault("skillForge", {})
    current = [str(x) for x in (section.get("blocklist") or [])]
    lowered = {x.casefold() for x in current}
    if blocked:
        if name.casefold() in lowered:
            return current
        current.append(name)
    else:
        if name.casefold() not in lowered:
            return current
        current = [x for x in current if x.casefold() != name.casefold()]
    section["blocklist"] = current
    write_json_atomic(path, data)
    logger.info(
        "config/update: skillForge.blocklist now {!r} ({} {!r})", current, "blocked" if blocked else "unblocked", name
    )
    return current


def set_language(
    language: str,
    *,
    config_path: Path | None = None,
) -> str | None:
    """Patch the top-level ``language`` on the on-disk config. Returns previous value.

    Set by the onboarding wizard's language screen. Read by the CLI/wizard copy
    (via ``_t``) and injected into the agent's system prompt so replies use the
    chosen language.
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    prev = data.get("language")
    data["language"] = language
    write_json_atomic(path, data)
    logger.info("config/update: language set to {!r} (was {!r})", language, prev)
    return prev


def set_web_search_key(key: str, *, config_path: Path | None = None) -> str | None:
    """Patch ``tools.web.braveApiKey`` on the on-disk config. Returns the previous value.

    Set by the onboarding wizard's web-search screen; an empty key removes
    it, and ``web_search`` then queries DuckDuckGo without one.
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    web = data.setdefault("tools", {}).setdefault("web", {})
    # The schema reads either spelling; one field is written, both are cleared.
    previous = [web.pop(name) for name in ("braveApiKey", "brave_api_key") if name in web]
    prev = previous[0] if previous else None
    if key:
        web["braveApiKey"] = key
    write_json_atomic(path, data)
    logger.info("config/update: web search key {}", "set" if key else "cleared")
    return prev


def set_default_model(
    model: str,
    *,
    provider: str | None = None,
    config_path: Path | None = None,
) -> str | None:
    """Patch ``agents.defaults.model`` on the on-disk config. Returns previous value.

    Used by the onboarding wizard after the user picks a provider: the wizard
    needs to swap the default model to one that matches the chosen provider
    (otherwise the TUI would still route to whatever the freshly created
    ``Config()`` baked in, which is typically a different vendor).

    ``provider`` qualifies a bare model id, and is how a caller that knows the
    provider hands it over. The id is what carries it: the field that used to
    name the provider separately is gone, because it overrode what an id said
    and a stale one routed the new model to the old vendor's key.
    """
    from opendde_harness.providers import model_id

    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    defaults = data.setdefault("agents", {}).setdefault("defaults", {})
    prev = defaults.get("model")
    qualified = model_id.join(provider, model) if provider else model
    defaults["model"] = qualified
    defaults.pop("provider", None)
    write_json_atomic(path, data)
    logger.info("config/update: default model set to {} (was {})", qualified, prev)
    return prev


def init_extension_block_defaults(*, config_path: Path | None = None) -> None:
    """Seed the user-facing subset of the memory / plugins / skillForge
    extension blocks into a fresh ``~/.opendde_harness/config.json``.

    Called once by the onboarding bootstrap so a new config shows these knobs
    at their schema defaults — discoverable and editable without reading the
    source. Each field is only written when absent (``setdefault``), so this is
    idempotent and never clobbers a value the user (or an earlier wizard step)
    already set. ``memory.backend`` is seeded to its schema default
    (the bundled long-term backend); a fresh install with no memory models configured degrades
    gracefully (empty recall + a warning, never a crash), and the wizard's
    Step 4 / skip-guard resolve it back to ``None`` when memory is opted out or
    left unconfigured.

    Defaults are pulled from the Pydantic models so this seed can't drift from
    the schema, with two deliberate onboard-time overrides:
      - ``plugins.config["long-term-memory"]`` is seeded with only ``base_url`` so
        the block is never empty and the user can see/edit it. Identity
        (``user_id`` / ``agent_id``) is deliberately NOT duplicated here — it
        comes from ``memory.userId`` / ``memory.agentId`` via the host's
        ``ServiceLocator`` at plugin activation time.

    Key casing follows each block's convention: ``memory`` / ``skillForge`` use
    camelCase (the file-level alias); ``plugins.config`` is a verbatim
    pass-through dict whose keys stay snake_case (each plugin owns its schema).
    """
    from opendde_harness.config.features import MemoryConfig, PluginsConfig, SkillForgeRouterConfig

    path = config_path or get_config_path()
    data = read_raw_or_raise(path)

    mem = MemoryConfig()
    memory = data.setdefault("memory", {})
    memory.setdefault("backend", mem.backend)
    memory.setdefault("userId", mem.user_id)
    memory.setdefault("agentId", mem.agent_id)
    memory.setdefault("memoryTopK", mem.memory_top_k)

    plugins = data.setdefault("plugins", {})
    plugins.setdefault("disabled", list(PluginsConfig().disabled))
    # snake_case keys: plugins.config is handed to the plugin factory verbatim.
    # Identity is not seeded here — it comes from ServiceLocator, sourced from
    # memory.userId / memory.agentId at plugin activation, not duplicated.
    plugins.setdefault("config", {}).setdefault(
        "long-term-memory",
        {"base_url": _DEFAULT_MEMORY_BASE_URL},
    )

    router_defaults = SkillForgeRouterConfig()
    skill_forge = data.setdefault("skillForge", {})
    skill_forge.setdefault("enabled", True)
    router = skill_forge.setdefault("router", {})
    router.setdefault("enabled", router_defaults.enabled)
    router.setdefault("weights", dict(router_defaults.weights))

    write_json_atomic(path, data)
    logger.info("config/update: seeded memory/plugins/skillForge extension defaults")


def set_plugin_config_fields(
    plugin_id: str,
    fields: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> None:
    """Merge ``fields`` into ``plugins.config[plugin_id]`` on the on-disk config.

    A merge rather than a replace: the slice holds several independent decisions
    (a memory service's address and the port it is meant to listen on) written at
    different moments, and a replacing write would drop whichever the caller did
    not happen to be carrying.
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    slice_ = data.setdefault("plugins", {}).setdefault("config", {}).setdefault(plugin_id, {})
    slice_.update(fields)
    write_json_atomic(path, data)
    logger.info(
        "config/update: plugins.config.{} updated ({})",
        plugin_id,
        ", ".join(fields),
    )


def set_memory_backend(
    backend: str | None,
    *,
    config_path: Path | None = None,
) -> str | None:
    """Patch ``memory.backend`` on the on-disk config. Returns previous value.

    ``"longterm"`` enables the bundled long-term memory backend; ``None``
    disables backend-driven memory entirely -- there is no second backend to
    fall back to, so recall and storage simply stop happening. The onboarding
    wizard's memory step writes the model sections to the memory root's config
    toml and flips this flag here.
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    section = data.setdefault("memory", {})
    prev = section.get("backend")
    section["backend"] = backend
    write_json_atomic(path, data)
    logger.info("config/update: memory.backend set to {!r} (was {!r})", backend, prev)
    return prev


def set_scoped_models(models: list[str] | None, *, config_path: Path | None = None) -> None:
    """Write ``agents.scopedModels``: pi's saved model scope, or ``None`` for all."""
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    section = data.setdefault("agents", {})
    if models is None:
        section.pop("scopedModels", None)
    else:
        section["scopedModels"] = list(models)
    write_json_atomic(path, data)
    logger.info("config/update: agents.scopedModels set to {}", "all" if models is None else len(models))


__all__ = [
    "set_default_model",
    "set_memory_backend",
    "set_scoped_models",
    "set_skill_blocked",
    "init_extension_block_defaults",
]
