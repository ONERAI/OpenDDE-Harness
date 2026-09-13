"""Configuration loading utilities."""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from opendde_harness.config.schema import FEATURE_FIELDS, Config

# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None

# Parsed configs keyed by path, with the file's (mtime_ns, size, inode) at
# parse time. The TUI RPC server loads the config on every request;
# re-validating an unchanged file each time is wasted work, and a changed file
# is caught by the stat check -- the inode because every writer replaces the
# file atomically, and the kernel's timestamp granularity would otherwise let a
# same-size rewrite within one tick read as unchanged. Entries are handed out
# as deep copies so a caller that edits its copy (``load_runtime_config``
# overrides the workspace) cannot leak the edit.
_cache: dict[str, tuple[tuple[int, int, int], Config]] = {}

# Paths already warned about as malformed in this process; repeated
# load_config calls (status/doctor load more than once) warn only once.
_warned_paths: set[str] = set()


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path() -> Path:
    """Get the configuration file path."""
    if _current_config_path:
        return _current_config_path
    return Path.home() / ".opendde_harness" / "config.json"


class ConfigReadError(Exception):
    """An existing config file could not be parsed. Callers doing a
    read-modify-write MUST NOT proceed: overwriting would replace the user's
    whole config with just their section (data loss). Only a genuinely-absent
    file is safe to create fresh.

    Deliberately NOT a RuntimeError: the CLI write commands wrap their ops in a
    broad ``except RuntimeError`` (for provider OAuth-refusal etc.), and we want
    a parse error to bypass those and reach the single ``run()`` handler (or a
    caller's explicit ``except ConfigReadError``), not be swept up implicitly."""


class ConfigSchemaError(ValueError):
    """A config file parsed as JSON but does not match the schema.

    Every model is ``extra='forbid'``, so a key this release does not know --
    a typo, or one a retired release wrote -- lands here rather than being
    silently ignored. Nothing rewrites an old config in place: the remedy is
    ``ddeharness onboard``, which writes a fresh one.
    """


#: Keys a release before this one wrote into every config by default and this
#: one no longer defines. Read and dropped, never rewritten: the file stays the
#: user's, and the key stops mattering the day it stops being written. Not a
#: migration facility -- a typo or a hand-written retired key is still an error.
#: ``agent`` held two ``agent.*`` RPC preferences (``thinkingBudget``,
#: ``temperature``) that nothing ever read into a request; every release wrote
#: the section, so it is dropped whatever it holds.
_RETIRED_KEYS = (("tools", "web", "search"), ("agent",))


def _drop_retired_keys(data: dict[str, Any]) -> None:
    for *parents, leaf in _RETIRED_KEYS:
        node: Any = data
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict) and leaf in node:
            del node[leaf]
            logging.getLogger(__name__).debug("config: dropped retired key %s", ".".join((*parents, leaf)))


#: The one sentence for a ``providers`` section written in the shape this
#: release replaced. Every key below belonged to the per-vendor sections the
#: providers block used to declare; the block is now pi's ``models.json`` shape,
#: keyed by pi's own provider ids, so there is no key-by-key move to describe --
#: the whole section is written differently and the wizard is what writes it.
_PROVIDERS_RESHAPED = (
    "The providers section has changed shape: it is now pi's models.json shape, keyed by pi's own "
    "provider ids ({ids}, ...) with pi's own field names -- apiKey, baseUrl, api, headers, models. "
    "Your config still carries {found}, which belonged to the per-vendor sections of the previous "
    "shape. Run `ddeharness onboard` to write a fresh config; nothing is migrated."
)

#: Keys that only ever appeared in the previous providers shape, so finding one
#: is proof the section was written against it. ``wire`` is now a provider's
#: ``api``; ``extraHeaders`` is ``headers``; ``modelOverlay`` is rows inside
#: ``models``; ``apiBase`` is ``baseUrl``.
_OLD_PROVIDER_KEYS = ("apiBase", "api_base", "extraHeaders", "extra_headers", "modelOverlay", "model_overlay", "wire")

#: Section names the previous shape declared that are not pi provider ids. A
#: config holding one was written against the old shape even if it carries none
#: of the old field names.
_OLD_PROVIDER_SECTIONS = (
    "aihubmix",
    "azure_openai",
    "azureOpenai",
    "custom",
    "dashscope",
    "gemini",
    "hosted_vllm",
    "hostedVllm",
    "minimax_cn",
    "minimaxCn",
    "minimax_global",
    "minimaxGlobal",
    "moonshot",
    "ollama",
    "ollama_chat",
    "ollamaChat",
    "openai_codex",
    "openaiCodex",
    "siliconflow",
    "volcengine",
    "vllm",
    "zhipu",
)


def _refuse_old_providers_shape(path: Path, data: dict[str, Any]) -> None:
    """Refuse a providers section written in the shape this release replaced.

    One rule and one sentence, rather than a per-key move: the section is keyed
    differently now (pi provider ids, not our own section names) and its fields
    are pi's, so nothing in it maps across on its own. Said here rather than by
    a validator, because a validator's error quotes the value it rejected, and
    the value is a providers block holding the user's keys.
    """
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return
    found: list[str] = []
    for name, section in providers.items():
        entry = section if isinstance(section, dict) else {}
        # A name from the old shape is proof only while the entry is not one
        # of this shape's own: a declared provider carries ``baseUrl``, which
        # the old sections never did (theirs was ``apiBase``), and the wizard
        # itself declares an endpoint under ``custom``.
        declared = "baseUrl" in entry or "base_url" in entry
        if name in _OLD_PROVIDER_SECTIONS and not declared:
            found.append(f"providers.{name}")
        if not isinstance(section, dict):
            continue
        found.extend(f"providers.{name}.{key}" for key in _OLD_PROVIDER_KEYS if key in section)
    if not found:
        return
    ids = ", ".join(("anthropic", "openai", "openai-codex", "google", "openrouter"))
    raise ConfigSchemaError(
        f"Config at {path}: " + _PROVIDERS_RESHAPED.format(ids=ids, found=", ".join(sorted(set(found))[:6]))
    )


#: Settings this release removed, and how to treat a config that still carries
#: one. Each entry says where the key sits, what the release that wrote it left
#: there by default, and what to tell somebody whose value was not that.
#:
#: The rule is the one ``_refuse_removed_providers`` already holds to, and it
#: exists for the same reason: every config this project has written carries a
#: key for every setting the release that wrote it declared, so refusing on the
#: key alone fails every existing config over a value nobody chose. An empty
#: mapping or an untouched default carries no information, and refusing over
#: noise leaves somebody unable to start the tool at all. A value that *would*
#: have changed behaviour is a different thing and is named, with where it goes
#: now.
#:
#: Not a migration facility. Nothing is rewritten and nothing is carried across;
#: the key stops mattering the day it stops being written, and ``ddeharness
#: onboard`` writes a fresh file.
_RETIRED_SETTINGS: tuple[dict[str, Any], ...] = (
    {
        "where": ("agents", "defaults"),
        "keys": ("modelOverrides", "model_overrides"),
        "harmless": ({}, None),
        "message": (
            "agents.defaults.modelOverrides held {held}, and nothing reads it any more: it was keyed by a "
            "*substring* of a model id and consumed by a driver this release does not use, so those values "
            "were being accepted and dropped. Move each one into that model's row in "
            "providers.<provider>.models, which is keyed by the exact model id and is read by the "
            "request path, then delete agents.defaults.modelOverrides."
        ),
    },
    {
        "where": ("providers", "*"),
        "keys": ("endpoints",),
        "harmless": ([], None),
        "message": (
            "providers.{section}.endpoints held {held}, and nothing reads it any more: the model layer "
            "serves one address per provider, and only the first entry's key was ever sent. Put that "
            "entry's key in providers.{section}.apiKey and its address in providers.{section}.baseUrl, "
            "then delete providers.{section}.endpoints. A second account is a second provider section."
        ),
    },
    {
        "where": ("providers", "*"),
        "keys": ("endpointStrategy", "endpoint_strategy"),
        # "sticky" is the only behaviour there has ever been -- one address,
        # used until it fails -- so a config asking for it is asking for what
        # it gets; only "round_robin" names a thing that is gone.
        "harmless": ("sticky", None, ""),
        "message": (
            "providers.{section}.endpointStrategy is {held}, and the multi-endpoint failover it steered "
            "is gone: the model layer serves one address per provider. Delete the key -- there is "
            "nothing to replace it with."
        ),
    },
    {
        "where": ("providers", "*"),
        "keys": ("apiKeyList", "api_key_list"),
        "harmless": ([], None),
        "message": (
            "providers.{section}.apiKeyList held {held}, and key rotation is gone: only the first key "
            "was ever sent. Keep that one key in providers.{section}.apiKey and delete "
            "providers.{section}.apiKeyList."
        ),
    },
    {
        "where": ("providers", "*"),
        "keys": ("deployment",),
        "harmless": ("", None),
        "message": (
            "providers.{section}.deployment is {held}, and the deployment is no longer a connection "
            "parameter of its own: the model layer addresses it by the model id. List that deployment "
            "name in providers.{section}.models (and name it in agents.defaults.model), then delete "
            "providers.{section}.deployment. A models row's catalogModel is where you say which "
            "catalogue model it serves."
        ),
    },
    {
        "where": ("providers", "*"),
        "keys": ("apiVersion", "api_version"),
        # The value every release that wrote this key seeded by default.
        "harmless": ("2024-10-21", "", None),
        "message": (
            "providers.{section}.apiVersion is {held}, and nothing reads it any more: the model layer's "
            "Azure route pins api-version=v1. Delete the key -- there is nothing to replace it with."
        ),
    },
    {
        "where": ("agents", "defaults"),
        "keys": ("provider",),
        # "auto" is the value every release that wrote this key seeded by
        # default, and it means "the model id says" -- which is now the only
        # rule there is, so a config asking for it is asking for what it gets.
        "harmless": ("auto", "", None),
        "message": (
            "agents.defaults.provider is {held}, and the field is gone: a model id names its own "
            'provider now (agents.defaults.model = "<provider>/<model>"), so this field said the same '
            "thing a second time and won when the two disagreed. Write the provider into "
            "agents.defaults.model and delete agents.defaults.provider."
        ),
    },
    {
        "where": ("agents", "defaults"),
        "keys": ("llmCallTimeout", "llm_call_timeout"),
        "harmless": (600, None),
        "message": (
            "agents.defaults.llmCallTimeout is {held}, and there is no total call timeout any more: "
            "every request is streamed and bounded per silence instead. Set "
            "agents.defaults.llmFirstTokenTimeout (the wait before the first event) and "
            "agents.defaults.llmIdleTimeout (every gap after it), then delete "
            "agents.defaults.llmCallTimeout."
        ),
    },
    {
        "where": ("agents", "defaults"),
        "keys": ("temperature",),
        # 0.1 is what every release that wrote this key seeded by default.
        "harmless": (0.1, None),
        "message": (
            "agents.defaults.temperature is {held}, and there is no sampling temperature for every model "
            "any more: it was sent on every request, and a model that does not take the parameter refused "
            'every turn over it (the Codex wire answers "Unsupported parameter: temperature"). Write that '
            "temperature into the row of the model you want it for -- "
            "providers.<provider>.models[].temperature -- and delete agents.defaults.temperature."
        ),
    },
    {
        "where": ("providers", "*"),
        "keys": ("implementation",),
        # "native" is what every provider does now, so a config asking for it is
        # asking for what it gets; only "legacy" names a route that is gone.
        "harmless": (None, "", "native"),
        "message": (
            "providers.{section}.implementation is {held}, and that route no longer exists: every provider "
            "is served by the adapter its api names. Delete the key -- there is nothing to replace it with."
        ),
    },
    # ── The Curator's keys. History selection is deterministic now: there is
    # no planner to pick a model for, to time out, or to score relevance for,
    # and no second copy of the session for it to read.
    {
        "where": ("context",),
        "keys": ("engine",),
        # "unified" is the only engine there has ever been under this name, so
        # a config asking for it is asking for what it gets.
        "harmless": ("unified", "", None),
        "message": (
            "context.engine is {held}, and there is one context engine: the field stopped selecting "
            "anything when the legacy / curator / default split was collapsed. Delete the key -- there is "
            "nothing to replace it with."
        ),
    },
    {
        "where": ("context",),
        "keys": ("curatorModel", "curator_model"),
        "harmless": ("", None),
        "message": (
            "context.curatorModel is {held}, and the Curator's planning loop is gone: history selection is "
            "deterministic and makes no model call at all. Delete the key -- there is no model to point it "
            "at. agents.defaults.model is the conversation's model."
        ),
    },
    {
        "where": ("context",),
        "keys": ("curatorProvider", "curator_provider"),
        "harmless": ("", None, "auto"),
        "message": (
            "context.curatorProvider is {held}, and the Curator's planning loop is gone: history selection "
            "is deterministic and makes no model call at all. Delete the key -- there is nothing to replace "
            "it with."
        ),
    },
    {
        "where": ("context",),
        "keys": ("curatorTimeoutSeconds", "curator_timeout_seconds"),
        "harmless": (60, 60.0, None),
        "message": (
            "context.curatorTimeoutSeconds is {held}, and there is no planning call to time out: history "
            "selection is deterministic and local. Delete the key -- there is nothing to replace it with."
        ),
    },
    {
        "where": ("context",),
        "keys": ("fastPathThreshold", "fast_path_threshold"),
        "harmless": (0.6, 0.60, None),
        "message": (
            "context.fastPathThreshold is {held}, and there is no slow path for it to gate: every turn "
            "selects history the same deterministic way. Delete the key -- context.protectFirstN is the "
            "only selection setting left."
        ),
    },
    {
        "where": ("context",),
        "keys": ("relevanceDecay", "relevance_decay"),
        "harmless": (0.95, None),
        "message": (
            "context.relevanceDecay is {held}, and nothing has read it for some time: the per-message "
            "relevance scores it decayed are gone with the Curator's manifest. Delete the key -- selection "
            "is by recency and tool-exchange closure now."
        ),
    },
    {
        "where": ("context",),
        "keys": ("relevanceReferenceBoost", "relevance_reference_boost"),
        "harmless": (0.15, None),
        "message": (
            "context.relevanceReferenceBoost is {held}, and nothing has read it for some time: the "
            "per-message relevance scores it raised are gone with the Curator's manifest. Delete the key "
            "-- selection is by recency and tool-exchange closure now."
        ),
    },
    {
        "where": ("context",),
        "keys": ("archiveDir", "archive_dir"),
        "harmless": ("memory/.curator/archive", "", None),
        "message": (
            "context.archiveDir is {held}, and nothing writes there any more: the Curator's second copy of "
            "the session is gone and the session file is the archive. Delete the key -- files already "
            "written are left alone and can be deleted by hand."
        ),
    },
    # ── SkillForge's retrieval-service knobs. No code has read them since
    # retrieval became local BM25 plus the memory backend's own recall.
    {
        "where": ("skillForge",),
        "keys": ("embeddingModel", "embedding_model"),
        "harmless": ("default", "", None),
        "message": (
            "skillForge.embeddingModel is {held}, and nothing reads it: skill retrieval is local BM25 plus "
            "the memory backend's own recall, and that backend is configured under plugins.config. Delete "
            "the key."
        ),
    },
    {
        "where": ("skillForge",),
        "keys": ("embeddingUrl", "embedding_url"),
        "harmless": ("http://localhost:1357", "", None),
        "message": (
            "skillForge.embeddingUrl is {held}, and nothing reads it: skill retrieval makes no embedding "
            "call. Put the address of a memory service in plugins.config, and delete the key."
        ),
    },
    {
        "where": ("skillForge",),
        "keys": ("embeddingApiKey", "embedding_api_key"),
        "harmless": ("", None),
        "message": (
            "skillForge.embeddingApiKey is set, and nothing reads it: skill retrieval makes no embedding "
            "call. Put the credential of a memory service in plugins.config, and delete the key."
        ),
    },
    {
        "where": ("skillForge",),
        "keys": ("embeddingDimensions", "embedding_dimensions"),
        "harmless": (None,),
        "message": (
            "skillForge.embeddingDimensions is {held}, and nothing reads it: skill retrieval makes no "
            "embedding call. Delete the key."
        ),
    },
    {
        "where": ("skillForge",),
        "keys": ("rerankerEnabled", "reranker_enabled"),
        "harmless": (True, None),
        "message": (
            "skillForge.rerankerEnabled is {held}, and there is no reranking pass to switch off: skill "
            "retrieval is local BM25 plus the memory backend's own recall. Delete the key."
        ),
    },
    {
        "where": ("skillForge",),
        "keys": ("rerankerModel", "reranker_model"),
        "harmless": ("default", "", None),
        "message": (
            "skillForge.rerankerModel is {held}, and nothing reads it: there is no reranking pass. Delete the key."
        ),
    },
    {
        "where": ("skillForge",),
        "keys": ("rerankerUrl", "reranker_url"),
        "harmless": ("http://localhost:1357", "", None),
        "message": (
            "skillForge.rerankerUrl is {held}, and nothing reads it: there is no reranking pass. Delete the key."
        ),
    },
    {
        "where": ("skillForge",),
        "keys": ("rerankerApiKey", "reranker_api_key"),
        "harmless": ("", None),
        "message": (
            "skillForge.rerankerApiKey is set, and nothing reads it: there is no reranking pass. Delete the key."
        ),
    },
    {
        "where": ("skillForge",),
        "keys": ("memory",),
        # {"enabled": true} is what onboarding wrote into every config it
        # seeded, so a block holding only that is asking for what it gets.
        "harmless": ({"enabled": True}, None),
        "message": (
            "skillForge.memory held {held}, and nothing reads any of it: durable extraction is the memory "
            "backend's, configured under plugins.config, and these thresholds were never wired to it. "
            "Delete the block."
        ),
    },
)


def _drop_retired_settings(data: dict[str, Any]) -> str | None:
    """Drop every retired key left at its old default; report one that was set.

    Reported rather than raised, so the caller can put
    ``_refuse_removed_providers`` first: a section for a provider this project
    removed is gone whatever else it holds, and "your endpoints list moved" is
    the wrong sentence to hand somebody whose whole provider went away. The
    whole pass still runs, so a refusal never leaves a harmless key behind for
    the schema to trip over.
    """
    refusals: list[str] = []
    for entry in _RETIRED_SETTINGS:
        for section, node in _retired_nodes(data, entry["where"]):
            for key in entry["keys"]:
                if not isinstance(node, dict) or key not in node:
                    continue
                held = node[key]
                if held in entry["harmless"] or (isinstance(held, (dict, list)) and not held):
                    del node[key]
                    logging.getLogger(__name__).debug("config: dropped retired setting %s", key)
                    continue
                refusals.append(entry["message"].format(held=_summarise(held), section=section))
    return refusals[0] if refusals else None


def _retired_nodes(data: dict[str, Any], where: tuple[str, ...]) -> list[tuple[str, Any]]:
    """``(section name, node)`` for each place a retired key may sit.

    A ``*`` matches every key at that level, which is how a per-provider setting
    is found without naming the providers.
    """
    found: list[tuple[str, Any]] = [("", data)]
    for part in where:
        stepped: list[tuple[str, Any]] = []
        for name, node in found:
            if not isinstance(node, dict):
                continue
            if part == "*":
                stepped.extend((str(key), value) for key, value in node.items())
            elif isinstance(node.get(part), dict):
                stepped.append((name, node[part]))
        found = stepped
    return found


def _summarise(held: Any) -> str:
    """What the key held, said without quoting a value that may be a secret.

    A dict is named by its keys, which are field names; a list is named by its
    length alone, because a retired list held credentials (an ``endpoints``
    entry, an ``apiKeyList``) and its elements are the secrets themselves. The
    rule for an error message is the same everywhere here: name the shape, not
    the contents.
    """
    if isinstance(held, dict):
        return f"{len(held)} entr{'y' if len(held) == 1 else 'ies'} ({', '.join(sorted(map(str, held))[:5])})"
    if isinstance(held, (list, tuple)):
        return f"{len(held)} entr{'y' if len(held) == 1 else 'ies'}"
    return repr(held)


def _refuse_removed_providers(path: Path, data: dict[str, Any]) -> None:
    """Stop a config that still *configures* a provider this project removed.

    Dropping the section from the schema is not enough to say so: an
    unrecognized provider key is accepted, because that is how a vendor with no
    spec of its own is configured. A config that named a removed provider would
    load quietly, keep its credential and route nowhere, which reads as a
    mystery rather than as the removal it is.

    An untouched section is a different thing and is dropped in silence. Every
    config this project has ever written carries a block for every provider it
    declared, so refusing on the key alone would fail every config in the world
    over a section nobody filled in -- the same reason ``_drop_retired_keys``
    exists one function above.

    Said here rather than by a validator, because a validator's error quotes
    the value it rejected, and the value is a providers block holding the
    user's keys.
    """
    from opendde_harness.providers.pi_ids import removed_message as removed_provider_message

    providers = data.get("providers")
    if not isinstance(providers, dict):
        return
    for key in list(providers):
        reason = removed_provider_message(str(key))
        if reason is None:
            continue
        if not _section_was_filled_in(providers[key]):
            del providers[key]
            logging.getLogger(__name__).debug("config: dropped retired default provider %s", key)
            continue
        raise ConfigSchemaError(f"Config at {path}: providers.{key} -- {reason}")


def _section_was_filled_in(section: Any) -> bool:
    """Did anyone put something in this provider entry?

    Compared against a fresh instance rather than against ``{}``, because a
    config written by an earlier release carries a block for every provider it
    declared, defaults included -- "has any truthy value" would call every
    untouched block configured. Under the current shape an entry is only there
    because somebody wrote it, so this is nearly always true; it still answers
    for the leftovers of a release that wrote them all.
    """
    from opendde_harness.config.schema import ProviderEntry

    if not isinstance(section, dict):
        # A default write is always a block. Anything else -- null, a bare
        # string -- was put there by hand, and is not something to drop in
        # silence.
        return True
    try:
        return ProviderEntry.model_validate(section) != ProviderEntry()
    except ValidationError:
        # Fields we cannot even parse are not the leftovers of a default write.
        return True


def _unknown_keys(exc: ValidationError) -> list[str]:
    return [".".join(str(part) for part in err["loc"]) for err in exc.errors() if err["type"] == "extra_forbidden"]


def _schema_error(path: Path, exc: ValidationError) -> ConfigSchemaError:
    """The validation failure as a message that names keys and never values.

    Built from ``exc.errors()`` rather than from ``str(exc)``, which renders an
    ``input_value=`` for every error. The rejected input is whatever block
    failed, and several of them hold secrets -- a providers entry, a web-search
    key, an MCP server's headers -- so the readable form of a pydantic error is
    one this project cannot print. The location and the message carry everything
    a person needs to find the key and fix it.
    """
    unknown = _unknown_keys(exc)
    lines = [f"Config at {path} fails schema validation:"]
    if unknown:
        lines.append("unknown key(s): " + ", ".join(unknown))
        lines.append(
            "Remove them, or run `ddeharness onboard` to write a fresh config. "
            "Keys from earlier releases are not migrated."
        )
    for error in exc.errors():
        where = ".".join(str(part) for part in error.get("loc") or ()) or "config"
        message = str(error.get("msg") or "").removeprefix("Value error, ")
        lines.append(f"{where}: {message}")
    return ConfigSchemaError("\n".join(lines))


def read_raw_or_raise(path: Path) -> dict[str, Any]:
    """Read a config file as raw JSON for a read-modify-write cycle.

    Returns ``{}`` ONLY when the file is absent. A present-but-unreadable file
    raises :class:`ConfigReadError` rather than returning ``{}`` -- returning
    ``{}`` and then writing was the bug that wiped a real config over a lone
    JSON syntax error (e.g. a // comment). The single read path for every
    ``update_*`` write module.
    """
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}  # empty file: no data to lose, safe to create fresh (like absent)
        data = json.loads(text)
        # A valid-JSON non-object (null / list / scalar) is not a usable config;
        # return {} so callers get a mapping (not None) without an AttributeError.
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ConfigReadError(
            f"{path} is not valid JSON ({exc}). Fix it first (JSON allows no comments or "
            "trailing commas); your config was left unchanged."
        ) from exc


def _file_stamp(path: Path) -> tuple[int, int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    cached = _cache.get(str(path))
    if cached is not None and cached[0] == _file_stamp(path):
        return cached[1].model_copy(deep=True)

    config: Config | None = None
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            # Boot on defaults for a malformed file (a transient mid-write race
            # shouldn't brick callers) but warn LOUDLY -- a persistent syntax
            # error would else revert every setting with no visible cause.
            # Raising instead needs atomic save_config first (separate change).
            msg = (
                f"config at {path} is not valid JSON ({e}) -- IGNORING it and running on "
                "DEFAULTS. Fix the file (JSON allows no comments or trailing commas) and restart."
            )
            # Single user-visible channel: the stderr print (visible under any
            # loguru sink config). The log-file trace uses stdlib logging, NOT
            # loguru — loguru's default sink echoes DEBUG to stderr, which
            # would re-duplicate the warning on plain CLI runs; the stdlib
            # record reaches the file sink via the CLI's logging intercept.
            if str(path) not in _warned_paths:
                _warned_paths.add(str(path))
                print(f"WARNING: {msg}", file=sys.stderr)
            logging.getLogger(__name__).debug(msg)
        else:
            # A clean parse re-arms the warning: the dedup exists to silence
            # repeated loads of the same broken state within one command, not
            # to spend the one warning a long-lived process (the TUI RPC
            # server reloads every turn) gets for a later re-breakage.
            _warned_paths.discard(str(path))
            if isinstance(data, dict):
                _drop_retired_keys(data)
                retired = _drop_retired_settings(data)
                _refuse_removed_providers(path, data)
                _refuse_old_providers_shape(path, data)
                if retired is not None:
                    raise ConfigSchemaError(retired)
            try:
                config = Config.model_validate(data)
            except ValidationError as e:
                # Schema mismatch is a user/programmer error — surface
                # loudly rather than masking with defaults. Silently
                # using defaults makes "feature X did nothing" debug
                # take 24h instead of 24s.
                raise _schema_error(path, e) from e
            stamp = _file_stamp(path)
            if stamp is not None:
                _cache[str(path)] = (stamp, config.model_copy(deep=True))

    if config is None:
        config = Config()

    return config


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save the base blocks of ``config`` to file.

    The feature blocks (``FEATURE_FIELDS``) are not written: the onboarding
    wizard seeds the user-facing subset of them with
    ``config.update.init_extension_block_defaults`` and every later change
    is patched in place by the ``update_*`` modules.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(by_alias=True, exclude=set(FEATURE_FIELDS))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
