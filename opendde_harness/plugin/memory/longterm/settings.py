"""The long-term memory root and the library's settings file inside it.

The library reads its settings from ``<root>/<config toml>`` through its own
loader, so this is the one writer of that file. What it writes is small: the
address a server for this root listens on (``[api]``), and placeholders in
the two model sections (``[llm]``, ``[multimodal]``) so the library builds a
client at all -- the client it builds is this plugin's own
(:mod:`._service_llm`), which reads none of those values and runs every call
on the conversation's default model through the model service. There is no
key, endpoint or model of memory's own to configure, and no embedding or
rerank role: the library treats both as optional, and without them it
searches by keyword.

**Which root.** Always :func:`memory_root` -- ``<data dir>/memory``. The
library resolves its root from its root env variable; this module *writes*
that variable and never reads it as an input. Following an ambient value
silently pointed the program at a root nothing decided on, so the next run
without the variable reported no memories while they sat on disk.
"""

from __future__ import annotations

import logging
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from opendde_harness.plugin.memory.longterm._library import (
    CONFIG_FILENAME,
    OME_CONFIG_FILENAME,
    ROOT_ENV_VAR,
    config_templates,
)

logger = logging.getLogger(__name__)

_DATA_SUBDIR = "memory"

#: What the model sections hold. Placeholders: the library refuses to build a
#: client from an empty key or address, and the client it then builds is this
#: plugin's, which ignores all three. Spelled so that anyone reading the file
#: can tell nothing here is a credential.
PLACEHOLDER_MODEL = "conversation-model"
PLACEHOLDER_KEY = "served-by-the-model-service"
PLACEHOLDER_URL = "http://model-service.invalid"

#: The sections that get the placeholders: extraction, and the parser's
#: multimodal model, which is the same conversation model.
MODEL_SECTIONS = ("llm", "multimodal")


def memory_root() -> Path:
    """The memory root: ``<data dir>/memory``.

    Under the program's data directory, so it follows ``--config`` and reads as
    the program's property rather than a squatter in the library's default
    root. There is one root and the program owns it; nothing in the user's
    config can point this elsewhere.
    """
    from opendde_harness.config.paths import get_data_dir

    return get_data_dir() / _DATA_SUBDIR


def get_memory_config_path() -> Path:
    """Path of the user-level memory config toml."""
    return memory_root() / CONFIG_FILENAME


def configure_memory_env(root: Path | str | None = None) -> None:
    """Point the memory library at ``root`` (default: :func:`memory_root`).

    Assigns rather than ``setdefault``: an ambient value is not an input to the
    program's choice of root. Must run before the library's ``load_settings()``
    -- which is cached -- first executes, or in-process library imports keep
    the earlier root.
    """
    resolved = Path(root).expanduser() if root is not None else memory_root()
    os.environ[ROOT_ENV_VAR] = str(resolved)


def ensure_memory_home(root: Path | str | None = None) -> None:
    """Make the root a place the library can start from. Idempotent.

    The config toml and ``ome.toml`` are created from the shipped templates
    when absent (without ``ome.toml`` the library's engine raises on start),
    and the model sections are written with the placeholders on every call:
    the file is generated, and a value anyone put there would be read by
    nothing.
    """
    base = Path(root).expanduser() if root is not None else memory_root()
    base.mkdir(parents=True, exist_ok=True)

    templates = config_templates()
    if templates is not None:
        for target, template in ((base / CONFIG_FILENAME, templates[0]), (base / OME_CONFIG_FILENAME, templates[1])):
            if not target.exists():
                shutil.copy2(template, target)
                logger.info("created %s from template", target)

    path = base / CONFIG_FILENAME
    data = _load(path)
    changed = False
    for section in MODEL_SECTIONS:
        current = dict(data.get(section) or {})
        wanted = {**current, "model": PLACEHOLDER_MODEL, "api_key": PLACEHOLDER_KEY, "base_url": PLACEHOLDER_URL}
        if wanted != current:
            data[section] = wanted
            changed = True
    if changed or not path.exists():
        _write_atomic(path, data)


def memory_ready(root: Path | str | None = None) -> bool:
    """Whether the root's settings file is one a server can start from."""
    base = Path(root).expanduser() if root is not None else memory_root()
    llm = _load(base / CONFIG_FILENAME).get("llm") or {}
    return bool(llm.get("model") and llm.get("api_key") and llm.get("base_url"))


def load_memory_config() -> dict[str, Any]:
    """Return the parsed user-level toml, or ``{}`` when absent."""
    return _load(get_memory_config_path())


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as TOML via temp-file + rename, so a Ctrl+C mid-write
    never leaves a half-written toml.

    No sidecar lock: the memory server owns ``<root>/.lock`` as a lock file,
    and the ``.lock/`` directory that ``atomic_replace`` creates would shadow it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        tomli_w.dump(data, f)
    os.replace(tmp, path)


def memory_declared_address() -> str | None:
    """The address the root's config toml declares, or ``None`` when unset.

    This is the authority on where a server for that root listens: the server
    reads ``[api]`` at startup, and nothing overrides it on the command line.
    Everything else -- the plugin's ``base_url``, a doctor probe -- is a copy.
    """
    api = load_memory_config().get("api") or {}
    host = api.get("host")
    port = api.get("port")
    if not host or not port:
        return None
    return f"http://{host}:{port}"


def set_memory_api(*, host: str, port: int) -> None:
    """Record the address a server for this root must listen on.

    Written into the toml rather than passed as ``--port``: a command-line
    override left the file describing an address nobody was using. The root is
    self-describing, with no second place to drift out of sync.
    """
    path = get_memory_config_path()
    data = _load(path)
    data["api"] = {**(data.get("api") or {}), "host": host, "port": int(port)}
    _write_atomic(path, data)


__all__ = [
    "MODEL_SECTIONS",
    "PLACEHOLDER_KEY",
    "PLACEHOLDER_MODEL",
    "PLACEHOLDER_URL",
    "configure_memory_env",
    "ensure_memory_home",
    "get_memory_config_path",
    "load_memory_config",
    "memory_declared_address",
    "memory_ready",
    "memory_root",
    "set_memory_api",
]
