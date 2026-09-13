"""pi's credential store, as this project reads it.

pi-ai keeps its credentials in one file (``{provider id: credential}``, mode
0600, written atomically) and owns every write to it: a login run through the
model service, an OAuth refresh, and the keys ``configure`` hands over all land
there through pi's own serialised store. This module is the Python side's read
of that file, and nothing here writes it -- a second writer working from its
own snapshot of the file would put back what pi had just replaced.

**Nothing here reports a secret.** :func:`stored_kind` answers with a word --
``"oauth"``, ``"api_key"``, ``"invalid"``, ``""`` -- and never with a value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: What :func:`stored_kind` says when the store file is there and cannot be read.
KIND_INVALID = "invalid"


def stored_kind(name: str, *, store_path: Path | None = None) -> str:
    """What pi's store holds for this section, as one word and never a value.

    ``"oauth"`` for a sign-in, ``"api_key"`` for a key, :data:`KIND_INVALID`
    when the file is there and unreadable, ``""`` when there is nothing.

    The last two are different answers and are fixed differently -- one is a
    sign-in, the other needs the file moved aside first -- which is why a
    damaged store is not reported as "never signed in".
    """
    path = store_path or _store_path()
    if not path.exists():
        return ""
    entries = _read(path)
    if entries is None:
        return KIND_INVALID
    entry = entries.get(name)
    kind = entry.get("type") if isinstance(entry, dict) else None
    return kind if kind in {"api_key", "oauth"} else ""


def stored_oauth(name: str, *, store_path: Path | None = None) -> bool:
    """Is a sign-in stored for this section? The whole OAuth presence question.

    Read by ``providers.auth``'s gate, and through it by ``provider list``,
    ``status``, ``doctor`` and the TUI's setup panel, so all of them answer from
    the one file that decides it.
    """
    return stored_kind(name, store_path=store_path) == "oauth"


def _store_path() -> Path:
    from opendde_harness.providers.pi_service import credential_store_path

    return credential_store_path()


def _read(path: Path) -> dict[str, Any] | None:
    """Every credential the store holds, or None when there is no readable store.

    A file that cannot be parsed reads as None, and :func:`stored_kind` is what
    tells that apart from a file that is not there.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None
