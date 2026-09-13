"""Says when a newer OpenDDE Harness is on PyPI, and how to get it.

The package is distributed on PyPI, so PyPI is what is asked: its view of the
project names the latest release, and that is compared with the version this
process runs. The answer is cached in the runtime cache dir and refreshed at
most once a day, in a daemon thread, so a launch never waits on the network --
except the little it takes for a refresh already in flight to land, so the
first launch after a release is the one that says so rather than the second.

Where it is said: the TUI's status bar and one line of its transcript at
launch (``session.info`` carries the version and the command), and
``ddeharness doctor``. The command is the one this install kind updates with
(``upgrade_commands.upgrade_command``); a source checkout is a developer's and
is told nothing. Any network or parse failure is swallowed: a nudge must never
break a launch.

Set ``OPENDDE_HARNESS_NO_UPDATE_CHECK=1`` to opt out of both the fetch and the notice.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_CACHE_NAME = "update_check.json"
_REFRESH_TTL_SECONDS = 24 * 60 * 60
_OPT_OUT_ENV = "OPENDDE_HARNESS_NO_UPDATE_CHECK"
_TRUTHY = {"1", "true", "yes", "on"}

#: The refresh in flight, if one is: what a notice asked for right after a
#: launch can wait a moment for, instead of reading yesterday's answer.
_refreshing: threading.Thread | None = None
_lock = threading.Lock()


def _cache_path() -> Path:
    # Resolved per call, not at import: get_cache_dir() follows the active
    # config path, which set_config_path() can move after this module loads.
    from opendde_harness.config import paths

    return paths.get_cache_dir() / _CACHE_NAME


def _disabled() -> bool:
    return os.environ.get(_OPT_OUT_ENV, "").strip().lower() in _TRUTHY


def _release_prefix(value: str) -> str:
    """Reduce ``0.2.0rc1`` / ``0.1.9.dev1`` to the ``X.Y.Z`` it builds on.

    The strict parser matches the whole string, so a prerelease or dev suffix
    would read as unparseable and silence the hint for anyone running one. The
    suffix is dropped rather than ordered: an rc of a release compares equal to
    it, so an rc user is not nagged to "upgrade" to the version they are
    already testing.
    """
    raw = value.strip().lstrip("vV")
    parts = []
    for part in raw.split(".")[:3]:
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            return raw
        parts.append(digits)
    return ".".join(parts) if len(parts) == 3 else raw


def _version_key(value: str) -> tuple[int, int, int] | None:
    """Parse ``1.2.3`` / ``v1.2.3`` / ``1.2.3rc1``, ``None`` when unparseable.

    ``upgrade_commands._version_key`` is the single source of truth for the
    grammar; it raises for anything it cannot read, which here just means
    "show no notice".
    """
    from opendde_harness.cli.upgrade_commands import UpgradeError
    from opendde_harness.cli.upgrade_commands import _version_key as strict_key

    try:
        return strict_key(_release_prefix(value))
    except (UpgradeError, AttributeError):
        return None


def _read_cache() -> dict | None:
    try:
        parsed = json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    # A hand-edited cache can be valid JSON and still not an object; without
    # this guard the .get() below raises and takes `ddeharness tui` down with it.
    return parsed if isinstance(parsed, dict) else None


def _write_cache(latest_version: str | None, *, now: float) -> None:
    payload: dict[str, object] = {"checked_at": now}
    if latest_version is not None:
        payload["latest_version"] = latest_version
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def _stale(cache: dict | None) -> bool:
    if cache is None:
        return True
    checked_at = cache.get("checked_at")
    return not isinstance(checked_at, (int, float)) or (time.time() - checked_at) >= _REFRESH_TTL_SECONDS


def _refresh() -> None:
    # Imported lazily: the fetch pulls in httpx, which stays off the
    # session-create hot path (this runs in a daemon thread).
    cache = _read_cache() or {}
    previous = cache.get("latest_version")
    keep = previous if isinstance(previous, str) else None

    try:
        from opendde_harness.cli.upgrade_commands import fetch_latest_version

        _write_cache(fetch_latest_version(), now=time.time())
    except Exception:
        # Offline, or PyPI answered with something that is not a version.
        # Stamp checked_at anyway so we back off for a full TTL instead of
        # refetching on every launch, and keep whatever version we had.
        _write_cache(keep, now=time.time())


def _command() -> str | None:
    """How this install updates, or ``None`` for one that is told nothing."""
    from opendde_harness.cli.upgrade_commands import upgrade_command

    return upgrade_command()


def maybe_refresh_async() -> threading.Thread | None:
    """Refresh the cached latest version in the background if it is stale.

    Fire-and-forget: spawns a daemon thread only when the cache is missing or
    older than the TTL, so a normal launch touches the network at most once a
    day and never blocks. Returns the thread doing it, this call's or an
    earlier one's, for a caller willing to wait a moment for its answer.
    Installs that are told nothing -- a source checkout -- skip the fetch.
    """
    global _refreshing

    if _disabled() or _command() is None or not _stale(_read_cache()):
        return None

    with _lock:
        if _refreshing is None or not _refreshing.is_alive():
            _refreshing = threading.Thread(target=_refresh, daemon=True)
            _refreshing.start()
        return _refreshing


def update_notice(current_version: str, *, wait: float = 0.0) -> tuple[str, str] | None:
    """``(latest version, the command that installs it)`` when PyPI has a
    newer release than ``current_version``.

    ``None`` when up to date, when the check is opted out of, when the cache
    is absent or unreadable, when either version is unparseable, or when this
    install is one that is told nothing. ``wait`` is how long to give a refresh
    in flight -- started here if the cache is stale -- before reading the cache.
    """
    if _disabled():
        return None
    command = _command()
    if command is None:
        return None

    if wait > 0:
        refreshing = maybe_refresh_async()
        if refreshing is not None:
            refreshing.join(wait)

    cache = _read_cache()
    latest = cache.get("latest_version") if cache else None
    if not isinstance(latest, str):
        return None

    latest_key = _version_key(latest)
    current_key = _version_key(current_version)
    if latest_key is None or current_key is None or latest_key <= current_key:
        return None
    return latest, command


def sentence(current_version: str, notice: tuple[str, str]) -> str:
    """The notice as one line to print: what is out, what this is, what to run."""
    latest, command = notice
    return f"OpenDDE Harness {latest} is on PyPI; this is {current_version}. Update with: {command}"
