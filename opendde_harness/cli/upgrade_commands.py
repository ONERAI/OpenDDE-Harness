"""``ddeharness upgrade``: the update this install kind takes, run for you.

OpenDDE Harness is distributed on PyPI, and an install is one of three kinds:
a uv tool (``uv tool install opendde-harness``, the documented way), a plain
``pip`` or ``uv pip`` install into some environment, or an editable source
checkout. Each updates with its own command -- ``uv tool upgrade``, ``pip
install --upgrade``, or a ``git pull`` that is the developer's own business --
and this command runs it, or with ``--check`` only says whether PyPI has
anything newer.

The same two facts feed the update notice (:mod:`cli.update_notice`): what the
latest version on PyPI is, and which command gets it here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import httpx
import typer
from rich.console import Console

PACKAGE = "opendde-harness"
#: PyPI's JSON view of the project: ``info.version`` is its latest release.
PYPI_JSON = f"https://pypi.org/pypi/{PACKAGE}/json"
_VERSION_RE = re.compile(r"^v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_REQUEST_TIMEOUT = 10.0

#: The three kinds of install, and the command each one updates with.
UV_TOOL = "uv-tool"
PIP = "pip"
EDITABLE = "editable"
UPGRADE_COMMANDS: dict[str, str | None] = {
    UV_TOOL: f"uv tool upgrade {PACKAGE}",
    PIP: f"pip install --upgrade {PACKAGE}",
    # A source checkout is a developer's: they pull and reinstall themselves.
    EDITABLE: None,
}

console = Console()


class UpgradeError(RuntimeError):
    pass


class ReleaseLookupError(UpgradeError):
    """PyPI could not say what the latest release is. The local installation
    is fine, so the caller must not advise reinstalling."""


def _version_key(value: str) -> tuple[int, int, int]:
    match = _VERSION_RE.fullmatch(value)
    if match is None:
        raise UpgradeError(f"Unsupported OpenDDE Harness version: {value!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _current_version() -> str:
    return metadata.version(PACKAGE)


def fetch_latest_version(client: httpx.Client | None = None) -> str:
    """The latest release on PyPI, as ``X.Y.Z``.

    PyPI's ``info.version`` is the project's latest release, pre-releases
    excluded, so nobody is told to upgrade to an rc. Anything but a version
    string there, or no answer at all, is a :class:`ReleaseLookupError`.
    """
    if client is None:
        with httpx.Client(timeout=_REQUEST_TIMEOUT, follow_redirects=True) as owned:
            return fetch_latest_version(owned)
    try:
        response = client.get(PYPI_JSON, headers={"Accept": "application/json"})
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ReleaseLookupError(f"PyPI could not be asked for the latest release: {exc}") from exc
    info = payload.get("info") if isinstance(payload, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    if not isinstance(version, str):
        raise ReleaseLookupError("Malformed PyPI response: no version in it")
    try:
        return ".".join(str(part) for part in _version_key(version))
    except UpgradeError as exc:
        raise ReleaseLookupError(str(exc)) from exc


def _editable() -> bool:
    """Was this package installed from a source directory in editable mode?"""
    try:
        raw = metadata.distribution(PACKAGE).read_text("direct_url.json")
        data = json.loads(raw) if raw else None
    except (metadata.PackageNotFoundError, ValueError):
        return False
    directory = data.get("dir_info") if isinstance(data, dict) else None
    return bool(isinstance(directory, dict) and directory.get("editable"))


def _uv_tool() -> bool:
    """Does uv's receipt beside this interpreter name the package as its tool?"""
    receipt = Path(sys.prefix) / "uv-receipt.toml"
    try:
        tool = tomllib.loads(receipt.read_text(encoding="utf-8")).get("tool")
    except (OSError, tomllib.TOMLDecodeError):
        return False
    requirements = tool.get("requirements") if isinstance(tool, dict) else None
    return isinstance(requirements, list) and any(
        isinstance(item, dict) and item.get("name") == PACKAGE for item in requirements
    )


def install_kind() -> str:
    """Which of the three kinds this install is. Never raises: metadata this
    cannot read is treated as a plain pip install, whose command is the
    generic one."""
    if _editable():
        return EDITABLE
    if _uv_tool():
        return UV_TOOL
    return PIP


def upgrade_command(kind: str | None = None) -> str | None:
    """The command that updates this install, or ``None`` when none is offered."""
    return UPGRADE_COMMANDS[kind or install_kind()]


def _upgrade_argv(kind: str) -> list[str]:
    """What to run, resolved: ``uv`` from PATH, ``pip`` as this interpreter's own."""
    if kind == UV_TOOL:
        uv = shutil.which("uv")
        if uv is None:
            raise UpgradeError("uv was not found on PATH")
        return [uv, "tool", "upgrade", PACKAGE]
    if kind == PIP:
        return [sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE]
    raise UpgradeError(
        "Editable OpenDDE Harness installations cannot be upgraded automatically. "
        "Pull the source checkout and reinstall it."
    )


def register(app: typer.Typer) -> None:
    @app.command()
    def upgrade(
        check: bool = typer.Option(
            False,
            "--check",
            help="Check PyPI for a newer OpenDDE Harness release without installing it.",
        ),
    ) -> None:
        """Check for and install the latest OpenDDE Harness release from PyPI."""
        try:
            current_version = _current_version()
            latest = fetch_latest_version()
            current_key = _version_key(current_version)
            latest_key = _version_key(latest)

            if current_key == latest_key:
                console.print(f"OpenDDE Harness {current_version} is up to date.")
                return
            if current_key > latest_key:
                console.print(
                    f"OpenDDE Harness {current_version} is newer than the latest release on PyPI, {latest}; "
                    "no downgrade was performed."
                )
                return
            kind = install_kind()
            command = upgrade_command(kind)
            if check:
                console.print(f"OpenDDE Harness upgrade available: {current_version} -> {latest}")
                console.print(
                    f"Run [cyan]{command}[/cyan] to install it."
                    if command
                    else "This is a source checkout: pull it and reinstall."
                )
                return
            argv = _upgrade_argv(kind)
        except ReleaseLookupError as exc:
            console.print(f"[red]Unable to check for an upgrade:[/red] {exc}")
            raise typer.Exit(1) from exc
        except (UpgradeError, metadata.PackageNotFoundError) as exc:
            console.print(f"[red]Unable to upgrade OpenDDE Harness:[/red] {exc}")
            raise typer.Exit(1) from exc

        # The updater replaces this very install, so it takes over the process
        # rather than running under it.
        console.print(f"Running [cyan]{' '.join(argv)}[/cyan]")
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            os.execv(argv[0], argv)
        except OSError as exc:
            console.print(f"[red]Could not start the upgrade:[/red] {exc}")
            raise typer.Exit(1) from exc
