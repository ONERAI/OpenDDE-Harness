"""``ddeharness upgrade`` asks PyPI and runs the update this install kind takes."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from opendde_harness.cli import upgrade_commands


class _Response:
    def __init__(self, payload, *, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("nope", request=None, response=None)  # type: ignore[arg-type]

    def json(self):
        return self._payload


class _Client:
    def __init__(self, payload, *, status=200):
        self.payload, self.status, self.urls = payload, status, []

    def get(self, url, **_):
        self.urls.append(url)
        return _Response(self.payload, status=self.status)


def test_the_latest_version_is_pypis_own():
    """PyPI, not the GitHub release page: the package is installed from PyPI,
    so a release is what PyPI has, and ``info.version`` is its latest."""
    client = _Client({"info": {"version": "0.0.9"}, "releases": {"0.0.9": []}})

    assert upgrade_commands.fetch_latest_version(client) == "0.0.9"
    assert client.urls == ["https://pypi.org/pypi/opendde-harness/json"]

    for payload in ({"info": {}}, {"info": {"version": "tomorrow"}}, ["not", "an", "object"]):
        with pytest.raises(upgrade_commands.ReleaseLookupError):
            upgrade_commands.fetch_latest_version(_Client(payload))
    with pytest.raises(upgrade_commands.ReleaseLookupError):
        upgrade_commands.fetch_latest_version(_Client({}, status=503))


def test_each_install_kind_has_its_own_command(tmp_path, monkeypatch):
    """A uv tool is updated by uv, a pip install by pip, and a source checkout
    by its developer -- read off the install's own metadata."""
    from importlib import metadata

    receipt = tmp_path / "uv-receipt.toml"
    monkeypatch.setattr(upgrade_commands.sys, "prefix", str(tmp_path))
    direct_url: dict = {"text": None}
    monkeypatch.setattr(
        metadata, "distribution", lambda name: SimpleNamespace(read_text=lambda _file: direct_url["text"])
    )

    assert upgrade_commands.install_kind() == upgrade_commands.PIP
    assert upgrade_commands.upgrade_command() == "pip install --upgrade opendde-harness"

    receipt.write_text('[tool]\nrequirements = [{ name = "opendde-harness" }]\n')
    assert upgrade_commands.install_kind() == upgrade_commands.UV_TOOL
    assert upgrade_commands.upgrade_command() == "uv tool upgrade opendde-harness"

    receipt.write_text('[tool]\nrequirements = [{ name = "something-else" }]\n')
    assert upgrade_commands.install_kind() == upgrade_commands.PIP
    receipt.write_text("not toml at all [[[")
    assert upgrade_commands.install_kind() == upgrade_commands.PIP, "unreadable metadata is not a crash"

    direct_url["text"] = json.dumps({"url": "file:///src", "dir_info": {"editable": True}})
    assert upgrade_commands.install_kind() == upgrade_commands.EDITABLE
    assert upgrade_commands.upgrade_command() is None


@pytest.fixture
def app(monkeypatch):
    cli = typer.Typer()
    upgrade_commands.register(cli)
    # A second command, so `upgrade` is a subcommand as it is on the real app
    # and not typer's one-command root.
    cli.command("noop")(lambda: None)
    monkeypatch.setattr(upgrade_commands, "_current_version", lambda: "0.0.3")
    monkeypatch.setattr(upgrade_commands, "fetch_latest_version", lambda client=None: "0.0.9")
    return cli


def test_check_says_what_is_out_and_how_to_get_it(app, monkeypatch):
    monkeypatch.setattr(upgrade_commands, "install_kind", lambda: upgrade_commands.PIP)
    result = CliRunner().invoke(app, ["upgrade", "--check"])
    assert result.exit_code == 0
    assert "0.0.3 -> 0.0.9" in result.output
    assert "pip install --upgrade opendde-harness" in result.output

    monkeypatch.setattr(upgrade_commands, "fetch_latest_version", lambda client=None: "0.0.3")
    result = CliRunner().invoke(app, ["upgrade", "--check"])
    assert "is up to date" in result.output


def test_upgrade_hands_the_process_to_the_installers_own_command(app, monkeypatch):
    """The updater replaces this very install, so it takes over the process."""
    handed: list = []
    monkeypatch.setattr(os, "execv", lambda path, argv: handed.append((path, argv)))
    monkeypatch.setattr(upgrade_commands.shutil, "which", lambda name: f"/usr/local/bin/{name}")

    monkeypatch.setattr(upgrade_commands, "install_kind", lambda: upgrade_commands.UV_TOOL)
    assert CliRunner().invoke(app, ["upgrade"]).exit_code == 0
    assert handed[-1] == ("/usr/local/bin/uv", ["/usr/local/bin/uv", "tool", "upgrade", "opendde-harness"])

    monkeypatch.setattr(upgrade_commands, "install_kind", lambda: upgrade_commands.PIP)
    assert CliRunner().invoke(app, ["upgrade"]).exit_code == 0
    python = upgrade_commands.sys.executable
    assert handed[-1] == (python, [python, "-m", "pip", "install", "--upgrade", "opendde-harness"])

    monkeypatch.setattr(upgrade_commands, "install_kind", lambda: upgrade_commands.EDITABLE)
    result = CliRunner().invoke(app, ["upgrade"])
    assert result.exit_code == 1
    assert "source checkout" in result.output
    assert len(handed) == 2


def test_a_pypi_that_cannot_be_asked_is_not_a_reason_to_reinstall(app, monkeypatch):
    def refuse(client=None):
        raise upgrade_commands.ReleaseLookupError("PyPI could not be asked for the latest release: offline")

    monkeypatch.setattr(upgrade_commands, "fetch_latest_version", refuse)
    result = CliRunner().invoke(app, ["upgrade"])
    assert result.exit_code == 1
    assert "offline" in result.output
    assert "reinstall" not in result.output
