"""``ddeharness status``: one short row per fact."""

from __future__ import annotations

import typer
from typer.testing import CliRunner

from opendde_harness.cli import status_commands
from tests._config import keyed, write_config


def _status(tmp_path, monkeypatch, **blocks):
    from opendde_harness.config import loader

    path = write_config(tmp_path / "config.json", keyed("deepseek"), model="deepseek/deepseek-v4-flash", **blocks)
    monkeypatch.setattr(loader, "_current_config_path", path)
    loader._cache.clear()
    app = typer.Typer()
    status_commands.register(app)
    app.command("noop")(lambda: None)
    return CliRunner().invoke(app, ["status"], terminal_width=120).output


def test_the_web_search_row_is_one_short_line_like_the_rows_above_it(tmp_path, monkeypatch):
    """It used to carry two parentheticals and wrap on an ordinary terminal;
    the rows above it are a label and a mark. The engine, and for the keyless
    one where a key goes, is all a status row has to say."""
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    out = _status(tmp_path, monkeypatch)
    row = next(line for line in out.splitlines() if line.startswith("Web search:"))

    assert row == "Web search: DuckDuckGo (no key; ddeharness onboard adds Brave)"
    assert len(row) <= 72
    assert "hosted search" not in out

    out = _status(tmp_path, monkeypatch, tools={"web": {"braveApiKey": "brave-0123456789"}})
    assert "Web search: Brave Search API" in out.splitlines()
