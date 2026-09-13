"""A newer OpenDDE Harness on PyPI is said, once a day, with the command that gets it here."""

from __future__ import annotations

import json
import time

import pytest

from opendde_harness.cli import update_notice, upgrade_commands


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A cache file of this test's own, and no refresh thread left from another."""
    path = tmp_path / "update_check.json"
    monkeypatch.setattr(update_notice, "_cache_path", lambda: path)
    monkeypatch.delenv(update_notice._OPT_OUT_ENV, raising=False)
    monkeypatch.setattr(update_notice, "_refreshing", None)
    return path


def _cached(path, latest, *, age=0.0):
    path.write_text(json.dumps({"checked_at": time.time() - age, "latest_version": latest}))


@pytest.mark.parametrize(
    ("kind", "command"),
    [
        (upgrade_commands.UV_TOOL, "uv tool upgrade opendde-harness"),
        (upgrade_commands.PIP, "pip install --upgrade opendde-harness"),
    ],
)
def test_a_newer_release_is_said_with_the_command_this_install_updates_by(cache, monkeypatch, kind, command):
    """Every install that has an update command is told, not only a uv tool:
    a plain pip install is the other way the package is distributed."""
    monkeypatch.setattr(upgrade_commands, "install_kind", lambda: kind)
    _cached(cache, "0.0.9")

    assert update_notice.update_notice("0.0.3") == ("0.0.9", command)
    assert update_notice.sentence("0.0.3", ("0.0.9", command)) == (
        f"OpenDDE Harness 0.0.9 is on PyPI; this is 0.0.3. Update with: {command}"
    )


def test_a_source_checkout_is_told_nothing(cache, monkeypatch):
    monkeypatch.setattr(upgrade_commands, "install_kind", lambda: upgrade_commands.EDITABLE)
    _cached(cache, "0.0.9")

    assert update_notice.update_notice("0.0.3") is None
    assert update_notice.maybe_refresh_async() is None


def test_nothing_is_said_when_up_to_date_opted_out_or_unreadable(cache, monkeypatch):
    monkeypatch.setattr(upgrade_commands, "install_kind", lambda: upgrade_commands.PIP)

    _cached(cache, "0.0.3")
    assert update_notice.update_notice("0.0.3") is None
    assert update_notice.update_notice("0.0.4") is None, "newer than PyPI is not behind it"
    # An rc of the release is that release, not something behind it.
    assert update_notice.update_notice("0.0.3rc1") is None

    _cached(cache, "0.0.9")
    monkeypatch.setenv(update_notice._OPT_OUT_ENV, "1")
    assert update_notice.update_notice("0.0.3") is None
    monkeypatch.delenv(update_notice._OPT_OUT_ENV)

    cache.write_text("[]")
    assert update_notice.update_notice("0.0.3") is None
    cache.unlink()
    assert update_notice.update_notice("0.0.3") is None


def test_the_first_launch_after_a_release_waits_a_moment_for_the_refresh(cache, monkeypatch):
    """A stale cache is refreshed in the background at launch; the notice
    built right after gives that refresh a moment rather than reading
    yesterday's answer, so the release is announced on the launch that found
    it and not the next one."""
    monkeypatch.setattr(upgrade_commands, "install_kind", lambda: upgrade_commands.PIP)
    monkeypatch.setattr(upgrade_commands, "fetch_latest_version", lambda: "0.0.9")
    _cached(cache, "0.0.3", age=2 * update_notice._REFRESH_TTL_SECONDS)

    assert update_notice.update_notice("0.0.3", wait=5.0) == ("0.0.9", "pip install --upgrade opendde-harness")
    assert json.loads(cache.read_text())["latest_version"] == "0.0.9"


def test_a_fresh_cache_is_not_refetched_and_a_failed_fetch_keeps_what_it_had(cache, monkeypatch):
    monkeypatch.setattr(upgrade_commands, "install_kind", lambda: upgrade_commands.PIP)
    fetched = []
    monkeypatch.setattr(upgrade_commands, "fetch_latest_version", lambda: fetched.append(1) or "0.0.9")

    _cached(cache, "0.0.5")
    assert update_notice.maybe_refresh_async() is None
    assert fetched == []

    def refuse():
        raise RuntimeError("offline")

    monkeypatch.setattr(upgrade_commands, "fetch_latest_version", refuse)
    _cached(cache, "0.0.5", age=2 * update_notice._REFRESH_TTL_SECONDS)
    update_notice.update_notice("0.0.3", wait=5.0)
    stamped = json.loads(cache.read_text())
    assert stamped["latest_version"] == "0.0.5", "the last good answer stays"
    assert time.time() - stamped["checked_at"] < 60, "and the failure backs off for a day"
