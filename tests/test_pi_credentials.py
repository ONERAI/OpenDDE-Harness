"""pi's credential store as this project reads it.

Read only: the store is written by the model service (``configure`` lands the
keys, a login lands its grant, a refresh rewrites it), and the Python side
never writes it, so what is tested here is the read. Everything is synthetic
and lands in ``tmp_path``: no real credential directory is read, and no
environment key is allowed to answer for one.

Every provider is named by its pi id, here and in the store, because that is
the only name it has: the translation this suite used to assert
(``pi_provider_id``) existed to reconcile our own section names with pi's ids,
and the config is keyed by pi's ids now.
"""

from __future__ import annotations

import json
import os

import pytest

from opendde_harness.providers.pi_credentials import stored_kind, stored_oauth
from tests._config import config as build_config


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch):
    """No vendor key in the environment answers for an entry in these tests."""
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_AUTH_TOKEN")) or name.startswith("CHATGPT_"):
            monkeypatch.delenv(name, raising=False)


def grant(access: str = "pi-wrote-this") -> dict:
    """An OAuth entry in pi's own shape, as its login writes one."""
    return {"type": "oauth", "access": access, "refresh": "r", "expires": 1_900_000_000_000}


# ---------------------------------------------------------------------------
# Reading the store
# ---------------------------------------------------------------------------


def test_a_stored_sign_in_is_what_oauth_presence_means(tmp_path):
    store = tmp_path / "pi-credentials.json"
    store.write_text(json.dumps({"openai-codex": grant(), "anthropic": {"type": "api_key", "key": "k"}}))

    assert stored_oauth("openai-codex", store_path=store) is True
    assert stored_kind("anthropic", store_path=store) == "api_key"
    assert stored_oauth("anthropic", store_path=store) is False
    assert stored_kind("google", store_path=store) == ""


def test_a_damaged_store_is_not_reported_as_never_signed_in(tmp_path):
    """Present and unreadable is a different answer, and a different fix.

    Reporting it as "signed out" told someone whose store was damaged to run a
    sign-in that would then fail on the same file.
    """
    store = tmp_path / "pi-credentials.json"
    store.write_text("{ not json")

    assert stored_kind("openai-codex", store_path=store) == "invalid"
    assert stored_oauth("openai-codex", store_path=store) is False


def test_no_store_at_all_reads_as_nothing_stored(tmp_path):
    assert stored_kind("openai-codex", store_path=tmp_path / "absent.json") == ""


def test_the_gate_reads_the_store_the_service_writes(tmp_path, monkeypatch):
    """``providers.auth`` asks this file and nothing else about a sign-in.

    Asked of an entry that names the sign-in, because that is what a sign-in
    leaves behind in the config (``login: "oauth"``): the grant itself is in the
    store and never in the file, so the entry says which of the two kinds this
    provider is and the store says whether it is there.
    """
    from opendde_harness.providers import auth

    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    signs_in = build_config({"openai-codex": {"login": "oauth"}}, model="openai-codex/gpt-5.6-luna")
    entry = signs_in.providers.get("openai-codex")

    assert auth.has_credentials("openai-codex") is False
    assert auth.credential_status("openai-codex", entry, include_external=True).ok is False

    (tmp_path / "pi-credentials.json").write_text(json.dumps({"openai-codex": grant()}))
    assert auth.has_credentials("openai-codex") is True
    assert auth.credential_status("openai-codex", entry, include_external=True).ok is True
    # And with no entry at all there is nothing to report on: an entry is there
    # because somebody wrote it, so its absence is the answer rather than a
    # reason to go looking in the store.
    assert auth.credential_status("openai-codex", None, include_external=True).ok is False
