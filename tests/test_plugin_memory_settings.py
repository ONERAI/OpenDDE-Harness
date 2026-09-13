"""The memory root, its settings file, and what doctor says about it.

The settings file is generated: the address a server listens on, and
placeholders in the model sections so the library builds a client -- the
client being this plugin's own, which runs on the conversation's model. No
role is configured by anyone, so nothing here asks for a key.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from opendde_harness.config import loader
from opendde_harness.plugin.memory.longterm import _health, _library, doctor, settings


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_current_config_path", tmp_path / "config.json")
    return tmp_path


def test_the_root_is_the_memory_dir_under_the_data_dir(data_dir: Path) -> None:
    assert settings.memory_root() == data_dir / "memory"
    assert settings.get_memory_config_path() == data_dir / "memory" / _library.CONFIG_FILENAME


def test_configure_memory_env_exports_the_one_root(data_dir: Path, monkeypatch) -> None:
    monkeypatch.setenv(_library.ROOT_ENV_VAR, "/nowhere")

    settings.configure_memory_env()

    assert Path(os.environ[_library.ROOT_ENV_VAR]) == data_dir / "memory"


def test_ensure_memory_home_writes_the_templates_and_the_placeholders(data_dir: Path) -> None:
    """A fresh root gets both template files and model sections the library can
    build from. Neither section holds a credential: the values name what serves
    them, and every reader of the file can see that."""
    assert not settings.memory_ready()

    settings.ensure_memory_home()

    root = data_dir / "memory"
    assert (root / _library.CONFIG_FILENAME).is_file()
    assert (root / _library.OME_CONFIG_FILENAME).is_file()
    toml = settings.load_memory_config()
    for section in settings.MODEL_SECTIONS:
        assert toml[section]["model"] == settings.PLACEHOLDER_MODEL
        assert toml[section]["api_key"] == settings.PLACEHOLDER_KEY
        assert toml[section]["base_url"] == settings.PLACEHOLDER_URL
    # The optional roles ship from the template with no key, and stay that way:
    # the library then boots without them and searches by keyword.
    assert not toml.get("embedding", {}).get("api_key")
    assert not toml.get("rerank", {}).get("api_key")
    assert settings.memory_ready()
    assert not (root / ".lock").exists() and not (root / f"{_library.CONFIG_FILENAME}.tmp").exists()


def test_a_value_somebody_put_in_a_model_section_is_replaced_on_the_next_boot(data_dir: Path) -> None:
    """The file is generated. A key typed into it would be read by nothing,
    and leaving it there would make the file look like a place to configure
    one."""
    settings.ensure_memory_home()
    path = settings.get_memory_config_path()
    text = path.read_text(encoding="utf-8").replace(settings.PLACEHOLDER_KEY, "sk-typed-by-hand", 1)
    path.write_text(text, encoding="utf-8")

    settings.ensure_memory_home()

    assert "sk-typed-by-hand" not in path.read_text(encoding="utf-8")


def test_the_declared_address_is_what_set_memory_api_wrote(data_dir: Path) -> None:
    settings.ensure_memory_home()
    # The template ships an address of its own; ours replaces it.
    assert settings.memory_declared_address() != "http://localhost:18791"

    settings.set_memory_api(host="localhost", port=18791)

    assert settings.memory_declared_address() == "http://localhost:18791"
    # Written beside the placeholders, not over them.
    assert settings.memory_ready()


def _config(backend: str | None, model: str = "openai/gpt-5") -> SimpleNamespace:
    return SimpleNamespace(
        memory=SimpleNamespace(backend=backend),
        agents=SimpleNamespace(defaults=SimpleNamespace(model=model)),
        plugins=SimpleNamespace(config={}),
    )


def test_doctor_reports_the_model_memory_runs_on_and_what_the_server_built(data_dir: Path, monkeypatch) -> None:
    settings.ensure_memory_home()
    monkeypatch.setattr(
        doctor,
        "probe_capabilities",
        lambda url: _health.CapabilityReport(reachable=True, capabilities={"llm": True, "multimodal_llm": False}),
    )

    info = doctor.probe_memory(_config("longterm"))

    assert info.model == "openai/gpt-5"
    assert info.address == "http://localhost:18791"
    assert info.server_running and info.reports_capabilities
    assert (info.llm_built, info.multimodal_built) == (True, False)
    assert not info.broken

    monkeypatch.setattr(
        doctor, "probe_capabilities", lambda url: _health.CapabilityReport(reachable=True, capabilities={"llm": False})
    )
    assert doctor.probe_memory(_config("longterm")).broken


def test_doctor_says_why_memory_on_disk_is_off(data_dir: Path) -> None:
    (data_dir / "memory").mkdir()
    assert doctor.probe_memory(_config(None)).disabled_reason == "the memory root has no settings file"

    settings.ensure_memory_home()
    assert doctor.probe_memory(_config(None)).disabled_reason == "turned off in config"
    assert doctor.probe_memory(_config("longterm")).disabled_reason is None


def test_a_start_failure_is_named_when_its_words_do_not_say_what_it_is():
    """The last line of a traceback is the library's words. An OSError about
    inotify instances is a system limit with a one-line fix, and nothing in
    those words says so, so the cause is named and the wizard prints the fix."""
    from opendde_harness.plugin.memory.longterm import _server

    exited = (
        "Long-term memory service exited with code 3 while starting at http://localhost:18791. "
        "OSError: [Errno 24] inotify instance limit reached Full log: /root/.opendde_harness/logs/memory-server.log"
    )

    assert _server.failure_cause(exited) == "inotify"
    assert _server.failure_cause("OSError: [Errno 24] too many open files") is None
    assert _server.failure_cause("Address already in use") is None
