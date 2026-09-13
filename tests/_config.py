"""Building a :class:`Config` for a test, in the shape the file actually has.

Every suite that needs a configured provider used to write its own literal, so
the shape of the ``providers`` section was stated a few dozen times and each
statement had to be found and changed together. One builder here instead: a
test says which providers exist and which model is the default, and this is the
only place that knows how those two are spelled.

The spelling is pi's ``models.json``: keys are pi provider ids, a model id is
``"<provider>/<model>"``, and a provider pi does not ship needs ``baseUrl`` and
``api``. :func:`declared` writes that second kind so a test does not have to
remember the pairing rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opendde_harness.config.schema import Config

#: The default a test gets when it does not care which vendor answers: one key
#: under one of pi's own providers, serving the model the config names.
DEFAULT_MODEL = "anthropic/claude-sonnet-5"


def keyed(*providers: str, key: str = "test-key") -> dict[str, Any]:
    """``{provider: {"apiKey": key}}`` for each of pi's own providers named."""
    return {provider: {"apiKey": key} for provider in providers}


def declared(
    provider: str,
    *,
    base_url: str = "http://127.0.0.1:8000/v1",
    api: str = "openai-completions",
    models: list[Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """One entry for a provider this config declares, with the pairing satisfied.

    ``baseUrl`` and ``api`` together, because one without the other is refused;
    and at least one model, because pi has no catalogue for a declared provider
    and an entry naming none reaches the service as nothing at all.
    """
    return {
        provider: {
            "baseUrl": base_url,
            "api": api,
            "models": models if models is not None else ["test-model"],
            **fields,
        }
    }


def config_dict(
    providers: dict[str, Any] | None = None,
    model: str = DEFAULT_MODEL,
    **blocks: Any,
) -> dict[str, Any]:
    """The raw JSON a config file would hold for this provider set and model."""
    return {
        "agents": {"defaults": {"model": model}},
        "providers": providers if providers is not None else keyed(model.split("/", 1)[0]),
        **blocks,
    }


def config(
    providers: dict[str, Any] | None = None,
    model: str = DEFAULT_MODEL,
    **blocks: Any,
) -> Config:
    """A validated :class:`Config` for this provider set and default model.

    With no ``providers`` the default model's own provider is configured with a
    key, which is what almost every test wants: a config that routes.
    """
    return Config.model_validate(config_dict(providers, model, **blocks))


def write_config(
    path: Path,
    providers: dict[str, Any] | None = None,
    model: str = DEFAULT_MODEL,
    **blocks: Any,
) -> Path:
    """Write that config to ``path`` as the loader would read it back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_dict(providers, model, **blocks), indent=2), encoding="utf-8")
    return path


__all__ = ["DEFAULT_MODEL", "config", "config_dict", "declared", "keyed", "write_config"]
