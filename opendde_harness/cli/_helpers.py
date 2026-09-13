"""Shared CLI helpers used by multiple top-level command modules.

Extracted from commands.py so that per-command modules
(``doctor_commands.py``, ``tui_commands.py``, ...) can import them
directly instead of going through lazy wrappers.

Function names drop the leading underscore: the file itself is marked
internal with the ``_helpers`` prefix, so members do not also need the
private-name convention.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer
from rich.console import Console

from opendde_harness.config.schema import Config
from opendde_harness.providers import messages as pi_messages

console = Console()


DEFAULT_PROBE_MESSAGE = "Hi! Say hello in one sentence."


def check_provider_credentials(config: Config) -> None:
    """Fail-fast when the configured provider is missing required credentials.

    Cheap (nothing is constructed), so it can run at startup even when the real
    provider is built lazily.

    Raises ``MissingCredentialsError`` rather than printing and exiting: three entry
    points call this, and only one of them is a terminal. Each renders the
    failure in its own idiom -- the CLI as a red line and exit 1, the TUI as an
    RPC error carrying the same sentence.

    What counts as configured is `providers.auth`, the same declaration routing
    and `provider list` consult. Deciding it here as well is what produced three
    verdicts on one config: Azure with a key and no address was routed and
    displayed as configured yet rejected here.
    """
    from opendde_harness.providers import model_id
    from opendde_harness.providers.auth import MissingCredentialsError, credential_status

    model = config.agents.defaults.model
    # The id's own prefix names the provider whether or not it is configured,
    # so the report is about the provider the user named rather than nothing.
    provider_name = model_id.provider_of(model)
    if not provider_name:
        raise MissingCredentialsError(
            config.explain_unrouted(model),
            remedy="Or run `ddeharness onboard` for guided setup.",
        )

    status = credential_status(provider_name, config.providers.get(provider_name), include_external=True)
    if status.ok:
        return

    # A first run fails this check while naming a provider the user never chose:
    # with nothing configured, routing falls back to the schema's default model,
    # whose vendor then gets reported as the thing to go fix. Sending someone who
    # only has an OpenRouter key to `provider set anthropic` is the wrong errand,
    # so answer the wizard instead. Both halves are required -- a user who picked
    # this model, or who has some other provider working, gets the specific
    # verdict, which for the OAuth families names a sign-in rather than a key.
    chose_a_model = config.agents.defaults.model != type(config.agents.defaults)().model
    if not chose_a_model and not any(
        credential_status(name, entry, include_external=True).ok for name, entry in config.providers.items()
    ):
        raise MissingCredentialsError(
            "no provider is configured yet -- run `ddeharness onboard` for guided setup",
            remedy="Already have a key? ddeharness provider set <name> --api-key <key>",
        )

    raise MissingCredentialsError(
        status.summary,
        provider=provider_name,
        remedy="Run `ddeharness onboard` for guided setup.",
    )


def make_provider(config: Config):
    """The provider for the configured default model.

    One model layer: pi-ai in the Node model service, for every model and every
    provider. What an entry declares -- its address, its key or its sign-in, the
    protocol named by its ``api`` -- decides which pi adapter serves it, and
    there is nothing here to choose between.

    Constructing one is a constructor call: the service is started on the first
    request, not here, so this is cheap enough to run at startup.
    """
    from opendde_harness.providers import model_id, pi_ids
    from opendde_harness.providers.base import GenerationSettings
    from opendde_harness.providers.pi_provider import build_pi_provider

    model = config.agents.defaults.model
    provider_name = model_id.provider_of(model)
    # First, before the credential gate and before any driver is constructed:
    # the answer for a removed provider is the removal, not a missing key.
    pi_ids.refuse_removed(provider_name)

    check_provider_credentials(config)

    provider = build_pi_provider(config, model, provider_name or "")
    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        reasoning_effort=defaults.reasoning_effort,
        first_token_timeout=defaults.llm_first_token_timeout,
        idle_timeout=defaults.llm_idle_timeout,
        retries=defaults.llm_retries,
    )
    return provider


def declared_compaction_provider(config: Config) -> str:
    """What the provider ``make_provider`` would build writes into a marker.

    Named from the configuration alone, so the question can be answered before
    anything is built: the history selector asks it while budgeting the first
    prompt, and both a wrong yes and a wrong no cost the whole history. Empty
    for every backend that compacts nothing, which is all but the Codex login.

    The default model is the only one this answers for, and the only one it has
    to: a session on any other model is bound through ``ProviderPool``, which
    builds eagerly.
    """
    from opendde_harness.providers import model_id
    from opendde_harness.providers.pi_provider import pi_compaction_provider

    return pi_compaction_provider(model_id.provider_of(config.agents.defaults.model))


def send_probe(
    *,
    message: str = DEFAULT_PROBE_MESSAGE,
    # A reasoning model can spend half a minute on its first token, and a
    # gateway in front of one adds to that. Fifteen seconds failed setups whose
    # only fault was being slow.
    timeout_s: int = 60,
    max_tokens: int = 200,
) -> tuple[str, int | None, float]:
    """Build provider from current config and exchange one chat message.

    Shared by ``onboard`` Step 3 and ``doctor --probe``. Bypasses the full
    ``AgentLoop`` so the probe only proves the provider answers, not that
    the agent runtime is healthy.

    Returns ``(response_text, tokens_used, elapsed_s)``. Raises ``RuntimeError``
    on provider error, ``asyncio.TimeoutError`` on timeout, or whatever
    ``load_config`` / ``make_provider`` raise on config failure.
    """
    from opendde_harness.config.loader import load_config

    config = load_config()
    provider = make_provider(config)

    # A probe proves the provider answers. It is not the place to exercise
    # thinking, and asking for it here made the check impossible on a model
    # whose thinking has a floor: a budget profile needs at least 1024 tokens
    # and this cap is 200, so the adapter refused before any request was built
    # and the wizard reported a working provider as broken. ``off`` explicitly
    # -- stated rather than left to the model's own default, which an overlay
    # may have set to medium -- and the cap stays small.
    #
    # No temperature either, for the same reason one turn away: a number pinned
    # here would go out to whatever model is configured, and a model that does
    # not take the parameter would refuse the probe over it. Whatever the
    # model's own row asks for is what it gets.
    from opendde_harness.providers.pi_service import run_then_shutdown

    start = time.monotonic()
    # The service is ended on the loop that owns its pipes; see run_then_shutdown.
    response = run_then_shutdown(
        asyncio.wait_for(
            provider.chat_with_retry(
                messages=[pi_messages.user_message(message)],
                max_tokens=max_tokens,
                reasoning_effort="off",
            ),
            timeout=timeout_s,
        )
    )
    elapsed = time.monotonic() - start

    if response.finish_reason == "error":
        raise RuntimeError(response.content or "provider returned an error")

    usage = response.usage or {}
    tokens = usage.get("total_tokens") or usage.get("completion_tokens")
    return (response.content or "").strip(), tokens, elapsed


def print_probe_troubleshooting(provider: str | None) -> None:
    """Common-case hints when a probe fails.

    Shared by ``onboard`` Step 3 and ``doctor --probe`` so the diagnostic
    advice stays in one place.
    """
    console.print("\n  [dim]Troubleshooting:[/dim]")
    if provider:
        console.print(
            f"  [dim]·[/dim] [cyan]ddeharness provider test {provider}[/cyan] — re-check credentials without spending tokens"
        )
        console.print(
            f"  [dim]·[/dim] [cyan]ddeharness provider get {provider}[/cyan] — inspect what's actually stored on disk"
        )
    console.print(
        "  [dim]·[/dim] Check the model id in [cyan]~/.opendde_harness/config.json[/cyan] "
        "under [cyan]agents.defaults.model[/cyan] — it should match a model the "
        "provider serves."
    )


def load_runtime_config(config: str | None = None, workspace: str | None = None) -> Config:
    """Load config and optionally override the active workspace."""
    from opendde_harness.config.loader import load_config, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            raise typer.Exit(1)
        set_config_path(config_path)
        Console(stderr=True).print(f"[dim]Using config: {config_path}[/dim]")

    loaded = load_config(config_path)
    if workspace:
        loaded.agents.defaults.workspace = workspace
    return loaded


__all__ = [
    "DEFAULT_PROBE_MESSAGE",
    "make_provider",
    "send_probe",
    "print_probe_troubleshooting",
    "load_runtime_config",
]
