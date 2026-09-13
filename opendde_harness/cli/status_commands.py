"""Top-level ``status`` command — show config / workspace / provider status."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console

from opendde_harness import __logo__

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opendde_harness.config.schema import ProviderEntry
    from opendde_harness.providers.auth import CredentialStatus

console = Console()


def _credential_line(status: CredentialStatus, entry: ProviderEntry) -> str:
    """What answers for this provider, as the green half of its status line.

    ``source`` is the part worth saying out loud: a key in the config, a key in
    the vendor's own environment variable and a sign-in in pi's credential store
    are all "configured", and somebody reading this to work out why a provider
    answers -- or why it answers with the wrong account -- needs to know which of
    them it was.
    """
    from opendde_harness.providers.auth import KIND_AMBIENT, KIND_DEVICE_FLOW

    if status.kind == KIND_DEVICE_FLOW:
        # Checked before the address, because an entry can carry both and the
        # sign-in is what reaches the vendor.
        return "[green]✓ (OAuth)[/green]"
    if status.kind == KIND_AMBIENT:
        # The credential is a whole environment chain (the AWS one, Google ADC);
        # there is no key to have a source.
        return "[green]✓ (the environment's own credentials)[/green]"
    if status.source == "declared":
        # For a provider this config declares, the address IS what makes it
        # reachable -- and the one field most worth seeing spelled out.
        return f"[green]✓ {entry.base_url}[/green]"
    if status.source == "environment":
        return "[green]✓ (key from the environment)[/green]"
    if status.source == "store":
        return "[green]✓ (key from the credential store)[/green]"
    return "[green]✓[/green]"


def register(app: typer.Typer) -> None:
    """Attach the ``status`` command to ``app``."""

    @app.command()
    def status():
        """Show OpenDDE Harness status."""
        from opendde_harness.config.loader import (
            ConfigReadError,
            get_config_path,
            load_config,
            read_raw_or_raise,
        )

        config_path = get_config_path()
        config = load_config()
        workspace = config.workspace_path

        console.print(f"{__logo__} OpenDDE Harness Status\n")

        # Three states, not two: a present-but-unparseable file used to show the
        # same checkmark as a healthy one, right before the provider walk below
        # aborted on it. The detailed remedy still comes from that walk.
        if not config_path.exists():
            config_state = "[red]missing[/red]"
        else:
            try:
                read_raw_or_raise(config_path)
            except ConfigReadError:
                config_state = "[red]invalid[/red]"
            else:
                config_state = "[green]✓[/green]"
        console.print(f"Config: {config_path} {config_state}")
        console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

        if config_path.exists():
            from opendde_harness.providers import pi_ids
            from opendde_harness.providers.auth import credential_status

            console.print(f"Model: {config.agents.defaults.model}")

            # The loaded config is the whole source: its `providers` section is
            # the configured set -- an entry is there because somebody wrote it --
            # and `credential_status` is what adds the material the file does not
            # hold, a key in the vendor's own environment variable or a sign-in in
            # pi's credential store. Reading the file alone reported both of those
            # as unset; deciding it here instead of asking `providers.auth` is what
            # once called Azure configured with a key and no address.
            # Only providers that answer get a row; the rest fold into one count,
            # so a single-provider setup is not buried under 20 'not set' rows.
            unconfigured = 0
            for provider, entry in config.providers.items():
                status_of = credential_status(provider, entry, include_external=True)
                if not status_of.ok:
                    unconfigured += 1
                    continue
                label = pi_ids.display_name(provider, entry.name)
                console.print(f"{label}: {_credential_line(status_of, entry)}")
            if unconfigured:
                console.print(
                    f"[dim]{unconfigured} providers not configured (ddeharness provider list to see all)[/dim]"
                )
            from opendde_harness.agent.tools.web import brave_key

            # One short row like the ones above it: the engine, and for the
            # keyless one where a key goes. Which providers search through
            # their own service is the troubleshooting guide's to explain.
            if brave_key(config.tools.web.brave_api_key):
                console.print("Web search: Brave Search API")
            else:
                console.print("Web search: DuckDuckGo [dim](no key; ddeharness onboard adds Brave)[/dim]")


__all__ = ["register"]
