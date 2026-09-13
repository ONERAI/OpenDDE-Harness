"""What ``ddeharness doctor`` says about long-term memory.

Local HTTP only, never raises: the probe talks to the memory service on
localhost, which spends no tokens and reaches no third party. Two facts are
kept apart on purpose -- what the root declares (backend on, root present)
and what a running server reports it built (``/health`` capabilities) --
because a server can answer 200 and still have failed to build its parser,
and that disagreement is the fault worth reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from opendde_harness.plugin.memory.longterm._health import (
    BACKEND_NAME,
    capability_available,
    configured_base_url,
    probe_capabilities,
)
from opendde_harness.plugin.memory.longterm.settings import memory_ready, memory_root


@dataclass
class MemoryInfo:
    """What the memory backend is, and what it can actually do."""

    backend: Optional[str] = None
    root: Optional[str] = None
    address: Optional[str] = None
    #: The model memory runs on: the conversation's default.
    model: Optional[str] = None
    server_running: bool = False
    reports_capabilities: bool = False
    #: ``/health``'s answer per section: built, not built, or not reported.
    llm_built: Optional[bool] = None
    multimodal_built: Optional[bool] = None
    #: Set when memory exists on disk but the runtime will not use it.
    disabled_reason: Optional[str] = None

    @property
    def broken(self) -> bool:
        """The server is up and could not build the one thing memory cannot work without."""
        return self.llm_built is False


def probe_memory(config: Any) -> MemoryInfo:
    """Ask the memory server what it can do."""
    backend = config.memory.backend
    info = MemoryInfo(backend=backend, model=str(config.agents.defaults.model or "") or None)
    if backend != BACKEND_NAME:
        # Memory can sit on disk and still be off: the wizard turned it off, or
        # a root was left without its settings file. Recall answers zero hits
        # forever and every other surface looks healthy, so this is the one
        # place that can say why.
        if backend is None and memory_root().is_dir():
            info.root = str(memory_root())
            info.disabled_reason = "turned off in config" if memory_ready() else "the memory root has no settings file"
        return info
    info.root = str(memory_root())
    info.address = configured_base_url(config)
    report = probe_capabilities(info.address)
    info.server_running = report.reachable
    info.reports_capabilities = report.reports_capabilities
    info.llm_built = capability_available(report.capabilities, "llm")
    info.multimodal_built = capability_available(report.capabilities, "multimodal")
    return info


def render_memory(console: Any, memory: MemoryInfo) -> None:
    """Print the memory block of the doctor report."""
    from opendde_harness.plugin.memory.longterm._server import server_log_path

    if memory.disabled_reason:
        console.print("\n[bold]Memory[/bold]")
        console.print(f"  Memories:   {memory.root}")
        console.print(f"  [yellow]Disabled:   {memory.disabled_reason}; recall returns nothing.[/yellow]")
        console.print("  [dim]Run ddeharness onboard to turn memory on.[/dim]")
        return
    if memory.backend != BACKEND_NAME:
        return
    console.print(f"  Memories:   {memory.root}")
    console.print(f"  Address:    {memory.address}")
    console.print(f"  Model:      {memory.model or '[yellow]no default model configured[/yellow]'}")
    console.print("  Retrieval:  keyword")
    if not memory.server_running:
        console.print("  Server:     [dim]not running  (starts on demand)[/dim]")
        return
    console.print("  Server:     [green]running[/green]")
    if not memory.reports_capabilities:
        console.print("  [dim]This server does not report capabilities.[/dim]")
        return
    for label, built in (("extraction", memory.llm_built), ("attachments", memory.multimodal_built)):
        state = (
            "[green]✓[/green]" if built else ("[dim]not reported[/dim]" if built is None else "[red]✗ not built[/red]")
        )
        console.print(f"  {label + ':':<12}{state}")
    if memory.broken:
        console.print(
            "\n  [yellow]⚠ The server could not build its model client; memory cannot work until this is fixed.[/yellow]"
        )
        console.print(f"  [dim]Check the server log: {server_log_path()}[/dim]")
    elif memory.multimodal_built is False:
        console.print(
            "\n  [yellow]⚠ Attachments (images, PDFs, audio) stay out of memory until the parser builds.[/yellow]"
        )
        console.print(f"  [dim]Check the server log: {server_log_path()}[/dim]")


__all__ = ["MemoryInfo", "probe_memory", "render_memory"]
