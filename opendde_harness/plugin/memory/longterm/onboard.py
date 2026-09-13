"""The wizard's long-term memory step: one question, then the service.

Memory runs on the conversation's default model through the model service,
so there is nothing to configure -- no model, no key, no address of its own.
The step asks whether to turn it on, writes the root's settings file, records
where the service listens, starts it, and says what it will run on. The
wizard's own console, translation and prompt helpers are used so the step
reads like the rest of the wizard; everything that knows the library lives
here and in the plugin's other modules, not in the wizard.
"""

from __future__ import annotations

import asyncio
import socket
import sys
from typing import Any
from urllib.parse import urlparse

import typer

from opendde_harness.cli import _chrome as chrome
from opendde_harness.cli import onboard_commands as oc
from opendde_harness.plugin.memory.longterm._health import BACKEND_NAME, PLUGIN_ID, configured_base_url
from opendde_harness.plugin.memory.longterm._server import DEFAULT_MEMORY_BASE_URL, _probe_health, ensure_memory_server
from opendde_harness.plugin.memory.longterm.settings import ensure_memory_home, memory_ready, memory_root


def memory_enabled() -> bool:
    """Whether long-term memory is on: the backend is named and its root is ready."""
    data = oc._load_raw_config()
    if (data.get("memory") or {}).get("backend") != BACKEND_NAME:
        return False
    return memory_ready()


def _set_backend(backend: str | None) -> None:
    from opendde_harness.config.update import set_memory_backend

    set_memory_backend(backend)


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def choose_address() -> str:
    """Where the service will listen.

    The default port, unless something else holds it: a service of ours
    already answering there is adopted, and anything else is stepped around
    with a free port rather than asked about -- the wizard's memory step
    asks one question, and a port is not it.
    """
    default = urlparse(DEFAULT_MEMORY_BASE_URL)
    port = int(default.port or 18791)
    if _port_is_free(port) or _probe_health(DEFAULT_MEMORY_BASE_URL):
        return DEFAULT_MEMORY_BASE_URL
    return f"http://{default.hostname}:{_free_port()}"


def _record_address(base_url: str) -> None:
    """The address in the program's config: a copy of what the root declares,
    so the runtime does not scan for a root on every session."""
    from opendde_harness.config.update import set_plugin_config_fields

    fields: dict[str, Any] = {"base_url": base_url}
    port = urlparse(base_url).port
    if port:
        fields["port"] = int(port)
    set_plugin_config_fields(PLUGIN_ID, fields)


def _default_model() -> str:
    return str(oc._load_current_default_model() or "")


def step(*, skip: bool, non_interactive: bool, warnings: list[str]) -> object:
    """Step 2 -- long-term memory. Returns ``None`` to advance."""
    del warnings
    oc._step_header(2, oc._t("Long-term memory", "长期记忆"))

    if sys.platform == "win32":
        oc.console.print(
            oc._t(
                "  [warn]⚠ The long-term memory engine does not support native Windows.[/warn]\n"
                "  [dim]Run OpenDDE Harness inside WSL for memory.[/dim]",
                "  [warn]⚠ 长期记忆引擎暂不支持 Windows 原生环境。[/warn]\n"
                "  [dim]在 WSL 中运行 OpenDDE Harness 可获得记忆支持。[/dim]",
            )
        )
        _set_backend(None)
        return None

    if skip:
        _set_backend(None)
        oc.console.print(
            oc._t(
                "  [dim]Long-term memory stays off. Run `ddeharness onboard` again to turn it on.[/dim]",
                "  [dim]长期记忆保持关闭。随时可以重新运行 ddeharness onboard 开启。[/dim]",
            )
        )
        return None

    oc.console.print(
        oc._t(
            "  Memory runs on your default model -- nothing else to configure.\n"
            "  [dim]Conversations are summarised into memories after each session and recalled by keyword.[/dim]",
            "  记忆直接使用你的默认模型，无需额外配置。\n  [dim]每次会话后对话会被整理成记忆，之后按关键词召回。[/dim]",
        )
    )
    if non_interactive:
        enabled = True
    else:
        from opendde_harness.cli import _choice

        enabled = _choice.row(
            oc._t("Turn long-term memory on?", "开启长期记忆？"),
            [(oc._t("on", "开启"), True), (oc._t("off", "不开启"), False)],
            default=True,
        )
        if enabled is None:
            raise typer.Exit(1)
    if not enabled:
        _set_backend(None)
        oc.console.print(
            oc._t(
                "  [dim]Long-term memory stays off. Run `ddeharness onboard` again to turn it on.[/dim]",
                "  [dim]长期记忆保持关闭。随时可以重新运行 ddeharness onboard 开启。[/dim]",
            )
        )
        return None

    root = memory_root()
    ensure_memory_home(root)
    _record_address(choose_address())
    _set_backend(BACKEND_NAME)
    oc.console.print(oc._t(f"  [dim]Memories: {root}[/dim]", f"  [dim]记忆目录：{root}[/dim]"))

    from opendde_harness.config.opendde_harness import load_opendde_harness_config

    base_url = configured_base_url(load_opendde_harness_config())
    while True:
        try:
            with chrome.working(oc.console, oc._t("Starting the memory service…", "正在启动记忆服务…")):
                asyncio.run(ensure_memory_server(base_url))
        except RuntimeError as exc:
            _explain_failure(exc)
            if non_interactive:
                # Kept on: the runtime starts the service again on the next session.
                return None
            if not _retry(exc):
                return None
            continue
        model = _default_model()
        oc.console.print(
            oc._t(
                f"  [ok]✓ Long-term memory is on, running on {model or 'your default model'}.[/ok]",
                f"  [ok]✓ 长期记忆已开启，运行在 {model or '默认模型'} 上。[/ok]",
            )
        )
        return None


def _explain_failure(exc: Exception) -> None:
    """Why the service did not start, in the operator's terms.

    The last line of a traceback is the library's words. An OSError about
    inotify instances is a system limit with a one-line fix, and nothing in
    those words says so -- so the cause is named (``_server.failure_cause``)
    and the fix is printed under it. What the library said stays, one line
    down, because it is what a search engine and a bug report want.
    """
    from opendde_harness.plugin.memory.longterm import _server

    oc.console.print(
        oc._t("  [error]✗ The memory service did not start[/error]", "  [error]✗ 记忆服务没有启动[/error]")
    )

    if _server.failure_cause(str(exc)) == "inotify":
        chrome.caption(
            oc.console,
            oc._t(
                "This machine has no inotify instances left, and the service needs one to watch the memory folder.",
                "这台机器的 inotify 实例已经用满，而记忆服务需要一个来监视记忆目录。",
            ),
        )
        chrome.caption(
            oc.console,
            oc._t(
                "Raise the limit and try again: sudo sysctl -w fs.inotify.max_user_instances=1024",
                "提高上限后重试：sudo sysctl -w fs.inotify.max_user_instances=1024",
            ),
        )
    chrome.hint(oc.console, str(exc))


def _retry(exc: Exception) -> bool:
    """After a failed start: retry, keep the settings for the next session, or turn memory off."""
    from opendde_harness.cli import _choice

    chrome.caption(
        oc.console,
        oc._t(
            "Leaving it for later keeps the settings; OpenDDE Harness tries again next start.",
            "暂时跳过会保留配置，下次启动 OpenDDE Harness 时再试。",
        ),
    )
    action = _choice.row(
        oc._t("What to do?", "怎么办？"),
        [
            (oc._t("retry", "重试"), "retry"),
            (oc._t("later", "暂时跳过"), "defer"),
            (oc._t("turn it off", "关闭记忆"), "disable"),
        ],
        default="retry",
    )
    if action is None:
        raise typer.Exit(1) from exc
    if action == "retry":
        return True
    if action == "defer":
        oc.console.print(
            oc._t(
                "  [warn]! Memory settings kept. OpenDDE Harness will try to start the service again next session.[/warn]",
                "  [warn]⚠ 已保留记忆配置。下次会话启动时 OpenDDE Harness 会再尝试启动服务。[/warn]",
            )
        )
        return False
    _set_backend(None)
    oc.console.print(
        oc._t(
            "  [warn]! Long-term memory turned off. Run `ddeharness onboard` again whenever you want it back.[/warn]",
            "  [warn]⚠ 已关闭长期记忆。随时可以重新运行 ddeharness onboard 开启。[/warn]",
        )
    )
    return False


__all__ = ["choose_address", "memory_enabled", "step"]
