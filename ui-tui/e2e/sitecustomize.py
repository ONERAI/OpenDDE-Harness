"""Opt-in test injection, loaded only when e2e/run.py prepends this directory.

No production module imports this file. An injection failure must stop Python,
not silently continue with a real provider (sitecustomize normally does that).
"""

import ipaddress
import json
import os
import sys
import time
from pathlib import Path


def journal(kind, **fields):
    path = os.environ["OPENDDE_HARNESS_E2E_JOURNAL"]
    data = json.dumps({"kind": kind, "pid": os.getpid(), "at": time.monotonic(), **fields}) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data.encode())
    finally:
        os.close(fd)


def install():
    def local(host):
        if host in (None, "", "localhost"):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def audit(event, args):
        host = None
        if event == "socket.getaddrinfo":
            host = args[0]
        elif event == "socket.connect" and isinstance(args[1], tuple):
            host = args[1][0]
        if host is not None and not local(host):
            journal("network_blocked", event=event)
            raise PermissionError("TUI e2e forbids non-loopback network access")

    sys.addaudithook(audit)

    from opendde_harness.providers import registry

    # Some CLI modules import the tuple by value; declare the fixture before
    # importing those consumers, so readiness and request routing agree.
    registry.PROVIDERS += (
        registry.ProviderSpec(
            name="scripted",
            keywords=("scripted",),
            display_name="Scripted E2E",
            default_model="scripted/e2e",
            billing="plan",  # There is no real price/provider behind the fixture.
            bypasses_litellm=True,
            vision_override=False,
            image_tool_result_override=False,
        ),
    )

    from scripted_provider import ScriptedProvider

    from opendde_harness.cli import _helpers, update_notice
    from opendde_harness.config import update_providers
    from opendde_harness.config.loader import set_config_path
    from opendde_harness.config.schema import ProviderConfig
    from opendde_harness.tui_rpc.dispatcher import Dispatcher
    from opendde_harness.tui_rpc.server import RpcServer

    set_config_path(Path(os.environ["OPENDDE_HARNESS_E2E_CONFIG"]))

    # The CLI's provider-list/readiness path only reflects schema fields or
    # known LiteLLM names. Supply the real section schema for this test-only
    # name; credential validation and the onboarding gate still run normally.
    schema_for = update_providers._provider_schema_cls

    def provider_schema(name, *, authoritative=True):
        return ProviderConfig if name == "scripted" else schema_for(name, authoritative=authoritative)

    update_providers._provider_schema_cls = provider_schema

    def make_provider(config):
        if config.get_provider_name() != "scripted":
            raise RuntimeError("TUI e2e only permits the scripted provider")
        _helpers.check_provider_credentials(config)
        return ScriptedProvider(config, journal)

    # Patch before commands.run imports consumers of the factory.
    _helpers.make_provider = make_provider
    update_notice.maybe_refresh_async = lambda: None

    dispatch = Dispatcher.dispatch
    send_frame = RpcServer.send_frame

    async def observed_dispatch(self, frame):
        journal("rpc_request", frame=frame)
        result = await dispatch(self, frame)
        journal("rpc_result", method=frame.get("method") if isinstance(frame, dict) else None, frame=result)
        return result

    async def observed_send(self, frame):
        await send_frame(self, frame)
        journal("rpc_sent", frame=frame)

    Dispatcher.dispatch = observed_dispatch
    RpcServer.send_frame = observed_send
    journal("injection_ready")


if os.environ.get("OPENDDE_HARNESS_E2E_SCRIPT"):
    try:
        install()
    except BaseException as exc:
        # Never log inherited environment values or credentials.
        sys.stderr.write(f"TUI e2e injection failed: {type(exc).__name__}: {exc}\n")
        os._exit(90)
