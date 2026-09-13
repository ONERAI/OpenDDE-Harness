"""``cli.dispatch`` owns a command until its thread ends, not until the await does.

Nothing can interrupt the Typer app once ``_invoke_ec_cli`` is running on its
thread, so a timeout that unwound the redirections and released the lock would
hand the real stdout back to a live command and let the next dispatch share the
process-global console patch with it. These tests pin the opposite: the RPC
fails at once, the ownership stays behind until the command is actually gone,
and a later command waits for it no longer than its own timeout allows.
"""

from __future__ import annotations

import asyncio
import io
import sys
import threading
import time

import pytest

from opendde_harness.tui_rpc.errors import CliCommandTimeoutError
from opendde_harness.tui_rpc.methods import cli_dispatch as mod

#: Dispatch-compatible argv; the command itself is always monkeypatched away.
_ARGV = ["doctor"]
_WIDTH = 80
_MARKER = "runaway-command-wrote-this"


@pytest.fixture
def dispatch_lock(monkeypatch):
    """Give the test its own dispatch lock.

    ``asyncio.Lock`` binds to the loop of the first waiter that contends on it,
    and these tests contend on purpose; the module-level lock would then belong
    to whichever test ran first. Every test here waits for the lock to come back
    before it ends, so the reaper never releases a lock the test has dropped.
    """
    lock = asyncio.Lock()
    monkeypatch.setattr(mod, "_dispatch_lock", lock)
    return lock


async def _wait_until_released(lock: asyncio.Lock, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while lock.locked():
        assert time.monotonic() < deadline, "the dispatch lock was never released"
        await asyncio.sleep(0.01)


def _blocking_command(monkeypatch, *, exit_code: int = 0, on_release=None):
    """Install an ``_invoke_ec_cli`` whose first call blocks until released."""
    gate = threading.Event()
    started = threading.Event()
    finished = threading.Event()
    calls: list[list[str]] = []

    def fake_invoke(argv: list[str]) -> int:
        calls.append(list(argv))
        if len(calls) > 1:
            return 0
        started.set()
        assert gate.wait(10), "the blocked command was never released"
        if on_release is not None:
            on_release()
        finished.set()
        return exit_code

    monkeypatch.setattr(mod, "_invoke_ec_cli", fake_invoke)
    return gate, started, finished, calls


async def test_timeout_holds_the_lock_until_the_runaway_command_exits(monkeypatch, dispatch_lock):
    gate, started, finished, calls = _blocking_command(monkeypatch, exit_code=3)

    with pytest.raises(CliCommandTimeoutError) as raised:
        await mod.cli_dispatch({"argv": _ARGV, "width": _WIDTH, "timeout_s": 0.01})

    assert raised.value.data["started"] is True
    assert raised.value.data["timeout_s"] == 0.01
    assert started.wait(5)

    # The command still owns the dispatch, so the next one cannot start.
    assert dispatch_lock.locked() is True
    second = asyncio.ensure_future(mod.cli_dispatch({"argv": _ARGV, "width": _WIDTH, "timeout_s": 5}))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(second), 0.05)
    assert calls == [_ARGV], "a second command started while the first was still running"

    gate.set()
    assert finished.wait(5)
    result = await asyncio.wait_for(second, 5)

    assert result["exit_code"] == 0
    assert calls == [_ARGV, _ARGV]
    await _wait_until_released(dispatch_lock)


async def test_a_runaway_commands_output_never_reaches_the_real_stdout(monkeypatch, dispatch_lock, capfd):
    seen: dict[str, object] = {}

    def _write() -> None:
        seen["stdout"] = sys.stdout
        print(_MARKER)

    gate, started, finished, _calls = _blocking_command(monkeypatch, on_release=_write)

    with pytest.raises(CliCommandTimeoutError):
        await mod.cli_dispatch({"argv": _ARGV, "width": _WIDTH, "timeout_s": 0.01})
    assert started.wait(5)

    gate.set()
    assert finished.wait(5)

    # The redirection was still in place when the abandoned command printed.
    assert isinstance(seen["stdout"], io.StringIO)
    assert _MARKER in seen["stdout"].getvalue()
    assert _MARKER not in capfd.readouterr().out

    await _wait_until_released(dispatch_lock)


async def test_cancelling_the_caller_also_leaves_the_command_in_charge(monkeypatch, dispatch_lock):
    """A client that walks away ends the await, not the command."""
    gate, started, finished, _calls = _blocking_command(monkeypatch)

    call = asyncio.ensure_future(mod.cli_dispatch({"argv": _ARGV, "width": _WIDTH, "timeout_s": 5}))
    assert await asyncio.to_thread(started.wait, 5)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert dispatch_lock.locked() is True

    gate.set()
    assert finished.wait(5)
    await _wait_until_released(dispatch_lock)


async def test_the_normal_path_still_returns_the_commands_exit_code(monkeypatch, dispatch_lock, capfd):
    def fake_invoke(argv: list[str]) -> int:
        print(_MARKER)
        return 7

    monkeypatch.setattr(mod, "_invoke_ec_cli", fake_invoke)

    result = await mod.cli_dispatch({"argv": _ARGV, "width": _WIDTH, "timeout_s": 5})

    assert result["exit_code"] == 7
    assert _MARKER in result["stdout"]
    assert result["stderr"] == ""
    assert _MARKER not in capfd.readouterr().out
    assert dispatch_lock.locked() is False


async def test_a_later_command_gives_up_inside_its_own_timeout(monkeypatch, dispatch_lock):
    """Waiting on the lock is waiting on the runaway command, which has no deadline."""
    gate, started, finished, calls = _blocking_command(monkeypatch)

    with pytest.raises(CliCommandTimeoutError):
        await mod.cli_dispatch({"argv": _ARGV, "width": _WIDTH, "timeout_s": 0.01})
    assert started.wait(5)

    began = time.monotonic()
    with pytest.raises(CliCommandTimeoutError) as raised:
        # The outer bound is the "it never returns" guard, not the assertion:
        # a dispatch that waits on the lock without its own deadline fails here
        # with a TimeoutError instead of hanging the suite.
        await asyncio.wait_for(mod.cli_dispatch({"argv": ["status"], "width": _WIDTH, "timeout_s": 0.01}), 2.0)
    waited = time.monotonic() - began

    assert waited < 1.0, f"the queued command waited {waited:.2f}s on a 0.01s timeout"
    assert raised.value.data["started"] is False
    assert calls == [_ARGV], "the queued command must not run while the first owns the dispatch"

    # And the same command goes through once the runaway is actually gone.
    gate.set()
    assert finished.wait(5)
    await _wait_until_released(dispatch_lock)

    result = await mod.cli_dispatch({"argv": ["status"], "width": _WIDTH, "timeout_s": 5})

    assert result["exit_code"] == 0
    assert calls == [_ARGV, ["status"]]
