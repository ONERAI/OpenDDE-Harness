"""Drive the real TUI entrypoint under a pseudo-terminal against a synthetic peer.

The other harness in this directory launches the whole product. This one
launches only `src/entry.ts`, against a Unix socket that answers the JSON-RPC
frames however the scenario wants, because the cases it covers are about
startup failing *after* the renderer has taken the terminal — which the real
gateway has no reason to do.

What it asserts is terminal ownership, not UI content: entering the alternate
screen and turning autowrap off are the renderer's, and whatever happens next,
the process must give both back before it exits.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ENTER_ALT = b"\x1b[?1049h"
LEAVE_ALT = b"\x1b[?1049l"
WRAP_OFF = b"\x1b[?7l"
WRAP_ON = b"\x1b[?7h"
OSC11_QUERY = b"\x1b]11;?"
WHITE_REPLY = b"\x1b]11;rgb:ffff/ffff/ffff\x07"
# The lockup's violet, one per palette, at truecolor. The welcome block is the
# first thing painted, so whichever of these appears is the palette the app
# believed in when it painted.
DARK_PRIMARY = b"38;2;167;139;250"
LIGHT_PRIMARY = b"38;2;124;58;237"
WORDMARK = b"OpenDDE Harness"
DEFAULT_TIMEOUT = 30.0

BOOT_RESULTS = {
    "system.hello": {"server_version": "synthetic", "server_capabilities": [], "session": {}},
    "commands.catalog": {"pairs": [], "categories": []},
    "config.get": {"config": {}},
    "setup.status": {"provider_configured": True},
    "session.create": {"session_id": "tui:synthetic", "info": {"model": "synthetic", "cwd": "/nonexistent"}},
    "turn.subscribe": {"subscription_id": "sub-1"},
}


def peer(path: str, *, reject: bool = False, stall: str | None = None) -> socket.socket:
    """A gateway that answers boot frames, refuses them, or goes quiet on one."""
    server = socket.socket(socket.AF_UNIX)
    server.bind(path)
    server.listen(1)

    def serve():
        try:
            conn, _ = server.accept()
        except OSError:
            return
        with conn, conn.makefile("rb") as reader:
            for raw in reader:
                if not raw.startswith(b"{"):
                    continue
                frame = json.loads(raw)
                if frame.get("method") == stall or "id" not in frame:
                    continue
                if reject:
                    reply = {
                        "jsonrpc": "2.0",
                        "id": frame["id"],
                        "error": {"code": -32603, "message": "synthetic boot failure"},
                    }
                else:
                    reply = {"jsonrpc": "2.0", "id": frame["id"], "result": BOOT_RESULTS.get(frame["method"], {})}
                try:
                    conn.sendall((json.dumps(reply) + "\n").encode())
                except OSError:
                    break

    threading.Thread(target=serve, daemon=True).start()
    return server


def drive(name: str, *, reject: bool, stall: str | None, terminate_on: bytes | None, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="tui-boot-pty-", dir="/tmp") as room:
        home = Path(room) / "home"
        home.mkdir()
        sock = str(Path(room) / "gateway.sock")
        server = peer(sock, reject=reject, stall=stall)
        master, slave = pty.openpty()
        before = termios.tcgetattr(slave)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(home),
            "OPENDDE_HARNESS_HOME": str(home),
            "OPENDDE_HARNESS_RPC_SOCKET": sock,
            "OPENDDE_HARNESS_TUI_THEME": "dark",
            "TERM": "xterm-256color",
            "NO_COLOR": "1",
            "npm_config_offline": "true",
            "npm_config_update_notifier": "false",
        }
        # The tsx launcher directly, not through `npx`: the process we start has
        # to be the one a signal reaches, and a shim in between swallows it.
        child = subprocess.Popen(
            [str(PACKAGE / "node_modules" / ".bin" / "tsx"), "src/entry.ts"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            cwd=str(PACKAGE),
            start_new_session=True,
        )

        data = b""
        signalled = terminate_on is None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.05)[0]:
                try:
                    data += os.read(master, 65536)
                except OSError:
                    break
            if not signalled and terminate_on in data:
                child.send_signal(signal.SIGTERM)
                signalled = True
            if child.poll() is not None:
                while select.select([master], [], [], 0.05)[0]:
                    try:
                        data += os.read(master, 65536)
                    except OSError:
                        break
                break

        if child.poll() is None:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            child.wait()
            raise AssertionError(f"{name}: the process never exited\n{data[-3000:]!r}")

        after = termios.tcgetattr(slave)
        os.close(master)
        os.close(slave)
        server.close()

        return {
            "scenario": name,
            "code": child.returncode,
            "enter_alt": data.count(ENTER_ALT),
            "leave_alt": data.count(LEAVE_ALT),
            "wrap_off": data.count(WRAP_OFF),
            "wrap_on": data.count(WRAP_ON),
            "canonical_restored": bool(after[3] & termios.ICANON) == bool(before[3] & termios.ICANON),
            "echo_restored": bool(after[3] & termios.ECHO) == bool(before[3] & termios.ECHO),
            "tail": data[-3000:].decode("utf-8", "replace"),
        }


def check(result: dict) -> dict:
    """Whatever else happened, the terminal has to come back as it was lent."""
    if not result["enter_alt"]:
        raise AssertionError(f"{result['scenario']}: the renderer never took the alternate screen\n{result['tail']}")
    if result["leave_alt"] < result["enter_alt"]:
        raise AssertionError(f"{result['scenario']}: the alternate screen was not left\n{result['tail']}")
    if result["wrap_off"] and not result["wrap_on"]:
        raise AssertionError(
            f"{result['scenario']}: autowrap was turned off and never turned back on\n{result['tail']}"
        )
    for flag in ("canonical_restored", "echo_restored"):
        if not result[flag]:
            raise AssertionError(f"{result['scenario']}: {flag} is false\n{result['tail']}")
    return result


def boot_failure(timeout: float) -> dict:
    """The handshake is refused after the renderer already took the screen."""
    result = check(drive("boot_failure", reject=True, stall=None, terminate_on=None, timeout=timeout))
    if result["code"] != 3:
        raise AssertionError(f"boot_failure: expected exit 3, got {result['code']}\n{result['tail']}")
    return result


def signal_during_boot(timeout: float) -> dict:
    """SIGTERM lands while the handshake is still outstanding."""
    result = check(
        drive("signal_during_boot", reject=False, stall="system.hello", terminate_on=ENTER_ALT, timeout=timeout)
    )
    if result["code"] != 143:
        raise AssertionError(f"signal_during_boot: expected exit 143, got {result['code']}\n{result['tail']}")
    return result


def light_terminal(timeout: float) -> dict:
    """A light terminal that answers the background query exactly once.

    The question goes out once, the answer comes back once, and the welcome
    block has to be painted once — in the palette the answer names. Both halves
    matter: painting it twice is the flash the query was added to remove, and
    painting it once in the guessed palette is worse, because it stays wrong.
    """
    with tempfile.TemporaryDirectory(prefix="tui-boot-pty-", dir="/tmp") as room:
        home = Path(room) / "home"
        home.mkdir()
        sock = str(Path(room) / "gateway.sock")
        server = peer(sock, reject=False, stall=None)
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(home),
            "OPENDDE_HARNESS_HOME": str(home),
            "OPENDDE_HARNESS_RPC_SOCKET": sock,
            # No theme pin and no NO_COLOR: this scenario is about what the
            # terminal says, painted in colors that name a palette.
            "OPENDDE_HARNESS_TUI_COLOR": "truecolor",
            "OPENDDE_HARNESS_TUI_FULLSCREEN": "0",
            "TERM": "xterm-256color",
            "npm_config_offline": "true",
            "npm_config_update_notifier": "false",
        }
        child = subprocess.Popen(
            [str(PACKAGE / "node_modules" / ".bin" / "tsx"), "src/entry.ts"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            cwd=str(PACKAGE),
            start_new_session=True,
        )

        data = b""
        answered = 0
        painted_at: float | None = None
        started = time.monotonic()
        deadline = started + timeout
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.02)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                data += chunk
                if OSC11_QUERY in chunk and not answered:
                    os.write(master, WHITE_REPLY)
                    answered += 1
            if WORDMARK in data and painted_at is None:
                painted_at = time.monotonic()
            # A second paint would land within a frame or two of the first.
            if painted_at is not None and time.monotonic() - painted_at > 1.0:
                break

        child.send_signal(signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            child.wait()
        while select.select([master], [], [], 0.05)[0]:
            try:
                data += os.read(master, 65536)
            except OSError:
                break
        os.close(master)
        os.close(slave)
        server.close()

        result = {
            "scenario": "light_terminal",
            "code": child.returncode,
            "asked": data.count(OSC11_QUERY),
            "answered": answered,
            "welcome_paints": data.count(WORDMARK),
            "dark_paints": data.count(DARK_PRIMARY),
            "light_paints": data.count(LIGHT_PRIMARY),
            "tail": data[-3000:].decode("utf-8", "replace"),
        }

    if result["answered"] != 1:
        raise AssertionError(f"light_terminal: the query was never asked\n{result['tail']}")
    if result["welcome_paints"] != 1:
        raise AssertionError(f"light_terminal: the welcome was painted {result['welcome_paints']} times, not once")
    if result["light_paints"] == 0:
        raise AssertionError(f"light_terminal: nothing was painted in the light palette\n{result['tail']}")
    if result["dark_paints"] != 0:
        raise AssertionError(
            "light_terminal: the dark palette reached a light terminal "
            f"({result['dark_paints']} times)\n{result['tail']}"
        )
    return result


SCENARIOS = {"boot_failure": boot_failure, "signal_during_boot": signal_during_boot, "light_terminal": light_terminal}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    failed = 0
    for name in names:
        try:
            result = SCENARIOS[name](args.timeout)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name} {json.dumps({k: v for k, v in result.items() if k != 'tail'})}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
