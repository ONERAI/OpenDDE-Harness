"""Startup failures under a real pseudo-terminal; opt in with -m e2e.

These launch `ui-tui/src/entry.ts` against a synthetic socket peer rather
than the whole product, because what they check is what happens when the boot
fails *after* the renderer has taken the terminal.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "ui-tui" / "e2e" / "boot_pty.py"
SPEC = importlib.util.spec_from_file_location("tui_boot_pty", PATH)
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("scenario", list(driver.SCENARIOS))
def test_tui_boot_pty(scenario):
    result = driver.SCENARIOS[scenario](driver.DEFAULT_TIMEOUT)
    # Scenarios report what they measured; only the exit code is common to all
    # of them, so print the rest of each result rather than assuming its keys.
    detail = ", ".join(f"{key} {value}" for key, value in sorted(result.items()) if key != "code")
    print(f"{scenario}: exit {result['code']}{', ' + detail if detail else ''}")
