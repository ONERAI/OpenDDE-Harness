"""Real-stack tests; opt in with -m e2e.

TUI_E2E_ARTIFACTS retains captures outside pytest's rolling tmp directory.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "ui-tui" / "e2e" / "run.py"
SPEC = importlib.util.spec_from_file_location("tui_e2e_driver", PATH)
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("scenario", list(driver.SCENARIOS))
def test_tui_e2e(scenario, tmp_path):
    repo = Path(os.environ.get("TUI_E2E_REPO", str(driver.REPO)))
    artifacts = Path(os.environ.get("TUI_E2E_ARTIFACTS", str(tmp_path)))
    result = driver.run_scenario(scenario, artifacts=artifacts, repo=repo)
    print(f"{scenario}: {result['seconds']:.3f}s — {result['artifacts']}")
