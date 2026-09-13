"""The compute container runs the harness's compute server with the image's own
Python, and the image ships the compute dependencies only (docker/environment.json).
A host-side dependency reached from the server's import closure -- ``loguru``
through ``tracing.semconv`` -> ``token_wise`` was the one that did it -- makes the
container exit on start, and the launcher then reports only "connection refused".
The closure is gated here against the image's package list.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from opendde_harness.cli.compute_environment import load_environment

REPO = Path(__file__).resolve().parents[1]

_GATE = r"""
import importlib.abc, importlib.metadata, json, sys

allowed = set(json.loads(sys.argv[1]))
dists = importlib.metadata.packages_distributions()
canon = lambda name: name.lower().replace("_", "-")


class OnlyTheImage(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        top = name.partition(".")[0]
        if top in sys.stdlib_module_names or top == "opendde_harness" or top.startswith("_"):
            return None
        if any(canon(dist) in allowed for dist in dists.get(top, ())):
            return None
        raise ModuleNotFoundError(f"{name} is not shipped in the compute image (distribution {dists.get(top)})")


sys.meta_path.insert(0, OnlyTheImage())
import opendde_harness.plugin.protein_design.servers.api  # noqa: E402,F401
"""


def test_the_compute_server_imports_only_what_the_image_ships() -> None:
    packages = [name.lower().replace("_", "-") for name in load_environment()["packages"]]
    run = subprocess.run(
        [sys.executable, "-c", _GATE, json.dumps(packages)],
        cwd=REPO,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr[-2000:]
