"""The detached design worker must import the installed package, not the launcher's cwd."""

from __future__ import annotations

import sys

from opendde_harness.plugin.protein_design.core.detached import DetachedDesignTaskController


def test_the_worker_never_puts_the_launchers_cwd_on_its_path(tmp_path):
    """A TUI started inside another checkout once made the worker import that
    checkout's older package (``python -m`` prepends the cwd) and refuse the
    workflow the current CLI had written. ``-P`` is what prevents it."""
    controller = DetachedDesignTaskController({}, python_executable=sys.executable)
    argv = controller.worker_argv("task-1")
    assert argv[:3] == [sys.executable, "-P", "-m"]
    assert "opendde_harness.plugin.protein_design.servers.worker" in argv
    assert argv[argv.index("--task-id") + 1] == "task-1"
