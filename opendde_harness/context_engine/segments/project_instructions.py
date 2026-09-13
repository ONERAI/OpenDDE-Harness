"""Segment 3 — ``# Project instructions``. The user's AGENTS.md / ODH.md.

Sits directly after the bootstrap files because it is the same kind of thing:
instructions somebody wrote for the agent to follow, rather than material for
it to reason over. The difference is who wrote them — bootstrap files belong to
the workspace, these belong to the user and the repository they are working in
— which is why they are a segment of their own and not another entry in
``BOOTSTRAP_FILES``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opendde_harness.context_engine.base import AssemblyContext, Segment
from opendde_harness.context_engine.segments import render


class ProjectInstructionsSegmentBuilder:
    name = "project_instructions"
    order = 3

    def __init__(self, workspace: Path, cwd: Path | None = None) -> None:
        self._workspace = workspace
        # Pinned at construction when a caller supplies it; otherwise read per
        # turn, so a tool that changes directory mid-run is followed rather
        # than remembered wrongly.
        self._cwd = cwd

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        # The conversation's own view: what it has switched off, and what each
        # file held when it first saw it.
        files = render.project_instruction_files(self._workspace, self._cwd, ctx.session_key)
        text = render.render_project_instructions(files)
        meta: dict[str, Any] = {
            "project_instruction_files": [f.path for f in files if f.enabled and not f.skipped],
            # Named here so a turn that followed changed instructions can be
            # told apart from one that did not, after the fact.
            "project_instructions_changed": [f.path for f in files if f.changed],
        }

        return Segment(text=text, meta=meta) if text else Segment(text="", meta=meta)
