"""Segment 6 — ``# Skills``. The catalogue the agent chooses from.

Advertising a skill is a deterministic decision, not a model call. Two
rules, one constant each:

1. **Small catalogue** (at most :data:`FULL_CATALOGUE_MAX` advertisable
   skills, and the pool is enumerable) — every skill is listed, with its
   qualified id and its one-line description, in a query-independent
   order. Nothing is left out, so nothing has to be recovered by a
   cleverer query.
2. **Large catalogue** — :class:`SkillForgeRouter` narrows to
   :data:`BM25_TOP_K` by BM25 over the user's message, fusing Local with
   the Memory source when one is wired.

Either way only names, descriptions and qualified ids reach the prompt.
The agent calls ``use_skill`` with an id to load that body and its
bundled resources, which keeps one authoritative loading path.

``build`` makes no LLM call. The query rewriter and the LLM gate that
used to run in front of every turn here are gone: on a curated catalogue
they spent the largest controllable slice of time-to-first-token to pick
two skills out of a dozen by name and description.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

from opendde_harness.context_engine.base import AssemblyContext, Segment
from opendde_harness.context_engine.segments import render
from opendde_harness.memory_engine.skill_forge.catalog import is_blocked, normalize_blocklist
from opendde_harness.tracing import semconv, trace

if TYPE_CHECKING:
    from collections.abc import Iterable

    from opendde_harness.memory_engine.skill_forge import SkillForgeRouter
    from opendde_harness.memory_engine.skill_forge.types import RouterHit

log = logging.getLogger(__name__)

#: Advertise the whole catalogue at or below this many skills. Above it the
#: block would cost more prefix than the choice is worth, so BM25 narrows
#: instead. Sized for a curated pool: the bundled builtins plus a plugin's
#: skills plus a workspace's own sit comfortably under it.
FULL_CATALOGUE_MAX = 40

#: How many skills a large catalogue advertises per turn.
BM25_TOP_K = 5


class SkillsSegmentBuilder:
    name = "skills"
    order = 6
    needs_prefix = False

    def __init__(
        self,
        router: "SkillForgeRouter | None",
        *,
        blocklist: "Iterable[str] | None" = None,
    ) -> None:
        self._router = router
        self._blocklist = normalize_blocklist(blocklist)

    @trace.instrument("skill.inject", kind="skill", detached=True, extract=semconv.skill_inject_skills)
    async def build(self, ctx: AssemblyContext) -> Segment | None:
        if self._router is None:
            return Segment(text="", meta=_meta([]))

        hits = self._router.full_catalogue(FULL_CATALOGUE_MAX)
        if hits is None:
            query = (ctx.current_message or "").strip()
            hits = (
                list(await self._router.select(query=query, history=ctx.session_messages, k=BM25_TOP_K))
                if query
                else []
            )

        # Blocklist — a hard drop across every source. The local pool hides
        # blocked skills from its own index; this is the backstop for the rest.
        if self._blocklist:
            kept: list["RouterHit"] = []
            for h in hits:
                if is_blocked(self._blocklist, h.name, h.meta.get("skill_id")):
                    log.warning("dropping blocklisted skill from the catalogue: %s", h.qualified_id)
                else:
                    kept.append(h)
            hits = kept

        body = render.render_router_skills(hits)
        return Segment(text=f"# Skills\n\n{body}" if body else "", meta=_meta(hits))


def _meta(hits: "list[RouterHit]") -> dict[str, Any]:
    return {
        "available_skill_ids": [h.qualified_id for h in hits if getattr(h, "qualified_id", None)],
        "skill_hits_by_source": dict(Counter((h.meta.get("source") or "?") for h in hits)),
    }
