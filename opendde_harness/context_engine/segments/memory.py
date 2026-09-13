"""Segment 4 — ``# Memory``. The profile ⊕ the backend's recall(user).

The one composite segment: a single ``# Memory`` heading whose body merges the
slow-changing profile with the backend's query-conditioned recall hits.

Which of the two lanes carries the profile depends on who owns ``user.md``. A
plugin backend does not know the file exists, so the store reads it directly and
the backend's hits join it as a second lane. The host's own writer owns the file
on both sides -- it writes the sections and its ``recall`` returns them -- so it
answers for the profile alone, and reading the store as well would put the same
sections in the prompt twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opendde_harness.context_engine.base import AssemblyContext, Segment
from opendde_harness.context_engine.segments import render
from opendde_harness.tracing import semconv, trace

if TYPE_CHECKING:
    from opendde_harness.memory_engine.backend import MemoryBackend
    from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore


class MemorySegmentBuilder:
    name = "memory"
    order = 4

    def __init__(
        self,
        memory_store: "MemoryStore",
        backend: "MemoryBackend | None" = None,
        user_id: str = "default",
        memory_top_k: int = 5,
    ) -> None:
        self._memory_store = memory_store
        self._backend = backend
        self._user_id = user_id
        self._memory_top_k = memory_top_k
        #: Whether the backend is the owner of ``user.md`` rather than an index
        #: beside it. Asked of the backend, not configured here: a plugin that
        #: grows into owning the profile says so the same way.
        self._backend_owns_profile = bool(getattr(backend, "owns_profile", False))

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        # The recall propagates on hard failure so a backend outage surfaces at
        # AgentLoop rather than silently dropping memory.
        recall_hits = await self._recall(ctx.current_message)
        if self._backend_owns_profile:
            # The backend's own ``recall`` is the profile read; its hits are the
            # block ``get_memory_context`` renders, so they go in as they are
            # rather than as recall bullets.
            host = "\n\n".join(text for text in ((hit.text or "").strip() for hit in recall_hits) if text)
            recall_bullets = ""
        else:
            host = self._memory_store.get_memory_context(current_message=ctx.current_message)
            recall_bullets = render.render_recalled_memory(recall_hits)

        sections = [s for s in (host, recall_bullets) if s]
        meta: dict[str, Any] = {"memory_hits": len(recall_hits)}
        if not sections:
            return Segment(text="", meta=meta)
        return Segment(text="# Memory\n\n" + "\n\n".join(sections), meta=meta)

    @trace.instrument("memory.recall", extract=semconv.memory_recall)
    async def _recall(self, query: str) -> list[Any]:
        if self._backend is None:
            return []
        return list(
            await self._backend.recall(
                query=query,
                user_id=self._user_id,
                top_k=self._memory_top_k,
            )
        )
