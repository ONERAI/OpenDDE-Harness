"""LocalSkillSource — wraps :class:`LocalPool` to emit :class:`RouterHit`.

Two ways out of the local pool, both deterministic:

- :meth:`full_catalogue` returns the complete pool when it is small
  enough to advertise whole, in a query-independent order. That order
  matters: the ``# Skills`` block sits in the cached system prefix, and a
  catalogue that reshuffles per message would invalidate the cache on
  every turn.
- :meth:`search` is the BM25 narrowing used once the pool is too large
  to list. :class:`LocalPool` already returns the cheap
  ``ScoredSkill(name, score, source)`` triple; the per-hit
  :class:`SkillRegistry` lookup that turns it into a hit is an O(1) dict
  access against the in-memory cache, not a fresh disk read.

Both paths drop a skill whose declared ``requires`` are unmet — the
agent cannot follow instructions whose binary or environment variable is
absent, so advertising it only buys a wasted turn. Hits whose name no
longer resolves in the registry (a file-watcher delete racing the BM25
snapshot) are skipped the same way. The router's contract is "at most k
hits", not "exactly k", so dropping is safer than emitting a hit the
agent cannot use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opendde_harness.memory_engine.skill_forge.types import RouterHit

if TYPE_CHECKING:
    from opendde_harness.memory_engine.skill_local.local_pool import LocalPool
    from opendde_harness.memory_engine.skill_local.registry import SkillRegistry
    from opendde_harness.memory_engine.skill_local.types import SkillMeta


class LocalSkillSource:
    """SkillSource adapter for the BM25 local pool.

    ``weight = 1.0`` makes Local the reference scale; Memory (0.9) is
    discounted because self-evolved skills are task-specific but
    unvalidated.
    """

    name: str = "local"
    weight: float = 1.0

    def __init__(
        self,
        pool: "LocalPool",
        registry: "SkillRegistry",
    ) -> None:
        self._pool = pool
        self._registry = registry

    async def search(
        self,
        query: str,
        history: list[dict[str, Any]],
        k: int,
    ) -> list[RouterHit]:
        # ``history`` is unused — local BM25 doesn't condition on prior
        # conversation. Future "smarter local ranker" could fold it in;
        # signature matches the Protocol so the seam stays.
        del history

        out: list[RouterHit] = []
        for h in self._pool.search(query, top_k=k):
            meta = self._registry.get(h.name, source=h.source)
            if meta is None or not self._usable(meta):
                continue
            out.append(self._to_hit(meta, h.score))

        return out

    def full_catalogue(self, limit: int) -> list[RouterHit] | None:
        """Every advertisable skill, or ``None`` when the pool is too large.

        ``always`` skills are excluded — the ``# Active Skills`` segment
        already lists them, and listing them twice spends prefix on a
        duplicate. Two skills sharing a display name are refused too: both
        would render the same ``local/<name>`` id, so the agent could not
        address either unambiguously and BM25 narrowing is the honest
        answer for that pool.
        """
        metas = [m for m in self._registry.list_all() if not m.always and self._usable(m)]
        if len(metas) > limit or len({m.name for m in metas}) != len(metas):
            return None
        # Query-independent order, so the rendered block is byte-identical
        # from turn to turn and the provider's prefix cache survives.
        metas.sort(key=lambda m: (m.name, m.source))
        return [self._to_hit(m, 0.0) for m in metas]

    def _usable(self, meta: "SkillMeta") -> bool:
        return self._registry.check_available(meta.name, source=meta.source)

    @staticmethod
    def _to_hit(meta: "SkillMeta", score: float) -> RouterHit:
        path = meta.path
        skill_dir = str(path.parent) if path is not None and not str(path).startswith("sqlite:") else None
        return RouterHit(
            qualified_id=f"local/{meta.name}",
            name=meta.name,
            content=meta.content,
            score=score,
            meta={
                "source": "local",
                "physical_source": meta.source,
                "always": meta.always,
                "skill_dir": skill_dir,
                "description": meta.description,
            },
        )


__all__ = ["LocalSkillSource"]
