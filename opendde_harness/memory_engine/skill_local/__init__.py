"""SkillForge primitives — the local pool: storage, retrieval, shared types.

File access, rendering and the blocklist live one level up, in
:class:`LocalSkillCatalog` under :mod:`opendde_harness.memory_engine.skill_forge`;
fusion across sources lives in :class:`SkillForgeRouter` beside it. Skill
feedback belongs to the :class:`MemoryBackend` plugin
(``backend.feedback`` / ``backend.store``), not here.

What this package is, the LOCAL-pool primitive layer:

- :class:`SkillRegistry` — workspace + builtin SKILL.md scanner
- :class:`LocalPool` — BM25 over the registry
- shared dataclasses (:class:`SkillMeta`, :class:`ScoredSkill`)
"""

from opendde_harness.memory_engine.skill_local.local_pool import LocalPool
from opendde_harness.memory_engine.skill_local.registry import SkillRegistry
from opendde_harness.memory_engine.skill_local.types import ScoredSkill, SkillMeta

__all__ = [
    # Data layer
    "SkillRegistry",
    "LocalPool",
    # Shared types
    "SkillMeta",
    "ScoredSkill",
]
