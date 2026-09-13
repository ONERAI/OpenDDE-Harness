"""Memory subsystems for the agent host.

Post-Phase-B layout:

- ``backend.py``        — :class:`MemoryBackend` Protocol + :class:`Memory`
  (the public plugin contract; what the bundled long-term memory backend and any
  third-party plugin implements).
- ``contract_test.py``  — base test class plugin authors inherit to
  verify their backend satisfies the host's expectations.
- ``consolidate/``      — :class:`MemoryStore`: ``user.md`` / ``episodes.md``
  as the prompt reads them and as the writer changes them, plus the
  transactional discipline behind them (one lock, atomic writes through a
  unique temp file, versioned state whose damaged contents are preserved).
- ``host_backend.py``   — :class:`HostMarkdownBackend`, the **default** backend:
  it annotates each completed turn into ``episodes.md`` and rewrites the
  ``user.md`` section behind a tag that has heated up.
- ``skill_local/``      — local-pool primitive layer:
  ``SkillRegistry``, ``LocalPool``, the SKILL.md watcher and shared types.
- ``skill_forge/``      — ``SkillForgeRouter`` + its sources (Local /
  Memory) plus :class:`LocalSkillCatalog`, the single owner of the local
  pool (rendering + feedback).

A workspace has exactly one owner of automatic durable extraction. With
``memory.backend`` unset that owner is :class:`HostMarkdownBackend`; naming a
plugin backend replaces it. Never both.
"""

from typing import TYPE_CHECKING

from opendde_harness.memory_engine.backend import Memory, MemoryBackend

if TYPE_CHECKING:
    from opendde_harness.memory_engine.contract_test import (
        LifecycleContractTests,
        MemoryBackendContractTests,
    )

__all__ = [
    "LifecycleContractTests",
    "Memory",
    "MemoryBackend",
    "MemoryBackendContractTests",
]


# The contract-test base classes live in ``contract_test``, which imports
# ``pytest`` (a dev-only dependency) at module top level. Importing them
# eagerly here would pull pytest into every ``import opendde_harness.memory_engine`` —
# breaking any production install without pytest (e.g. a packaged `ddeharness tui`),
# with ``ModuleNotFoundError: No module named 'pytest'``. Expose them lazily
# (PEP 562) so they resolve only when actually accessed — which happens under
# pytest in the test suite, where the import succeeds.
def __getattr__(name: str):
    if name in ("LifecycleContractTests", "MemoryBackendContractTests"):
        from opendde_harness.memory_engine import contract_test

        return getattr(contract_test, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
