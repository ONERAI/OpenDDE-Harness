"""The readable long-term memory files (user.md + episodes.md).

:class:`MemoryStore` is the single owner of both: the section-aware read the
prompt uses, the episode append, the one-section profile splice, and the
transactional discipline behind all of them -- one advisory lock, atomic writes
through a unique temp file, and a versioned state file whose damaged contents are
preserved rather than reset.

What writes through it is a :class:`~opendde_harness.memory_engine.backend.MemoryBackend`:
:class:`~opendde_harness.memory_engine.host_backend.HostMarkdownBackend` by
default, holding the annotation and profile-refresh prompts, or a plugin backend
that ignores these files entirely. The policy lives there; the files live here.
"""

from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore

__all__ = ["MemoryStore"]
