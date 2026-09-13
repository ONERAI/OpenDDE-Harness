"""The shipped MemoryBackend conformance suite, run against a backend in-tree.

``memory_engine/contract_test.py`` is what a plugin author subclasses to prove
their backend satisfies the host. It shipped with nothing in this repository
subclassing it, so the suite itself was never executed here and could rot
against the protocol it describes. This is the host's own conformant backend:
the smallest thing that satisfies :class:`MemoryBackend`, so a change to the
protocol or to the suite fails here first, before it reaches a plugin.
"""

from typing import Any

from opendde_harness.memory_engine import LifecycleContractTests, MemoryBackendContractTests
from opendde_harness.memory_engine.backend import Memory, MemoryBackend


class InMemoryBackend:
    """Substring recall over whatever was stored. No network, no files."""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.stored: list[tuple[str, list[dict[str, Any]], dict[str, Any] | None]] = []
        self.signals: list[dict[str, Any]] = []

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        top_k: int,
    ) -> list[Memory]:
        if (user_id is None) == (agent_id is None):
            return []  # neither or both set is a caller bug
        needle = query.lower()
        hits = [
            Memory(text=str(m.get("content", "")), score=1.0, metadata={"session_id": session})
            for session, messages, _ in self.stored
            for m in messages
            if needle in str(m.get("content", "")).lower()
        ]
        return hits[:top_k]

    async def store(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        self.stored.append((session_id, list(messages), metadata))
        return True

    async def feedback(self, signals: dict[str, Any]) -> None:
        self.signals.append(dict(signals))

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1


class TestInMemoryBackend(MemoryBackendContractTests, LifecycleContractTests):
    async def make_backend(self) -> MemoryBackend:
        return InMemoryBackend()
