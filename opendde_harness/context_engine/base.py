"""The contributor model every part of the turn context is built through.

One engine ships — :class:`~opendde_harness.context_engine.assembler.ContextAssembler` —
and it is a concrete class, not an implementation of an ABC: there was one
subclass, and the abstract layer only made the real contract harder to read.
AgentLoop holds one ``self.context_engine``, built by
:func:`~opendde_harness.context_engine.factory.build_context_engine`.

Naming note:
    Named ``context_engine`` (not ``context``) to mirror the L4
    ``memory_engine`` package and to avoid colliding with
    :mod:`opendde_harness.agent.context`, which hosts the lower-level
    :class:`ContextBuilder` holder.

Every part of the system prompt is produced by a :class:`SegmentBuilder`
(identity / bootstrap / project instructions / memory / active skills /
skills); they run concurrently and their ``text`` joins in ``order``. The
history is not a segment: it is chosen by the engine's one history selector
from the prefix those builders produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class TurnContext:
    """Per-turn inputs the loop hands the engine.

    Everything here is known only to the loop: the message being answered, the
    attachments, the conversation it belongs to, and the facts about the bound
    model and the configuration that the prompt is rendered against. They are
    passed rather than read from config inside rendering, so one turn's prompt
    can never be assembled from another turn's settings.
    """

    current_message: str
    media: list[str] | None = None
    channel: str | None = None
    chat_id: str | None = None
    # Whether this turn's model can see a picture. Decided by the loop (it owns
    # the provider and the model id) and carried here because the message is
    # built down in render, which knows neither. Defaults True so a caller that
    # does not set it keeps the old inline-everything behavior.
    can_see_images: bool = True
    # Name of a registered tool that can read an attachment the model cannot,
    # or None when none is (it comes from an optional plugin). Naming a tool the
    # model does not have reads as an instruction it cannot follow.
    describe_tool: str | None = None
    # The id the request actually reaches, told to the model so it never
    # guesses its own identity from pretraining. None omits the line.
    model: str | None = None
    # ``config.language``: "zh" adds the reply-language directive.
    language: str = "en"
    # Whether a long-term memory backend is configured, which decides whether
    # the workspace block may point the model at profile / episode paths.
    long_term_memory: bool = False
    # Tokens to hold back for the reply, from the model's own output ceiling.
    # The budget is the window less this.
    reserved_output: int = 0


@dataclass(frozen=True)
class AssembledPrefix:
    """The fixed part of one turn's prompt, as a value.

    Phase A's assembled system prefix + the user message + the tool
    definitions: everything the history selector must price *around* before it
    can decide what history fits. A per-turn argument rather than a field on a
    long-lived object, because two conversations assemble at once and one must
    never size itself against the other's prefix.
    """

    system_prefix: str
    user_message: dict[str, Any]
    tool_defs: list[dict[str, Any]]


@dataclass(frozen=True)
class AssemblyContext:
    """Per-turn, read-only inputs shared by every :class:`SegmentBuilder`."""

    session_key: str
    current_message: str
    media: list[str] | None
    channel: str | None
    chat_id: str | None
    session_messages: list[dict[str, Any]]
    can_see_images: bool = True
    describe_tool: str | None = None
    model: str | None = None
    language: str = "en"
    long_term_memory: bool = False


@dataclass
class Segment:
    """The uniform product of a :class:`SegmentBuilder`.

    - ``text`` — the segment's contribution to the **system** slot
      (joined by ``order``); ``""`` means "no segment this turn".
    - ``meta`` — merged into ``AssembledContext.metadata`` (e.g.
      ``injected_skill_ids`` / ``memory_hits``).
    """

    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SegmentBuilder(Protocol):
    """One contributor to the system prompt.

    ``order`` fixes the segment's position in the assembled prefix.
    """

    name: str
    order: int

    async def build(self, ctx: AssemblyContext) -> "Segment | None":
        """Return this turn's :class:`Segment`, or ``None`` to contribute nothing."""
        ...


__all__ = [
    "AssembledPrefix",
    "AssemblyContext",
    "Segment",
    "SegmentBuilder",
    "TurnContext",
]
