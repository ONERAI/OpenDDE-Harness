"""Context Management engine.

One engine — :class:`ContextAssembler` — assembled by
:func:`build_context_engine` from a flat list of :class:`SegmentBuilder`
plus one deterministic history selector (:class:`HistoryTrimmer`).
"""

from opendde_harness.context_engine.assembler import ContextAssembler
from opendde_harness.context_engine.base import (
    AssembledPrefix,
    AssemblyContext,
    Segment,
    SegmentBuilder,
    TurnContext,
)
from opendde_harness.context_engine.factory import build_context_engine
from opendde_harness.context_engine.history_trimmer import (
    ContextBudgetError,
    HistoryTrimmer,
    SelectionOutcome,
)
from opendde_harness.context_engine.types import AssembledContext, TokenBudget

__all__ = [
    "AssembledContext",
    "AssembledPrefix",
    "AssemblyContext",
    "ContextAssembler",
    "ContextBudgetError",
    "HistoryTrimmer",
    "Segment",
    "SegmentBuilder",
    "SelectionOutcome",
    "TokenBudget",
    "TurnContext",
    "build_context_engine",
]
