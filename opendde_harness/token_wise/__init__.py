"""TokenWise — token accounting around LLM calls.

Public API:
    - ``StrategyRegistry``  — chains TokenStrategy hooks around LLM calls.
    - ``UsageTracker``      — strategy: records tokens + cost per call.

What a call cost is not decided here. The model layer prices each call from the
catalogue that served it, and the agent loop records that figure on the snapshot
it hands the tracker (``AgentLoop._build_usage_snapshot``); a second formula
over a second rate table is what this package used to carry, and the two drifted.

This package depends on nothing above it; callers assemble the registry.
"""

from opendde_harness.token_wise.registry import StrategyRegistry
from opendde_harness.token_wise.usage_tracker import UsageTracker

__all__ = [
    "StrategyRegistry",
    "UsageTracker",
]
