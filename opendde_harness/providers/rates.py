"""What a resolved limit says, and what a model's published price is worth.

Two small things the rest of the tree agrees on and nothing else here decides.

:class:`Resolved` is the answer type for the window ladder: a number and the
tier that produced it, kept together so every surface that prints one can say
where it came from -- a window nothing measured used to print as "auto",
indistinguishable from a real one. :class:`ListRates` is the arithmetic on a
vendor's published rates, long-context tiers included.

The facts themselves come from the model service, which reports what pi's own
catalogue serves: ``PiModelProvider.context_window``, ``max_output_tokens``,
``input_modalities`` and ``list_rates`` are where the numbers are now read. The
vendored tables this module used to walk are gone, and with them the second
copy of every decision they held.

Unknown is answered as unknown. The window used to be answered with 65,536,
which was wrong for every model it applied to (the shipped default among them)
and drove the trimmer and the curator's slow path off a number nothing had
measured; callers treat an unknown window as "do not trim" and say so, the way
an unknown price reads as unknown rather than free. An unknown output ceiling is
no ceiling to check and no ceiling to send: on the OpenAI-shaped wires the
model service sends none and the server's default applies; on a wire that would
send the zero as written it refuses (``no_max_tokens``) rather than let pi
floor it at a token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: How a resolved number was arrived at, kept beside it so every surface that
#: prints one can say where it came from.
SOURCE_DECLARED = "declared"  # a row in providers.<id>.models, written by the user
SOURCE_SERVICE = "model-service"  # the row the model service reports for this model
SOURCE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class Resolved:
    """A number and the tier that answered it. ``tokens`` is None when nothing knows."""

    tokens: int | None
    source: str

    @property
    def known(self) -> bool:
        return self.tokens is not None and self.source != SOURCE_UNKNOWN


def is_plan_billed(model: str, providers: Any = None) -> bool:
    """Is this model's provider billed by subscription rather than per token?

    Asked wherever a dollar figure is about to be reported, because on a
    subscription there is no per-token figure to report -- not even zero, which
    reads as free.

    A sign-in is the arrangement: ``login: "oauth"`` on the provider's entry is
    a subscription (a ChatGPT account, a coding plan), and a key is metered.
    Read from the config rather than from a table of ours, because what a plan
    covers is the account and only the account's own entry says so. Without a
    providers map to ask, the answer is "metered" -- a figure of zero reported
    as free is the failure this guards, and claiming a plan nobody declared
    would suppress every real figure instead.
    """
    from opendde_harness.providers import model_id

    entry = providers.get(model_id.provider_of(model)) if providers is not None else None
    return getattr(entry, "login", None) == "oauth"


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ListRates:
    """A vendor's published price per token, USD, with the cache rates it names.

    ``tiers`` are the long-context prices: the highest tier whose threshold the
    request's whole input exceeds prices the whole request (pi's
    ``calculateCost``), so a call is priced by its own size and a session's
    cost is the sum of its calls, never a rate applied to a total.

    Built from a model service row's ``cost``, which states the same rates per
    million (:meth:`opendde_harness.providers.pi_provider.PiModelProvider.list_rates`).
    """

    input: float
    output: float
    cache_read: float
    cache_write: float
    tiers: tuple["ListTier", ...] = ()

    def cost(self, input_tokens: int, output_tokens: int, cache_read_tokens: int, cache_write_tokens: int) -> float:
        """What one call's tokens are worth at list. ``input_tokens`` is fresh only."""
        rates = self.for_input(input_tokens + cache_read_tokens + cache_write_tokens)
        return (
            input_tokens * rates.input
            + output_tokens * rates.output
            + cache_read_tokens * rates.cache_read
            + cache_write_tokens * rates.cache_write
        )

    def for_input(self, prompt_tokens: int) -> "ListRates":
        """The rates that price a request of this size, tiers included."""
        rates: ListRates = self
        for tier in self.tiers:
            if prompt_tokens > tier.input_tokens_above and tier.input_tokens_above >= rates.threshold:
                rates = tier
        return rates

    @property
    def threshold(self) -> int:
        return -1

    def per_million(self, field: str) -> str:
        """The rate as published, per million tokens: two decimals, three where two would hide a digit."""
        value = getattr(self, field) * 1e6
        return f"${value:,.2f}/M" if round(value, 2) == round(value, 3) else f"${value:,.3f}/M"


@dataclass(frozen=True)
class ListTier(ListRates):
    """The rates that apply once a request's input exceeds ``input_tokens_above``."""

    input_tokens_above: int = 0

    @property
    def threshold(self) -> int:
        return self.input_tokens_above


__all__ = [
    "SOURCE_DECLARED",
    "SOURCE_SERVICE",
    "SOURCE_UNKNOWN",
    "ListRates",
    "ListTier",
    "Resolved",
    "is_plan_billed",
]
