"""Which model each session runs on, and how large that model's window is.

Owns the per-session binding map, the restore-from-record path behind it, the
notice a failed restore leaves for the session's next turn, and the window
ladder for any model the loop asks about. Nothing here runs a turn: the loop
enters a binding for the turn's tree and reads the answers back through here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from loguru import logger

from opendde_harness.providers.base import LLMProvider, declared_tokens
from opendde_harness.providers.binding import ModelBinding, active_binding, active_window, set_active_window
from opendde_harness.providers.model_id import row_for
from opendde_harness.providers.rates import SOURCE_DECLARED, SOURCE_SERVICE, SOURCE_UNKNOWN, Resolved

if TYPE_CHECKING:
    from opendde_harness.config.schema import ProvidersConfig
    from opendde_harness.providers.pool import ProviderPool
    from opendde_harness.session.manager import SessionManager


def _noop(*_args: object) -> None:
    return None


class SessionBindings:
    """The loop's model identity: one default, one override per session."""

    def __init__(
        self,
        default: ModelBinding,
        *,
        providers: "ProvidersConfig",
        sessions: "SessionManager",
        provider_pool: "ProviderPool | None" = None,
    ) -> None:
        self._default = default
        self._providers = providers
        self._sessions = sessions
        self._pool = provider_pool
        self._by_session: dict[str, ModelBinding] = {}
        # Keys whose session record has been consulted for a stored model, hit
        # or miss. See ``_restore_once``.
        self._restore_attempted: set[str] = set()
        #: One message per session, waiting for that session's next turn to
        #: show it. Written by ``_restore_once`` when a stored choice could not
        #: be bound; read once and dropped.
        self._notices: dict[str, str] = {}
        #: Models whose window has been reported to the log, known or not.
        self._noted: set[str] = set()
        #: Where the out-of-turn fallback window goes besides here: the context
        #: engine sized itself against it at construction, which happens after
        #: this object exists, so the loop assigns this once the engine is built.
        self.on_default_window: Callable[[int | None], None] = _noop
        #: Called whenever any binding changes, so the loop can drop the
        #: capability verdicts a new provider may answer differently.
        self.on_rebind: Callable[[], None] = _noop
        # The default binding's window, for work that runs outside a turn; a
        # turn resolves its own at entry. None means unknown -- no row lists the
        # model and the user declared nothing -- and every consumer treats that
        # as "do not trim" rather than as a number (see TokenBudget).
        self._default_window: int | None = self.resolve_window().tokens

    # ── identity ────────────────────────────────────────────────────────

    @property
    def default(self) -> ModelBinding:
        """What a session with no switch of its own runs on."""
        return self._default

    @property
    def provider_pool(self) -> "ProviderPool | None":
        """Where a model id becomes a model id plus the credential for it."""
        return self._pool

    @property
    def provider(self) -> LLMProvider:
        """The provider of the binding the running turn entered, else the default's."""
        binding = active_binding()
        return binding.provider if binding is not None else self._default.provider

    @property
    def model(self) -> str:
        """The model id of the binding the running turn entered."""
        binding = active_binding()
        return binding.model if binding is not None else self._default.model

    @property
    def window(self) -> int | None:
        """How much the running turn's model can hold; None is unknown."""
        return active_window(self._default_window)

    def for_session(self, session_key: str) -> ModelBinding:
        """The binding this session runs on: its own switch, else the default.

        A new session has no entry, so it starts on the configured default
        rather than on whatever the last session switched to.

        A session whose choice is on disk but not yet in memory is restored
        here, on first ask. That is what makes the choice outlive a restart on
        every surface: the overrides live in this process, the session record is
        the only place they survive, and hanging the read off a TUI-only resume
        call would bring a conversation on a channel back on the default with
        its choice sitting unread in its own record.
        """
        binding = self._by_session.get(session_key)
        if binding is not None:
            return binding
        self._restore_once(session_key)
        return self._by_session.get(session_key, self._default)

    def model_for(self, session_key: str) -> str:
        """What to show this session's user, which is not the global default.

        A session whose stored choice could not be bound shows the default,
        which is what its turns now run on.
        """
        return self.for_session(session_key).model

    def has(self, session_key: str) -> bool:
        """Did this session switch, or is it just following the default?

        ``model_for`` cannot answer that -- it falls back to the default, so it
        never returns None. Callers that must distinguish "chose this" from
        "inherited this" ask here. Restores first, so a session that switched
        before a restart answers yes.
        """
        if session_key not in self._by_session:
            self._restore_once(session_key)
        return session_key in self._by_session

    def set(self, session_key: str, binding: ModelBinding) -> None:
        """Switch one session, leaving every other session where it was.

        Applied immediately and still safe mid-turn: a turn resolves its binding
        once at ``run_turn`` entry and holds it in a context var for its whole
        tree, including anything it detaches. So a switch during a turn cannot
        move that turn -- it lands on the next one -- and no parking is needed.
        """
        self._by_session[session_key] = binding
        self.on_rebind()

    def clear(self, session_key: str) -> None:
        """Drop a session's override so it follows the default again.

        Deliberately does not mark the key as consulted. The one caller is
        ``session.delete``, which unlinks the record before this runs, so there
        is nothing left for a later ask to read back in -- and a session that
        somehow kept its record is better served by re-reading it.
        """
        self._by_session.pop(session_key, None)

    def set_default(self, binding: ModelBinding) -> None:
        """Change what new sessions start on; switched sessions keep their own."""
        logger.info("default model is now {}", binding.model)
        self._default = binding
        self.on_rebind()

    def live_providers(self) -> list[LLMProvider]:
        """Every provider a turn can run on right now: the default's and each
        switched session's, once each. For a setting applied in place (a row
        declared in session) that must reach the provider the next turn will
        actually call, not only the default's."""
        out: list[LLMProvider] = []
        seen: set[int] = set()
        for binding in (self._default, *self._by_session.values()):
            if id(binding.provider) not in seen:
                seen.add(id(binding.provider))
                out.append(binding.provider)
        return out

    def take_notice(self, session_key: str) -> str | None:
        """The one-off message a failed restore left for this session."""
        return self._notices.pop(session_key, None)

    def _restore_once(self, session_key: str) -> None:
        """Read this session's stored model, at most once per key per process.

        The negative answer is remembered too. Most sessions never switched, and
        without that this would re-read a record on every turn to learn the same
        nothing.
        """
        if session_key in self._restore_attempted:
            return
        self._restore_attempted.add(session_key)
        if self._pool is None:
            return
        try:
            record = self._sessions.peek(session_key)
        except Exception as exc:
            logger.debug("cannot read session {!r} to restore its model: {}", session_key, exc)
            return
        metadata = getattr(record, "metadata", None) or {}
        model = metadata.get("model")
        if not model:
            return
        try:
            self.set(session_key, self._pool.bind(model, metadata.get("provider")))
        except Exception as exc:
            # Broad on purpose. Building a provider imports a vendor module and
            # checks credentials, so the failures reachable here are open-ended.
            # A resume that lands on the default is a worse session; a resume
            # that raises is no session at all. The user is told once, on this
            # session's next turn, so the substitution is not silent.
            logger.warning("session {!r} cannot resume on {!r} ({}); using the default", session_key, model, exc)
            self._notices[session_key] = (
                f"Could not restore model {model} for this session ({exc}); "
                f"running on {self._default.model}. "
                f"Use /model <provider>/<model> to choose again."
            )

    # ── window ──────────────────────────────────────────────────────────

    def resolve_window(self, model: str | None = None, binding: "ModelBinding | None" = None) -> Resolved:
        """Walk the window ladder for ``model`` (default: the running turn's).

        The model's own declared row is supplied from here so every caller --
        construction, a ``/model`` switch, the per-turn usage report, the session
        banner -- answers the same question the same way. Neither tier reaches
        the network: the row is configuration, and the provider is repeating a
        listing it already read.

        Two tiers and then unknown. The user's own declaration for this model
        (its row in ``providers.<id>.models``) -- somebody describing their own
        deployment is the authority on it, and for a self-hosted model the only
        source there is. Then the provider itself: the model service reports what
        its own catalogue serves, which is the figure the request will actually
        be measured against. It is absent until the service has answered once,
        and unknown is carried as unknown -- callers must not trim, gate or
        compact against a number in its place.

        ``binding`` is the route that serves this model, for a caller answering
        about a session rather than about the running turn. Outside a turn
        ``self.provider`` is the *default* binding's, so a session that had
        switched was reported the default model's window under its own model's
        name. A caller who knows the binding hands it over; one who does not gets
        the same answer as before.
        """
        model = model or self.model
        declared = declared_tokens(row_for(self._providers, model), "context_window")
        if declared:
            return Resolved(declared, SOURCE_DECLARED)
        provider = getattr(binding, "provider", None) or self.provider
        tokens = provider.context_window()
        return Resolved(tokens, SOURCE_SERVICE) if tokens else Resolved(None, SOURCE_UNKNOWN)

    def refresh_default_window(self) -> None:
        """Re-resolve the default window against the default binding's model.

        For the paths that run outside a turn: a default switch, or a model row
        the user declared in session (``model.overlay``), must reach the builders
        that sized themselves against the window at construction (the engine's
        history selector). Turns resolve their own window at entry, so they need
        nothing here.
        """
        self.apply_default_window(self.resolve_window(self._default.model).tokens)

    def apply_default_window(self, tokens: int | None) -> None:
        """Hand the out-of-turn fallback window to everything that keeps one."""
        self._default_window = tokens
        self.on_default_window(tokens)

    def note_window(self, model: str, resolved: Resolved) -> None:
        """Say once per model what its window resolved to, and size the turn by it.

        Called with the model that actually answered, which is not always the one
        the turn entered under -- a strategy rewrote it, or the chain fell back.
        That model's window sizes the rest of the turn, and the out-of-turn
        fallbacks too when it is the default model's.
        """
        if resolved.known and model == self.model:
            set_active_window(resolved.tokens)
        if resolved.known and model == self._default.model and self._default_window != resolved.tokens:
            self.apply_default_window(resolved.tokens)
        if model in self._noted:
            return
        self._noted.add(model)
        if resolved.known:
            logger.info("context window for {}: {} tokens ({})", model, resolved.tokens, resolved.source)
        else:
            logger.warning(
                "context window for {} is unknown: no table lists it, so history is not trimmed; "
                "give the model a row in providers.<provider>.models with contextWindow to size it",
                model,
            )


__all__ = ["SessionBindings"]
