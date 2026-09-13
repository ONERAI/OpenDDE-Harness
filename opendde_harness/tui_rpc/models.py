"""Pydantic DTOs used by the TUI RPC handlers and Python clients.

The registered handlers are authoritative. ``tests/test_tui_rpc_schema.py``
validates their actual results and notifications against OpenRPC; comparing
these DTOs alone cannot detect drift in a handler that returns plain dicts.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Re-usable model config.  ``extra="forbid"`` makes Pydantic emit
# ``additionalProperties: false`` in the generated JSON Schema, matching the
# OpenRPC schema's explicit ``additionalProperties: false`` on every object.
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    """Base class for all RPC models — forbids extra fields by default."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Public types (specs/tui-ipc.md §3.12)
# ---------------------------------------------------------------------------


# ``JsonValue`` in JSON Schema is the union of all primitive + container types.
# We use ``Any`` here because the schema declares ``JsonValue`` as a permissive
# any-of-primitives type and the schema-match test pins JsonValue to its OpenRPC
# spec rather than to its Pydantic schema (see ``components/schemas/JsonValue``).
JsonValue = Any


class SessionUsage(_Strict):
    #: The session's own totals as the banner opens on them. Zero for a session
    #: with no history; for a resumed one, what its stored records say its
    #: earlier turns moved and cost, so the footer and ``/status`` open on the
    #: real figure rather than on $0 (see ``session._history_usage``).
    input: int
    output: int
    cost_usd: float | None
    #: The same spend at the vendor's published price, whoever is billing. This
    #: is the figure the footer shows: on a plan ``cost_usd`` is None, and
    #: ``$0.000`` beside ``(sub)`` reads as free rather than as covered.
    list_cost_usd: float | None = None
    calls: int
    context_max: int
    context_source: str
    #: None when nothing measures the window: a compaction marker the session's
    #: own backend replays stands in front of the history, so the next call
    #: sends that marker rather than the messages it replaced, and their size
    #: says nothing about what the window will hold.
    context_used: int | None
    context_percent: int | None


class McpServerInfo(_Strict):
    """Metadata about a configured MCP server."""

    name: str
    transport: Literal["stdio", "sse", "streamableHttp"]
    connected: bool
    tool_count: int


class ProjectInstructionFile(_Strict):
    """One AGENTS.md / ODH.md the session found. Never its contents."""

    path: str = Field(..., description="Absolute path, which is what /memory on|off names.")
    display: str = Field(..., description="How it is shown: repo-relative, ~/... for the user scope.")
    size: int = Field(..., description="Size on disk in bytes, which is not the size sent when truncated.")
    truncated: bool = Field(..., description="Larger than the 32 KiB per-file cap; only the head is sent.")
    skipped: bool = Field(..., description="Found but not sent: the 128 KiB total was already spent.")
    enabled: bool = Field(..., description="False when /memory off switched it off for this session.")
    changed: bool = Field(..., description="Its content differs from what this session first saw in it.")


class SessionInfo(_Strict):
    """Init bundle returned by session.create and session.resume."""

    model: str
    model_id: str
    provider: str
    reasoning_effort: str | None
    context_window: int
    lazy: bool
    skills: dict[str, list[str]]
    tools: dict[str, list[str]]
    usage: SessionUsage
    version: str
    # When this version shipped, from the changelog the package carries. None
    # when the install has none to read.
    release_date: str | None
    cwd: str
    #: The instruction files this session sends, listed but never quoted.
    project_instructions: list[ProjectInstructionFile]
    mcp_servers: list[McpServerInfo]
    #: The model is billed by plan, so no per-call price describes it. Read from
    #: the provider entry the model id names: a sign-in is a plan, a key is
    #: metered, and nothing is guessed from the provider's name.
    subscription: bool
    #: Something compacts this session before its window runs out.
    auto_compact: bool
    update_available: str | None = None
    update_command: str | None = None


class SessionMessage(_Strict):
    """Stored messages mapped to the resume wire shape."""

    role: str
    text: str | None = None
    context: JsonValue = None
    name: JsonValue = None


class McpToolInfo(_Strict):
    """Metadata about a single tool exposed by an MCP server."""

    name: str = Field(..., description="Raw tool name (without mcp_<server>_ prefix).")
    description: str
    parameters: dict[str, JsonValue] = Field(..., description="JSON Schema for the tool's input arguments.")


class SkillInfo(_Strict):
    """Metadata about a skill (local or remote)."""

    name: str
    source: Literal["local", "remote"]
    pinned: bool
    description: str
    tags: list[str]


class UsageSnapshot(_Strict):
    """Token / cost usage reported at the end of a turn."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Share of the last call's prompt served from the provider's cache, 0-100;
    # None when the call sent no prompt.
    cache_hit_percent: int | None = None
    cost_usd: float | None = None
    #: What this call is worth at the vendor's published price, whoever is
    #: billing. On a plan ``cost_usd`` is None -- the subscription is the price
    #: -- and this is the figure a status line can show beside a `(sub)`.
    list_cost_usd: float | None = None
    context_used: int | None = None
    context_max: int | None = None
    context_percent: int | None = None
    # Which tier sized context_max (providers/rates SOURCE_*); "unknown" with
    # a zero context_max means no table lists the model, not that it is small.
    context_source: str | None = None
    #: The conversation was compacted after this call. ``context_used`` above
    #: measured a prompt that no longer exists, and nothing measures the one
    #: that replaced it until the next call reports: a status line that keeps
    #: showing the old percentage is stating a fact about a deleted
    #: conversation. None where no compaction ran, which is every ordinary turn.
    context_compacted: bool | None = None
    # -- the session, beside the turn ---------------------------------------
    #: Everything above describes this turn (its costs) or its last call (the
    #: token and context figures). These five describe the session: the totals
    #: from the tracker ``/status`` reads, so a footer showing the session's
    #: spend and a ``/status`` line reporting it cannot disagree. Absent only
    #: from a completion no turn runner filled in.
    session_input_tokens: int | None = None
    session_output_tokens: int | None = None
    #: None on a plan-billed provider, which states no per-token price.
    session_cost_usd: float | None = None
    #: None when no model used has a published price. Never zero for unknown.
    session_list_cost_usd: float | None = None
    session_calls: int | None = None


class CliResult(_Strict):
    """The result envelope returned by ``cli.dispatch``."""

    stdout: str = Field(..., description="Rich-rendered output with ANSI SGR sequences.")
    stderr: str = Field(..., description="Error / warning output with ANSI SGR sequences.")
    exit_code: int = Field(..., description="CLI command exit code; 0 = success.")
    error_code: int | None = Field(
        default=None,
        description=("Only present for timeout / not-dispatch-compatible cases (mirrors a JSON-RPC error code)."),
    )


class StubResult(_Strict):
    """Shared shape for all hermes-only stub method results (-32012)."""

    error: str = Field(..., description="Human-readable explanation of why this method is not supported in v0.1.")
    hint: str | None = Field(
        default=None,
        description="Optional hint to the user (e.g., 'Press Ctrl+C').",
    )


# ---------------------------------------------------------------------------
# TurnEvent — discriminated union over the 8 streaming event variants.
# ---------------------------------------------------------------------------


class MessageStartPayload(_Strict):
    turn_id: str


class MessageStartEvent(_Strict):
    type: Literal["message.start"]
    payload: MessageStartPayload


class EpisodeStartPayload(_Strict):
    index: int


class EpisodeStartEvent(_Strict):
    type: Literal["episode.start"]
    payload: EpisodeStartPayload


class TurnUsagePayload(_Strict):
    """Cumulative output of this turn's model calls so far, vendor-counted."""

    completion_tokens: int
    reasoning_tokens: int
    calls: int


class TurnUsageEvent(_Strict):
    type: Literal["turn.usage"]
    payload: TurnUsagePayload


class TurnRetryPayload(_Strict):
    attempt: int
    total: int
    reason: str
    #: True when text already streamed for this model call is void and the
    #: live buffer must start over; False when nothing had been shown yet.
    discard: bool


class TurnRetryEvent(_Strict):
    type: Literal["turn.retry"]
    payload: TurnRetryPayload


class TurnNoticePayload(_Strict):
    """A line the turn shows the user that is not part of the reply."""

    kind: Literal["model_fallback", "delivery_failed"]
    text: str


class TurnNoticeEvent(_Strict):
    type: Literal["turn.notice"]
    payload: TurnNoticePayload


class TokenDeltaPayload(_Strict):
    text: str


class TokenDeltaEvent(_Strict):
    type: Literal["token.delta"]
    payload: TokenDeltaPayload


class ThinkingDeltaPayload(_Strict):
    text: str


class ThinkingDeltaEvent(_Strict):
    type: Literal["thinking.delta"]
    payload: ThinkingDeltaPayload


class ToolStartPayload(_Strict):
    tool_call_id: str
    name: str
    arguments: dict[str, JsonValue]
    display: str | None = None


class ToolStartEvent(_Strict):
    type: Literal["tool.start"]
    payload: ToolStartPayload


class ToolProgressPayload(_Strict):
    tool_call_id: str
    preview: str


class ToolProgressEvent(_Strict):
    type: Literal["tool.progress"]
    payload: ToolProgressPayload


class ToolCompletePayload(_Strict):
    tool_call_id: str
    result_preview: str
    truncated: bool


class ToolCompleteEvent(_Strict):
    type: Literal["tool.complete"]
    payload: ToolCompletePayload


class MessageCompletePayload(_Strict):
    turn_id: str | None
    usage: UsageSnapshot


class MessageCompleteEvent(_Strict):
    type: Literal["message.complete"]
    payload: MessageCompletePayload


class ErrorEventPayload(_Strict):
    code: int
    message: str
    reason: Literal["cancelled_by_client", "internal"] | None = None
    detail: str | None = None


class ErrorEvent(_Strict):
    type: Literal["error"]
    payload: ErrorEventPayload


class CronDeliveredPayload(_Strict):
    job_id: str
    name: str
    text: str
    fired_at: str


class CronDeliveredEvent(_Strict):
    type: Literal["cron.delivered"]
    payload: CronDeliveredPayload


class CronMissedItem(_Strict):
    name: str
    scheduled_at: str
    message: str


class CronMissedPayload(_Strict):
    count: int
    items: list[CronMissedItem]


class CronMissedEvent(_Strict):
    type: Literal["cron.missed"]
    payload: CronMissedPayload


class ProteinDesignProgressPayload(_Strict):
    event_id: str
    task_id: str
    timestamp: str
    event_type: Literal["task", "cycle", "phase", "agent", "skill", "tool", "fold", "gate", "memory"]
    status: Literal["started", "progress", "completed", "failed"]
    cycle: int | None
    total_cycles: int | None
    phase: str
    actor: str
    skill: str | None
    tool: str | None
    summary: str
    duration_ms: float | None
    candidate_count: int | None
    error: str | None
    has_details: bool


class ProteinDesignProgressEvent(_Strict):
    type: Literal["protein_design.progress"]
    payload: ProteinDesignProgressPayload


TurnEvent = Annotated[
    Union[
        MessageStartEvent,
        EpisodeStartEvent,
        TurnRetryEvent,
        TurnUsageEvent,
        TurnNoticeEvent,
        TokenDeltaEvent,
        ThinkingDeltaEvent,
        ToolStartEvent,
        ToolProgressEvent,
        ToolCompleteEvent,
        MessageCompleteEvent,
        ErrorEvent,
        CronDeliveredEvent,
        CronMissedEvent,
        ProteinDesignProgressEvent,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# session.* methods
# ---------------------------------------------------------------------------


class SessionListItem(_Strict):
    """One row in the session picker (gatewayTypes.ts:130 SessionListItem)."""

    id: str = Field(..., description="Full session_key: <channel>:<chat_id>.")
    message_count: int
    preview: str
    source: str
    started_at: float = Field(..., description="Unix timestamp from created_at.")
    title: str


class SessionListParams(_Strict):
    limit: int | None = Field(default=None, description="Max sessions to return.")


class SessionListResult(_Strict):
    sessions: list[SessionListItem]


class SessionGetParams(_Strict):
    session_key: str


class SessionGetResult(_Strict):
    session: SessionInfo


class SessionCreateParams(_Strict):
    pass


class SessionCreateResult(_Strict):
    session_id: str
    info: SessionInfo


class SessionResumeParams(_Strict):
    session_id: str | None = None


class SessionResumeResult(_Strict):
    session_id: str
    info: SessionInfo
    messages: list[SessionMessage]


class SessionDeleteParams(_Strict):
    session_id: str = Field("", description="Full session_key as sent by the UI.")


class SessionDeleteResult(_Strict):
    deleted: str | None = Field(
        ...,
        description=(
            "The session_id that was deleted (matches the request param); null when no such session file existed."
        ),
    )


class SessionMostRecentParams(_Strict):
    pass


class SessionMostRecentResult(_Strict):
    session_id: str | None


class SessionTitleParams(_Strict):
    """Params per slash/commands/core.ts:201,218 — session_id + optional title."""

    session_id: str = Field("", description="Full session_key.")
    title: str | None = None


class SessionTitleResult(_Strict):
    """Response per gatewayTypes.ts:154 SessionTitleResponse.

    pending=True means the title is held in memory for a lazy (never-saved)
    session and lands with the session's first save.
    """

    title: str | None
    session_key: str
    pending: bool


class SessionClearParams(_Strict):
    """Params for session.clear — wipe messages in place, keep the sid."""

    session_id: str = Field("", description="Full session_key to clear.")


class SessionClearResult(_Strict):
    session_id: str = Field(..., description="The same session_key (no new id minted).")
    cleared: bool = Field(..., description="True when the in-place wipe ran.")


class SessionUndoParams(_Strict):
    """Params for session.undo — drop the last n turns (default 1)."""

    session_id: str = Field("", description="Full session_key to undo.")
    n: int = Field(1, description="Trailing turns to drop (role==user boundary).")


class SessionUndoResult(_Strict):
    removed: int = Field(..., description="Messages dropped (0 = nothing to undo).")
    #: What the conversation holds now the exchange is gone, so a status line
    #: can stop reporting a window that has not been that full since. Absent
    #: when nothing was removed, and null where nothing measures the window --
    #: the same answer session.info gives for that session, from the same
    #: policy. Estimated offline otherwise, and 0 rather than an error when
    #: nothing can size it.
    context_used: int | None = None


class SessionExportParams(_Strict):
    """Params for session.export — render a transcript to a Markdown file."""

    session_id: str | None = Field(
        default=None,
        description="Session id / prefix / full key to export; not_found when omitted.",
    )


class SessionExportResult(_Strict):
    exported: bool = Field(..., description="True when a Markdown file was written.")
    path: str | None = Field(..., description="Absolute path of the written file, or null on failure.")
    reason: str | None = Field(
        default=None,
        description="Failure reason when not exported: not_found | ambiguous | write_failed.",
    )
    candidates: list[str] | None = Field(
        default=None,
        description="Candidate full keys when reason is ambiguous.",
    )


class SessionInstructionsParams(_Strict):
    """Params for session.instructions — list the files, or switch one."""

    action: Literal["list", "on", "off"] = Field(
        default="list",
        description="list (default) reports; on / off switch one file for this session only.",
    )
    path: str | None = Field(
        default=None,
        description="Which file to switch: its path, its shown name, or its unambiguous basename.",
    )
    session_id: str | None = Field(
        default=None,
        description="Whose view this is; on / off apply to that conversation alone.",
    )


class SessionInstructionsResult(_Strict):
    cwd: str = Field(..., description="The directory the search ran from.")
    files: list[ProjectInstructionFile] = Field(..., description="Every file found, outermost first.")
    changed: str | None = Field(..., description="The shown name of the file just switched, else null.")
    error: str | None = Field(..., description="Why nothing was switched, in one line, else null.")


# ---------------------------------------------------------------------------
# turn.* methods
# ---------------------------------------------------------------------------


class TurnSendParams(_Strict):
    session_key: str
    content: str
    channel: str | None = None
    chat_id: str | None = None
    sender_id: str | None = None
    # Attachment paths, workspace-relative or absolute. The same lane channels
    # already use (``TurnRequest.media``): a vision-capable model gets the
    # picture inlined in the user message, anything else gets a note naming it.
    # Paths rather than bytes -- the caller has already put the file in the
    # workspace, and every file tool is workspace-scoped. Bounded here so a
    # malformed caller is refused at the schema rather than resolving thousands
    # of paths; the renderer caps how many are inlined regardless.
    media: list[str] | None = Field(default=None, max_length=64)


class TurnSendResult(_Strict):
    turn_id: str
    accepted: bool


class TurnSubscribeParams(_Strict):
    session_key: str


class TurnSubscribeResult(_Strict):
    subscription_id: str


class TurnUnsubscribeParams(_Strict):
    subscription_id: str


class TurnUnsubscribeResult(_Strict):
    unsubscribed: bool


class TurnCancelParams(_Strict):
    session_key: str


class TurnCancelResult(_Strict):
    cancelled: bool


# ---------------------------------------------------------------------------
# mcp.* methods
# ---------------------------------------------------------------------------


class McpListParams(_Strict):
    pass


class McpListResult(_Strict):
    servers: list[McpServerInfo]


class McpTestParams(_Strict):
    server_name: str


class McpTestResult(_Strict):
    ok: bool
    latency_ms: float
    error: str | None = None


class McpToolsParams(_Strict):
    server_name: str


class McpToolsResult(_Strict):
    tools: list[McpToolInfo]


# ---------------------------------------------------------------------------
# skill.* methods
# ---------------------------------------------------------------------------


class SkillListParams(_Strict):
    source: Literal["local", "remote", "all"] | None = Field(
        default=None,
        description="Filter by skill source; default 'all'.",
    )


class SkillListResult(_Strict):
    skills: list[SkillInfo]


class SkillPinParams(_Strict):
    skill_name: str


class SkillPinResult(_Strict):
    pinned: bool


class SkillUnpinParams(_Strict):
    skill_name: str


class SkillUnpinResult(_Strict):
    unpinned: bool


# ---------------------------------------------------------------------------
# model.* methods
# ---------------------------------------------------------------------------


class ModelLabel(_Strict):
    """How a model reads to a person, for the ids in ``models``.

    One per offered id. ``label`` is the name the model's own declared row
    gives it, else the one the model service reports, else the id with its
    provider prefix dropped -- the ids in ``models`` are qualified, and
    repeating the provider on every row of that provider's own list reads as
    noise.
    """

    label: str
    description: str | None = None


class ModelOptionProvider(_Strict):
    """One provider row in the ``/model`` picker.

    ``slug`` is a pi provider id, which is also the key of the ``providers``
    entry it describes and the prefix of every model id in ``models``.
    """

    slug: str
    name: str
    authenticated: bool
    is_current: bool
    #: How this provider is reached: a sign-in, an address this config declares,
    #: or a key. ``providers.auth``'s three shapes, and the only three.
    auth_type: Literal["oauth", "endpoint", "key"]
    #: The environment variable that already supplies this provider's key, or
    #: null when none does. Reported, never sent: pi resolves the environment
    #: itself.
    key_env: str | None
    models: list[str]
    models_loaded: bool
    model_labels: dict[str, ModelLabel]
    total_models: int
    #: Every way in pi offers for this provider, in pi's own order: a sign-in, a
    #: key, or both. Read from the model service, which reads pi's own provider
    #: objects. Empty when it could not be asked -- a picker that does not know
    #: offers no choice.
    auth_methods: list[Literal["oauth", "key"]] = Field(default_factory=list)
    #: pi's own label for the sign-in option ("Sign in with SuperGrok or X
    #: Premium"), or null where pi carries none and its generic sentence applies.
    login_label: str | None = None
    #: pi's own name for this provider's key ("Anthropic API key").
    key_label: str | None = None
    #: A key must be typed before this provider can serve anything.
    needs_api_key: bool
    #: This provider is one the config declares, so the form must ask for the
    #: address it lives at and the wire it speaks -- both, never one.
    needs_base_url: bool
    #: The declared address and wire as configured, for a form that prefills
    #: them; null for one of pi's own, which carries its own.
    base_url: str | None
    api: str | None
    warning: str


class ModelOptionsParams(_Strict):
    session_id: str | None = None
    slug: str | None = None
    # True by default, as ``openrpc.json`` declares and both clients rely on:
    # the picker opens with an explicit ``false`` for a cheap first screen and
    # then omits the field when it expands one provider, so a default of false
    # answered that expansion with the same short list and the client read it
    # as "failed to load more models". Either way the answer is local: the
    # catalogue is two bundled files, and no provider is asked what it serves.
    include_catalog: bool = True
    #: Ask the declared endpoints what they serve before answering: their
    #: ``/models``, fetched now. Off by default; the picker asks once, in the
    #: background, after it has painted the last known list.
    refresh: bool = False


class ModelOptionsResult(_Strict):
    model: str
    provider: str
    #: ``agents.defaults.model``: what new sessions start on, so the picker can
    #: mark its row the way pi marks its default.
    default_model: str = ""
    providers: list[ModelOptionProvider]
    #: By provider id, the endpoints a ``refresh`` could not ask, with the
    #: sentence each failed with. Empty when nothing was asked or all answered.
    refresh_errors: dict[str, str] = Field(default_factory=dict)


class ModelSaveKeyParams(_Strict):
    slug: str
    # Empty for a provider reached without one: a self-hosted endpoint that
    # wants no key, or a built-in whose key is already in the environment. The
    # handler rejects an empty one wherever a key is what reaches the vendor.
    api_key: str = ""
    # A provider this config declares: where it lives, and the wire it speaks.
    # Required together (the schema refuses either alone) and both meaningless
    # for one of pi's own, which carries its address and its protocol.
    base_url: str | None = None
    api: str | None = None
    session_id: str | None = None


class ModelSaveKeyResult(_Strict):
    provider: ModelOptionProvider


class ModelDeclareProviderParams(_Strict):
    """Declare a provider this config does not hold yet, in one submission.

    The three things such a provider needs and nothing else: the id it is filed
    under, the address it lives at, and a key if it wants one. What it serves is
    read from the endpoint itself (its ``/models``), the way pi treats
    OpenRouter; ``model`` names ids by hand only for an endpoint that publishes
    no list, comma-separated.

    The wire is not among them. This declares an OpenAI-compatible endpoint,
    which is what a relay or a self-hosted server implements, so ``api`` is
    ``openai-completions`` and there is nothing for the form to choose. A
    provider that speaks another wire is written by ``ddeharness provider set``,
    which takes every field the entry has.
    """

    provider: str
    base_url: str
    #: Optional: ids typed by hand, comma-separated, for an endpoint that
    #: publishes no list. Otherwise the list is discovered.
    model: str = ""
    #: Empty for the common case: a self-hosted server usually wants no key.
    api_key: str = ""
    session_id: str | None = None


class ModelDeclareProviderResult(_Strict):
    provider: ModelOptionProvider
    #: How many models the endpoint published when asked just now.
    discovered: int


class ModelLogoutParams(_Strict):
    """pi's ``/logout``: forget a provider's credential and keep its declaration."""

    slug: str
    session_id: str | None = None


class ModelLogoutResult(_Strict):
    #: False when there was nothing stored to forget.
    forgotten: bool


class ModelScopeParams(_Strict):
    """pi's scoped models: read the saved scope, or write it when ``write`` is set."""

    #: The ``<provider>/<model>`` ids of the scope; ``None`` is every model.
    models: list[str] | None = None
    write: bool = False
    session_id: str | None = None


class ModelScopeResult(_Strict):
    models: list[str] | None


class ModelLoginParams(_Strict):
    """Start a provider's sign-in. The steps arrive as ``login.step`` pushes."""

    provider: str
    #: The id the pushed steps will carry, chosen by the client so it knows its
    #: own flow's steps before the first one arrives. Minted by the gateway
    #: when absent.
    login_id: str | None = Field(default=None, min_length=1, max_length=64)
    session_id: str | None = None


class ModelLoginResult(_Strict):
    #: The id the pushed steps carried, so a client can tell which flow ended.
    login_id: str
    provider: ModelOptionProvider


class ModelLoginAnswerParams(_Strict):
    """What the sign-in asked for: an option id, or a pasted code."""

    login_id: str
    answer: str


class ModelLoginAnswerResult(_Strict):
    #: False when nothing was waiting: the login ended, or the callback won.
    answered: bool


class ModelLoginCancelParams(_Strict):
    login_id: str


class ModelLoginCancelResult(_Strict):
    cancelled: bool


class ModelOverlayParams(_Strict):
    """One field of the current model's declared row (``providers.<id>.models``).

    ``field`` is one of the row's own: name, description, api, catalog_model,
    context_window, max_tokens (a count, ``128k`` allowed), reasoning (on/off),
    reasoning_effort (pi's off/minimal/low/medium/high/xhigh/max) or
    temperature. ``default`` clears whichever is named.
    """

    field: str
    value: str
    session_id: str | None = None


class ModelOverlayResult(_Strict):
    model: str
    field: str
    value: str | int | float | bool | None
    #: The window the loop runs with after the change, for a size field.
    context_window: int | None = None


# ---------------------------------------------------------------------------
# config.* methods
# ---------------------------------------------------------------------------


class ConfigGetParams(_Strict):
    keys: list[str] | None = Field(
        default=None,
        description=("If omitted, return all whitelisted fields. Unknown keys are silently dropped."),
    )


class ConfigGetResult(_Strict):
    config: dict[str, JsonValue]


class ConfigSetParams(_Strict):
    key: str
    value: JsonValue
    # Model-switch extras. ``scope`` decides the reach of a ``key="model"``
    # switch: this conversation, or the default a new one starts on. There is no
    # provider beside it: a model id names its provider, and a second field
    # saying so again is what sent one vendor's model to another vendor's key.
    session_id: str | None = None
    scope: Literal["session", "default"] | None = None


class ConfigSetResult(_Strict):
    applied: bool
    # ``previous`` is a *required* field whose value may legitimately be
    # ``null``.  We type it as ``JsonValue`` (``Any``) because ``JsonValue``
    # already includes ``null``; a oneOf with a second null branch rejects it.
    previous: JsonValue = Field(...)
    # Present on a model switch: what was applied, and where it reached.
    value: str | None = None
    scope: Literal["session", "default"] | None = None
    session_id: str | None = None
    # Does the asking conversation now run this model? A default-scoped switch
    # moves the sessions that never chose one, so scope alone cannot answer it
    # and a client that guesses shows a model the conversation is not on.
    applies_to_session: bool | None = None


# ---------------------------------------------------------------------------
# system.* methods
# ---------------------------------------------------------------------------


class SystemHelloParams(_Strict):
    client_version: str
    client_capabilities: list[str] | None = None


class SystemHelloSession(_Strict):
    default_channel: Literal["tui"]
    default_session_key: str


class SystemHelloResult(_Strict):
    server_version: str
    server_capabilities: list[str]
    session: SystemHelloSession


class SystemPingParams(_Strict):
    pass


class SystemPingResult(_Strict):
    pong: Literal[True]
    server_time_ms: float


class SystemVersionParams(_Strict):
    pass


class SystemVersionResult(_Strict):
    server_version: str
    schema_version: str = Field(..., description="OpenRPC info.version mirrored back to client.")
    opendde_harness_version: str


# ---------------------------------------------------------------------------
# cli.dispatch
# ---------------------------------------------------------------------------


class CliDispatchParams(_Strict):
    argv: list[str] = Field(..., description="Pre-tokenized argv (TUI side has already shlex-split).")
    width: int = Field(
        ...,
        ge=20,
        le=500,
        description="Ink container width in cells; required for Rich Console wrapping.",
    )
    timeout_s: float | None = Field(
        default=None,
        description="Override the default 30s timeout for long-running commands.",
    )


CliDispatchResult = CliResult


# ---------------------------------------------------------------------------
# setup.status / reload.mcp
# ---------------------------------------------------------------------------


class SetupStatusParams(_Strict):
    pass


class SetupStatusResult(_Strict):
    provider_configured: bool
    compute_configured: bool | None = None
    error: str | None = None


class ReloadMcpParams(_Strict):
    pass


class ReloadMcpResult(_Strict):
    ok: bool
    reloaded: int
    tools_changed: bool


# ---------------------------------------------------------------------------
# commands.catalog (dynamic Typer-reflection slash catalog)
# ---------------------------------------------------------------------------


class CommandsCatalogParams(_Strict):
    pass


class CatalogCommand(_Strict):
    """One reflected CLI command, as a slash popup shows it."""

    name: str = Field(..., description='Space-separated argv without the leading slash, e.g. "provider list".')
    description: str = Field(..., description="One line from the command's own Typer help or docstring.")
    argument_hint: str | None = Field(
        default=None,
        description="Positional arguments, <required> and [optional]. None when the command takes none.",
    )
    timeout_s: float | None = Field(
        default=None,
        gt=0,
        description="Seconds this command is allowed where the ordinary slash timeout is not enough; None otherwise.",
    )


class CommandsCatalogResponse(_Strict):
    """Slash-command catalog reflected from opendde_harness.cli.commands.app.

    The TUI gates on non-empty ``pairs`` (createGatewayEventHandler.ts) and
    renders ``categories`` / ``skill_count`` in ``/help``. Every typed slash
    is sent to ``slash.exec`` verbatim: the TUI resolves no aliases, so the
    catalog carries none.
    """

    pairs: list[tuple[str, str]] = Field(
        ...,
        description=(
            "RETIRED WITH THE INK FRONT-END. Group heads only; read `commands` instead. Kept because the Ink "
            "client gates on non-empty pairs. Remove with that client at the cut-over."
        ),
    )
    commands: list[CatalogCommand] = Field(
        ...,
        description=(
            "Every command the gateway will run for a slash, with its own help text. Exactly what "
            "cli.dispatch accepts, so a bare group head is absent."
        ),
    )
    categories: list[str] = Field(
        ...,
        description="'(top-level)' first then alphabetical group names.",
    )
    skill_count: int = Field(
        ...,
        ge=0,
        description="Total skill count via skill_forge.store; 0 + warning if DB missing.",
    )
    warning: str | None = Field(
        default=None,
        description="Optional warning pushed to TUI activity strip.",
    )


# ---------------------------------------------------------------------------
# hermes-only stubs (10 methods, all share StubResult)
# ---------------------------------------------------------------------------


class VoiceToggleParams(_Strict):
    action: str | None = None


VoiceToggleResult = StubResult


class BrowserManageParams(_Strict):
    action: str | None = None
    url: str | None = None


BrowserManageResult = StubResult


class SpawnTreeSaveParams(_Strict):
    name: str | None = None


SpawnTreeSaveResult = StubResult


class SpawnTreeListParams(_Strict):
    pass


SpawnTreeListResult = StubResult


class SpawnTreeLoadParams(_Strict):
    name: str | None = None


SpawnTreeLoadResult = StubResult


class ProcessStopParams(_Strict):
    pass


ProcessStopResult = StubResult


class RollbackListParams(_Strict):
    pass


RollbackListResult = StubResult


class RollbackDiffParams(_Strict):
    id: str | None = None


RollbackDiffResult = StubResult


class RollbackRestoreParams(_Strict):
    id: str | None = None


RollbackRestoreResult = StubResult


class ToolsConfigureParams(_Strict):
    pass


ToolsConfigureResult = StubResult


# ---------------------------------------------------------------------------
# Method DTO registry. The dispatcher, not this subset of typed handlers,
# defines the registered surface; tests exercise it through actual requests.
# ---------------------------------------------------------------------------

METHOD_MODELS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    # session.*
    "session.list": (SessionListParams, SessionListResult),
    "session.create": (SessionCreateParams, SessionCreateResult),
    "session.resume": (SessionResumeParams, SessionResumeResult),
    "session.delete": (SessionDeleteParams, SessionDeleteResult),
    "session.most_recent": (SessionMostRecentParams, SessionMostRecentResult),
    "session.title": (SessionTitleParams, SessionTitleResult),
    "session.clear": (SessionClearParams, SessionClearResult),
    "session.undo": (SessionUndoParams, SessionUndoResult),
    "session.export": (SessionExportParams, SessionExportResult),
    "session.instructions": (SessionInstructionsParams, SessionInstructionsResult),
    # turn.*
    "turn.send": (TurnSendParams, TurnSendResult),
    "turn.subscribe": (TurnSubscribeParams, TurnSubscribeResult),
    "turn.unsubscribe": (TurnUnsubscribeParams, TurnUnsubscribeResult),
    "turn.cancel": (TurnCancelParams, TurnCancelResult),
    # model.*
    "model.options": (ModelOptionsParams, ModelOptionsResult),
    "model.save_key": (ModelSaveKeyParams, ModelSaveKeyResult),
    "model.declare_provider": (ModelDeclareProviderParams, ModelDeclareProviderResult),
    "model.scope": (ModelScopeParams, ModelScopeResult),
    "model.overlay": (ModelOverlayParams, ModelOverlayResult),
    "model.login": (ModelLoginParams, ModelLoginResult),
    "model.login_answer": (ModelLoginAnswerParams, ModelLoginAnswerResult),
    "model.login_cancel": (ModelLoginCancelParams, ModelLoginCancelResult),
    # config.*
    "config.get": (ConfigGetParams, ConfigGetResult),
    "config.set": (ConfigSetParams, ConfigSetResult),
    # system.*
    "system.hello": (SystemHelloParams, SystemHelloResult),
    "system.ping": (SystemPingParams, SystemPingResult),
    "system.version": (SystemVersionParams, SystemVersionResult),
    # cli.* / setup.* / reload.* / commands.*
    "cli.dispatch": (CliDispatchParams, CliResult),
    "setup.status": (SetupStatusParams, SetupStatusResult),
    "reload.mcp": (ReloadMcpParams, ReloadMcpResult),
    "commands.catalog": (CommandsCatalogParams, CommandsCatalogResponse),
}

__all__ = [
    # public types
    "SessionInfo",
    "ProjectInstructionFile",
    "SessionUsage",
    "ProteinDesignProgressEvent",
    "SessionListItem",
    "SessionMessage",
    "McpServerInfo",
    "McpToolInfo",
    "SkillInfo",
    "ModelOptionProvider",
    "UsageSnapshot",
    "CliResult",
    "StubResult",
    "CommandsCatalogResponse",
    "TurnEvent",
    "SessionMostRecentParams",
    "SessionMostRecentResult",
    "SessionInstructionsParams",
    "SessionInstructionsResult",
    "SessionTitleParams",
    "SessionTitleResult",
    "SessionClearParams",
    "SessionClearResult",
    "SessionUndoParams",
    "SessionUndoResult",
    "SessionExportParams",
    "SessionExportResult",
    "MessageStartEvent",
    "EpisodeStartEvent",
    "TurnRetryEvent",
    "TurnUsageEvent",
    "TurnNoticeEvent",
    "TokenDeltaEvent",
    "ThinkingDeltaEvent",
    "ToolStartEvent",
    "ToolProgressEvent",
    "ToolCompleteEvent",
    "MessageCompleteEvent",
    "ErrorEvent",
    "CronDeliveredEvent",
    "CronDeliveredPayload",
    "CronMissedEvent",
    "CronMissedItem",
    "CronMissedPayload",
    # registry
    "METHOD_MODELS",
]
