"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings, SettingsConfigDict

from opendde_harness.config.features import (
    ContextConfig,
    MemoryConfig,
    PluginsConfig,
    RuntimeConfig,
    SkillForgeConfig,
    TracingConfig,
)
from opendde_harness.providers import model_id, pi_ids


class Base(BaseModel):
    """Accepts both camelCase and snake_case keys; rejects keys it does not know.

    A key this release does not define is a typo or one an earlier release
    wrote, and either is an error naming the key, never a value silently
    dropped. Nothing is migrated: the config is the current schema.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


#: pi's thinking-level vocabulary (``ModelThinkingLevel``). Each family is sent
#: the nearest level it takes: OpenAI's effort ladder (``max`` on gpt-5.6 and
#: gpt-6-astra, ``xhigh`` from gpt-5.2), DeepSeek's three steps, a token
#: budget for DashScope, on/off for Z.ai. ``off`` switches thinking off.
ReasoningEffort = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

#: pi's own compatibility fields, per wire: what ``OpenAICompletionsCompat``,
#: ``OpenAIResponsesCompat`` and ``AnthropicMessagesCompat`` declare in
#: ``@earendil-works/pi-ai/dist/types.d.ts``. A ``compat`` block is a set of
#: overrides for what pi otherwise detects from a provider's address, and it
#: travels to pi untouched -- so a key pi has no field for would be sent and
#: ignored, which is why the key set is vendored here and an unknown one is
#: refused where the config is read.
#:
#: The two wires missing from this table take no compat at all: pi types
#: ``Model.compat`` as ``never`` for ``google-generative-ai`` and
#: ``mistral-conversations``, and a block written under either is refused.
#:
#: Regenerate from ``ui-tui/`` with::
#:
#:     node --input-type=module -e "
#:     import {readFileSync} from 'node:fs';
#:     const src = readFileSync('node_modules/@earendil-works/pi-ai/dist/types.d.ts', 'utf8');
#:     for (const name of ['OpenAICompletionsCompat','OpenAIResponsesCompat','AnthropicMessagesCompat']) {
#:       const start = src.indexOf('export interface ' + name + ' {');
#:       const body = src.slice(start, src.indexOf('\n}', start));
#:       console.log(name, [...body.matchAll(/^\s{4}([A-Za-z][A-Za-z0-9]*)\?:/gm)].map(m => m[1]))}"
_RESPONSES_COMPAT = frozenset(
    {
        "sessionAffinityFormat",
        "supportsAdditionalTools",
        "supportsDeveloperRole",
        "supportsExplicitPromptCacheMode",
        "supportsLongCacheRetention",
        "supportsMaxOutputTokens",
        "supportsOpenAIGrammarTools",
        "supportsStrictMode",
        "supportsToolSearch",
    }
)

COMPAT_KEYS: dict[str, frozenset[str]] = {
    "anthropic-messages": frozenset(
        {
            "allowEmptySignature",
            "allowedFallbackModels",
            "forceAdaptiveThinking",
            "sendSessionAffinityHeaders",
            "supportsCacheControlOnTools",
            "supportsEagerToolInputStreaming",
            "supportsLongCacheRetention",
            "supportsMidConvoEffort",
            "supportsStrictTools",
            "supportsTemperature",
            "supportsToolReferences",
        }
    ),
    "azure-openai-responses": _RESPONSES_COMPAT,
    "openai-codex-responses": _RESPONSES_COMPAT,
    "openai-completions": frozenset(
        {
            "cacheControlFormat",
            "chatTemplateArgs",
            "chatTemplateKwargs",
            "deferredToolsMode",
            "maxTokensField",
            "openRouterRouting",
            "requiresAssistantAfterToolResult",
            "requiresReasoningContentOnAssistantMessages",
            "requiresThinkingAsText",
            "requiresToolResultName",
            "sendSessionAffinityHeaders",
            "sessionAffinityFormat",
            "supportsDeveloperRole",
            "supportsFinishReason",
            "supportsLongCacheRetention",
            "supportsOpenAIGrammarTools",
            "supportsReasoningEffort",
            "supportsStore",
            "supportsStrictMode",
            "supportsThinkingTokenBudget",
            "supportsUsageInStreaming",
            "thinkingFormat",
            "thinkingTokenBudgetField",
            "vercelGatewayRouting",
            "vllmPriority",
            "zaiToolStream",
        }
    ),
    "openai-responses": _RESPONSES_COMPAT,
}


def _refuse_unknown_compat(compat: dict[str, object] | None, api: str, subject: str) -> None:
    """Refuse a ``compat`` key the wire it is written under does not read.

    Always against one named wire, never against the union of all of them: a
    list of every field every wire reads answers a question nobody asked, and
    the wire is always knowable by the time this matters -- a model row that
    leaves ``api`` to its provider is checked by :class:`ProviderEntry`, which
    can see both.
    """
    if not compat:
        return
    if api not in COMPAT_KEYS:
        raise ValueError(
            f"{subject} sets compat, and pi reads none on {api}: compatibility overrides exist for "
            f"{', '.join(sorted(COMPAT_KEYS))}. Delete the block."
        )
    unknown = sorted(key for key in compat if key not in COMPAT_KEYS[api])
    if unknown:
        raise ValueError(
            f"{subject} sets compat {', '.join(repr(key) for key in unknown)}, which pi does not read on "
            f"{api}: the block travels to pi untouched, so a field it does not have would be sent and "
            f"ignored. What that wire reads: {', '.join(sorted(COMPAT_KEYS[api]))}."
        )


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.opendde_harness/workspace"
    # "<pi provider id>/<model id>", exactly as pi spells it:
    # "openai-codex/gpt-5.4", "my-vllm/qwen3-32b". The prefix names the
    # `providers` entry whose credential and address serve the model, and it is
    # the only thing that does -- a bare id names nobody and is refused.
    model: str = "anthropic/claude-opus-4-5"
    # No maxTokens, no contextWindowTokens and no temperature here on purpose.
    # A number in a config file cannot be right for every model -- too large a
    # ceiling is a 400, too small truncates or trims silently, and a model that
    # rejects `temperature` outright refuses every request that carries one
    # ("Unsupported parameter: temperature" is what the Codex wire answers). So
    # all three are per model: the two limits resolve from the model's own row
    # and then from what the model layer reports, and a temperature is sent only
    # where a row asks for one. The place to declare any of them is a row in
    # `providers.<id>.models`: `contextWindow`, `maxTokens`, `temperature`.
    # A request is bounded per silence, never in total: the wait for its first
    # event (a model with hidden reasoning thinks for minutes before it), then
    # the gap between events. A stalled gateway surfaces after the idle bound,
    # and a reply that keeps arriving is never cut off for being long.
    llm_first_token_timeout: int = 300
    llm_idle_timeout: int = 120
    # How many times a request the server has not accepted may be sent again:
    # a refused connection, a 429 or a 5xx before any stream opened. The
    # transport applies it, honouring Retry-After and the first-event budget.
    # A stream that was accepted and began delivering is never re-sent
    # automatically; the user's own retry is the only second attempt there.
    llm_retries: int = Field(default=3, ge=0)
    max_tool_iterations: int = 40
    # Cap on subagent VMs running at once (excess spawns queue). ge=1: a
    # 0/negative cap would deadlock every subagent (Semaphore(0)).
    max_concurrent_subagents: int = Field(default=4, ge=1)
    # Spawn rate limit per session, per rolling hour — the concurrency gate
    # alone can't stop a prompt-injected agent from spawning indefinitely (each
    # finishes, freeing a slot for the next; the cross-turn re-injection loop
    # needs no user input). A rolling window bounds a runaway to N/hour yet
    # auto-recovers, so it never permanently locks out heavy legitimate use.
    # Counted per session so one busy session can't throttle others.
    max_subagent_spawns_per_hour: int = Field(default=30, ge=1)
    # Empty-response recovery: recover turns the model ends with no visible text
    # (post-tool empty / thinking-only) instead of surfacing a dud "no response
    # to give". Budgets are per-turn.
    empty_recovery_enabled: bool = True
    post_tool_empty_max_nudges: int = 1
    thinking_prefill_max_retries: int = 2
    empty_content_max_retries: int = 3
    # How hard every model thinks unless its own overlay says otherwise.
    # pi's DEFAULT_THINKING_LEVEL; None leaves each vendor's default in place.
    # A model the catalogue marks as non-reasoning is sent nothing.
    reasoning_effort: ReasoningEffort | None = "medium"

    @model_validator(mode="after")
    def _model_names_its_provider(self) -> "AgentDefaults":
        """The default model is "<provider>/<model>", and says so or is refused.

        One identity, stated once in the file: every consumer -- routing, the
        catalogue, the window and price readers, the model row lookup, the
        provider being built -- then sees "deepseek/deepseek-chat" and never
        has to work out who serves a bare "deepseek-chat".

        Whether that provider is configured is :class:`Config`'s question, not
        this model's: the providers map is a sibling field and is not visible
        from here.

        Empty is "not chosen yet", which the setup gate parks on and the wizard
        writes when the provider that served the default is removed; only a
        model that is named has to name its provider.
        """
        if self.model == "":
            return self
        provider, model = model_id.split(self.model)
        if not provider or not model:
            raise ValueError(
                f"agents.defaults.model = {self.model!r} names no provider. Write it as "
                f"<provider>/<model> -- the provider is a key of the providers section "
                f"(for example openrouter/{self.model or 'some-model'})."
            )
        return self

    # ``provider`` was here, naming the section that served a bare model id.
    # The id carries its provider now, so the field said a second time what the
    # id already says -- and when the two disagreed the field won, which sent
    # one vendor's model to another vendor's key.


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)
    #: pi's scoped models: the ``<provider>/<model>`` ids ``/model`` shows under
    #: its "scoped" tab, in the order ``/scoped-models`` saved them. ``None`` is
    #: pi's "all enabled": no scope, the picker opens on every model.
    scoped_models: list[str] | None = None


class TuiConfig(Base):
    """Terminal UI preferences shared by config files and the RPC handlers."""

    # Deliberately a plain string rather than a Literal: a name written by an
    # earlier release must not stop the whole config from loading. The TUI
    # validates it on read and falls back to `default` for anything else.
    theme: str = Field(
        default="default",
        strict=True,
        pattern=r"^[A-Za-z0-9_-]+$",
        description=(
            "Palette the TUI paints with: 'dark', 'light', or 'default' to follow the terminal. "
            "Any other name falls back to 'default'. "
            "OPENDDE_HARNESS_TUI_THEME and OPENDDE_HARNESS_TUI_LIGHT outrank this."
        ),
    )
    show_token_usage: bool = Field(default=True, strict=True)


class ModelCost(Base):
    """pi's four per-million rates for a model, as pi's ``cost`` row spells them.

    Only for a model no catalogue prices -- a self-hosted deployment, a relay
    that renamed what it fronts. Without one such a model reports unknown spend
    rather than borrowing a hosted model's rate, which is the honest answer and
    the default.
    """

    input: float = Field(default=0.0, ge=0)
    output: float = Field(default=0.0, ge=0)
    cache_read: float = Field(default=0.0, ge=0)
    cache_write: float = Field(default=0.0, ge=0)


class ModelEntry(Base):
    """One model of a provider: pi's model row, plus what only this project reads.

    Written either as a bare id string or as this row. The string is the whole
    declaration for a model whose facts are already known -- a built-in's own
    catalogue carries them -- and the row is for what no catalogue can carry:
    a deployment the operator configured, a model newer than pi's catalogue, a
    relay that renamed what it fronts.

    ``context_window`` and ``max_tokens`` are the first tier of both ladders,
    ahead of anything the model layer reports: a person describing their own
    deployment is the authority on it. Per model, and only per model -- one
    number for every model a session switches to is wrong for all but one of
    them. A deployment configured with a smaller window than the model's native
    one is exactly the case for this: the vendor's own figure would be an
    over-estimate, and an over-estimate is a refused request rather than wasted
    context. A limit nothing knows is left out rather than guessed.

    ``api`` picks the wire for this one model, over the provider's own: one
    relay can serve different models on different wires, so a provider-wide
    setting cannot always be right.

    ``compat`` overrides what pi otherwise detects from the provider's address,
    for this one model: the wire's own fields (:data:`COMPAT_KEYS`), passed to
    pi as the row's ``compat``. It replaces the provider's block rather than
    merging with it, so one row's answer is the whole answer for that row.

    The last three are this project's, not pi's, and the request path is what
    reads them. ``reasoning_effort`` and ``temperature`` are per model because a
    hybrid model one wants fast and a reasoning model one wants deep share a
    session, and because some models reject the usual default outright.
    ``catalog_model`` names the catalogue model a deployment serves, for an id
    that names a deployment rather than a model (an Azure deployment called
    "prod"): its limits and its price then apply. Without it such a
    deployment's limits are unknown -- a familiar-looking deployment name is
    not evidence of what is behind it.
    """

    id: str = Field(min_length=1)
    name: str = ""
    #: Shown under the name in the model picker. Ours; pi's rows carry none.
    description: str = ""
    api: str | None = None
    context_window: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    reasoning: bool | None = None
    input: list[Literal["text", "image"]] | None = None
    cost: ModelCost | None = None
    compat: dict[str, object] | None = None

    # --- read by this project's request path, never sent to pi as a row field
    reasoning_effort: ReasoningEffort | None = None
    temperature: float | None = Field(default=None, ge=0)
    catalog_model: str | None = None

    @model_validator(mode="after")
    def _api_is_one_pi_serves(self) -> "ModelEntry":
        if self.api is not None and self.api not in pi_ids.APIS:
            raise ValueError(f"model {self.id!r} names api {self.api!r}; known apis are {', '.join(pi_ids.APIS)}")
        # Only against a wire this row names itself. A row that leaves the wire
        # to its provider is checked by ProviderEntry, which can see both --
        # from in here the provider is not visible, and every wire's fields at
        # once is not a rule worth holding anything to.
        if self.api is not None:
            _refuse_unknown_compat(self.compat, self.api, f"model {self.id!r}")
        return self


class ProviderEntry(Base):
    """One provider, in pi's ``models.json`` shape.

    Two kinds, and ``base_url`` is what tells them apart. An entry with no
    address is one of pi's own: pi has the address, the wire and the catalogue,
    so the entry carries the credential and nothing else has to be said. An
    entry with an address is a provider this config **declares** -- a relay, a
    self-hosted server, an Azure resource -- and an address needs a protocol, so
    ``api`` is required with it and refused without it. That pairing is the whole
    rule: there is no per-vendor table saying which wire a name speaks, and
    nothing is probed.

    A key pi does not ship can only be the declared kind, so it needs both.

    ``login`` is the credential that is a sign-in rather than a key. The grant
    lives in the model service's credential store, never in this file, so an
    entry that names one carries no ``api_key``.

    An entry with neither ``login`` nor ``api_key`` is still a declaration: pi
    reads the vendor's own environment variable itself (``OPENAI_API_KEY`` and
    the rest), and a self-hosted server usually wants no key at all.
    """

    login: Literal["oauth"] | None = None
    api_key: str = Field(default="", json_schema_extra={"secret": True})
    base_url: str = ""
    api: str | None = None
    #: Display name. pi defaults it to the id; a declared provider is the only
    #: kind worth naming, since a built-in's name is pi's own.
    name: str = ""
    #: Sent with every request to this provider -- can carry a secret, so
    #: display faces redact the values and leave the keys visible.
    headers: dict[str, str] | None = Field(default=None, json_schema_extra={"secret": True})
    #: pi's compatibility overrides for every model this entry serves, in pi's
    #: own shape (:data:`COMPAT_KEYS`). What pi detects from an address is right
    #: for the hosts it has rules for and a guess for everyone else, and this is
    #: where the operator of a relay states what theirs actually accepts. Only a
    #: declared entry has one: pi owns its own providers' flags, and an entry
    #: with no address sends nothing but a credential.
    compat: dict[str, object] | None = None
    #: A bare id, or a full row. A built-in's list curates what the picker
    #: offers out of pi's catalogue; a declared provider's list IS its
    #: catalogue, because pi has none for one.
    models: list[str | ModelEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _an_address_names_its_protocol(self) -> "ProviderEntry":
        if self.api is not None and self.api not in pi_ids.APIS:
            raise ValueError(f"api {self.api!r} is not one of {', '.join(pi_ids.APIS)}")
        if self.base_url and not self.api:
            raise ValueError(
                "baseUrl is set but api is not: an address needs the protocol it serves, declared and "
                f"never probed. One of {', '.join(pi_ids.APIS)}."
            )
        if self.api and not self.base_url:
            raise ValueError(
                f"api is {self.api!r} but baseUrl is not set: naming a protocol is only meaningful for a "
                "provider this config declares, and a declared one is reached by address."
            )
        return self

    @model_validator(mode="after")
    def _compat_is_read_by_the_wire_it_is_written_under(self) -> "ProviderEntry":
        """Every ``compat`` block here, against the one wire that will read it.

        A row's wire is its own ``api`` or, far more often, this provider's, so
        this is where a row's block is finally checked -- from inside the row the
        provider is not visible.

        An entry with no address carries no block at all. pi detects the flags
        for its own providers, and such an entry sends nothing but a credential,
        so a block written there would be accepted and never sent anywhere.
        """
        # A declared entry always names its wire: `api` is required with
        # `baseUrl`, so a row that names none falls back to one, never to None.
        wire = self.api or ""
        blocks = [("this provider", self.compat, wire)]
        blocks += [(f"model {row.id!r}", row.compat, row.api or wire) for row in self.rows()]
        for subject, compat, api in blocks:
            if not compat:
                continue
            if not self.declared:
                raise ValueError(
                    f"{subject} sets compat, and this entry declares no baseUrl: pi detects the "
                    "compatibility of its own providers from the address it holds, and an entry with no "
                    "address sends nothing but its credential. Set baseUrl and api to declare the "
                    "provider, or delete the compat block."
                )
            _refuse_unknown_compat(compat, api, subject)
        return self

    @property
    def declared(self) -> bool:
        """Is this a provider this config declares, rather than one of pi's?

        The address is what decides. A built-in id carrying one is declared too:
        an Azure resource URL with its own deployment names is not something
        pi's catalogue can have, so it goes over as a declaration under that
        same id and replaces pi's built-in for it.
        """
        return bool(self.base_url)

    @property
    def model_ids(self) -> list[str]:
        """Every model id this entry declares, bare as the endpoint serves it."""
        return [m if isinstance(m, str) else m.id for m in self.models]

    def rows(self) -> list[ModelEntry]:
        """Every declared model as a row; a bare id becomes a row of just an id."""
        return [ModelEntry(id=m) if isinstance(m, str) else m for m in self.models]

    def row(self, model: str) -> ModelEntry | None:
        """The declared row for one bare model id, or None if it is not declared.

        None for a model declared as a bare string too: such an entry states
        the id and nothing else, and this is asked for what the user knows
        beyond it.
        """
        for entry in self.models:
            if isinstance(entry, ModelEntry) and entry.id == model:
                return entry
        return None


class ProvidersConfig(RootModel[dict[str, ProviderEntry]]):
    """The ``providers`` section: pi provider ids to pi provider declarations.

    This is pi's own ``models.json`` shape, so pi's documentation describes this
    section and there is no translation to keep in step. A key is a pi provider
    id -- ``anthropic``, ``openai-codex``, ``openrouter``, ``azure-openai-responses``
    -- or a name this config invents for a provider it declares outright, and
    the id is written exactly once, here, with no aliases and no spellings to
    reconcile.

    What a key may be is checked here rather than left to the model service:
    the config is read on machines with no Node at all (``ddeharness doctor``),
    and a key that is a near-miss for a built-in -- ``gemini`` for ``google``,
    ``azure-openai`` for ``azure-openai-responses`` -- would otherwise be
    accepted as a declared provider and then refused for having no address.
    """

    root: dict[str, ProviderEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _each_entry_is_reachable(self) -> "ProvidersConfig":
        for provider, entry in self.root.items():
            if not provider or "/" in provider or provider != provider.strip():
                raise ValueError(f"providers.{provider!r} is not a provider id")
            if pi_ids.is_builtin(provider) or entry.declared:
                continue
            meant = pi_ids.suggestion(provider)
            hint = f" Did you mean {meant!r}?" if meant else ""
            raise ValueError(
                f"providers.{provider} is not one of pi's built-in providers, so it is a provider this "
                f"config declares -- and a declared one needs baseUrl and api.{hint} Built-in ids are: "
                f"{', '.join(sorted(pi_ids.BUILTIN))}."
            )
        return self

    # -- mapping face. Callers hold this object, not the dict inside it.
    def get(self, provider: str | None) -> ProviderEntry | None:
        return self.root.get(provider or "")

    def keys(self):  # noqa: ANN201 - a dict view, typed by the dict
        return self.root.keys()

    def values(self):  # noqa: ANN201
        return self.root.values()

    def items(self):  # noqa: ANN201
        return self.root.items()

    def __getitem__(self, provider: str) -> ProviderEntry:
        return self.root[provider]

    def __contains__(self, provider: object) -> bool:
        return provider in self.root

    def __iter__(self):  # noqa: ANN204 - iterates keys, like a dict
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)


class WebToolsConfig(Base):
    """Web tools configuration."""

    proxy: str | None = None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"
    jina_api_key: str = ""  # Jina Reader API key (optional; raises its rate limit)
    # Brave Search API key (optional). With one, web_search queries Brave
    # first (free tier: 2,000 queries a month); without one it queries
    # DuckDuckGo, which needs no key. See ``agent.tools.web``.
    brave_api_key: str = ""


class ExecToolConfig(Base):
    """Configuration for the shell tool (advertised to the model as ``bash``)."""

    timeout: int = 60
    path_append: str = ""
    # Extra regex deny-patterns appended to ExecTool's built-in destructive-command
    # defaults. Empty by default. Operators (or eval harnesses running the agent
    # un-sandboxed) can add host-specific blocks, e.g. osascript / `open -a`.
    extra_deny_patterns: list[str] = Field(default_factory=list)


class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # auto-detected if omitted
    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    url: str = ""  # HTTP/SSE: endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE: custom headers
    tool_timeout: int = 30  # seconds before a tool call is cancelled


class ToolSearchConfig(Base):
    """Progressive tool disclosure.

    When the live tool catalog (built-ins + plugins + MCP) grows past
    ``compaction_threshold``, most tool schemas are withheld from each request and reached
    on demand through the ``tool_search`` / ``tool_call`` meta-tools, so context
    cost stops scaling with tool count and the per-turn tool list (and thus the
    prompt cache) stays stable. At or below the threshold every tool is exposed
    directly (unchanged behavior) and the meta-tools are omitted.
    """

    enabled: bool = False
    compaction_threshold: int = 50
    """Tool-catalog size that triggers compaction: at or below this many tools
    everything is exposed directly; above it, schemas are withheld."""
    search_result_limit: int = 10
    """Default number of hits ``tool_search`` returns per query."""
    always_visible: list[str] = Field(default_factory=list)
    """Extra tool names kept exposed every turn, on top of the core set."""


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # If true, restrict all tool access to workspace directory
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    tool_search: ToolSearchConfig = Field(default_factory=ToolSearchConfig)
    disabled_tools: list[str] = Field(default_factory=list)
    """Tool names to unregister after default-tool registration and MCP connect.
    Used by eval harnesses (e.g. BrowseComp-Plus) that need to constrain the
    agent to a specific tool subset. Names match those in ``ToolRegistry``
    (e.g. ``read``, ``web_search``, or ``mcp_bcp-search_search``)."""


class CliConfig(Base):
    """CLI surface configuration."""

    turn_summary: bool = True
    """Render a one-line tokens/cost summary after each successful CLI turn."""


#: The shape of ``config.json``, as a number that goes up by one whenever a
#: field is added, removed or renamed anywhere under :class:`Config`.
#:
#: It is what tells ``ddeharness onboard`` whether the configuration on disk is
#: one this build understands. A release that does not touch the shape leaves
#: it alone and the wizard reuses what is there; a release that does moves the
#: old file aside and starts clean, which is the rule this program has always
#: had -- nothing is migrated. ``tests/test_config_schema_version.py`` fails
#: when the shape moves and this does not, so it cannot be forgotten.
CONFIG_SCHEMA_VERSION = 1


class Config(BaseSettings):
    """Root configuration for opendde_harness: the base agent blocks plus the
    feature blocks (:mod:`opendde_harness.config.features`), all parsed from
    the one ``config.json``."""

    #: Which shape this file was written in; see :data:`CONFIG_SCHEMA_VERSION`.
    schema_version: int = CONFIG_SCHEMA_VERSION
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    tui: TuiConfig = Field(default_factory=TuiConfig)
    cli: CliConfig = Field(default_factory=CliConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    # UI language chosen during onboarding. Drives the wizard/CLI copy and the
    # agent's reply language (injected into the system prompt). "en" | "zh".
    language: Literal["en", "zh"] = "en"

    context: ContextConfig = Field(default_factory=ContextConfig)
    # SkillForge subsystem; its RRF routing policy nests at ``skill_forge.router``.
    skill_forge: SkillForgeConfig = Field(default_factory=SkillForgeConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    @model_validator(mode="after")
    def _default_model_names_a_provider(self) -> "Config":
        """The default model's prefix names a provider that could serve it.

        Configured, or one pi ships. Both count because an entry carrying
        nothing is still a declaration -- pi reads the vendor's own environment
        variable itself -- and because a fresh config names a built-in before
        anyone has written a key. A prefix that is neither is a typo, and
        saying so here is the difference between a named error and every
        request failing on a provider that was never built.
        """
        if self.agents.defaults.model == "":
            return self  # not chosen yet; the setup gate parks on it
        provider = model_id.provider_of(self.agents.defaults.model)
        if provider in self.providers or pi_ids.is_builtin(provider):
            return self
        raise ValueError(
            f"agents.defaults.model names provider {provider!r}, which is neither a key of the "
            f"providers section nor one of pi's built-in providers. Run `ddeharness onboard`, or "
            f"declare providers.{provider} with its baseUrl and api."
        )

    def _match_provider(self, model: str | None = None) -> tuple["ProviderEntry | None", str | None]:
        """The entry that serves ``model``, and the provider id it is filed under.

        The provider is part of the model's identity and the id's own prefix is
        what names it. A bare id names nobody and gets ``(None, None)`` -- the
        caller refuses it (see :meth:`explain_unrouted`) rather than guessing
        from the spelling, which is how a request and a key went to a vendor the
        user never chose for that model. A prefixed id naming an entry that is
        not configured is the same answer: nothing else is offered in its place,
        because a gateway that happens to hold a key is not the provider the id
        names.
        """
        provider = model_id.provider_of(model or self.agents.defaults.model)
        if not provider:
            return None, None
        entry = self.providers.get(provider)
        return (entry, provider) if entry is not None else (None, None)

    def explain_unrouted(self, model: str | None = None) -> str:
        """Why ``model`` has no provider, as the sentence to show the user.

        Only meaningful after :meth:`_match_provider` answered nothing. A bare
        id is told how to name a provider; a prefixed one is told which entry it
        names, so the credential report can be about that entry.
        """
        wanted = model or self.agents.defaults.model
        provider, bare = model_id.split(wanted)
        if provider:
            return f"{wanted!r} names providers.{provider}, which is not configured"
        return (
            f"model {wanted!r} names no provider. Write it as <provider>/{bare} "
            f"(for example openrouter/{bare}): ddeharness provider use <provider>/{bare}"
        )

    def get_provider(self, model: str | None = None) -> ProviderEntry | None:
        """The providers entry that serves this model, or None."""
        entry, _ = self._match_provider(model)
        return entry

    def get_provider_name(self, model: str | None = None) -> str | None:
        """The pi provider id serving this model (e.g. "deepseek", "openrouter")."""
        _, provider = self._match_provider(model)
        return provider

    def get_api_key(self, model: str | None = None) -> str | None:
        """The key configured for this model's provider, if the config holds one.

        Empty for a provider reached by a sign-in or by the vendor's own
        environment variable: pi resolves both itself, and neither is in
        this file.
        """
        entry = self.get_provider(model)
        return entry.api_key if entry else None

    def get_api_base(self, model: str | None = None) -> str | None:
        """The address configured for this model's provider, if it declares one.

        None for a built-in: pi carries its address, and inventing one here
        would be this project holding a second copy of a fact pi already has.
        """
        entry = self.get_provider(model)
        return (entry.base_url or None) if entry else None

    # The file spells the feature blocks camelCase (``skillForge``); both
    # spellings load, and dumps use the camelCase alias like every nested block.
    model_config = SettingsConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        env_prefix="OPENDDE_HARNESS_",
        env_nested_delimiter="__",
        extra="forbid",
    )


# Written to a fresh file by the onboarding wizard (``config.update``) and
# patched in place afterwards; ``save_config`` leaves them out so a bootstrap
# file stays the shape the wizard seeds.
FEATURE_FIELDS = frozenset({"context", "skill_forge", "runtime", "tracing", "plugins", "memory"})
