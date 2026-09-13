"""pi's own provider ids and api ids, as this project's config is written in.

The ``providers`` section of ``config.json`` is pi's ``models.json`` shape: its
keys are pi provider ids, its values are pi provider declarations. So a key is
one of two things, and this module is what tells them apart:

* a **built-in** -- a provider pi ships (:data:`BUILTIN`). It already has its
  address, its wire and its catalogue; all a config entry adds is the
  credential, and a key it does not carry is not reachable that way;
* anything else is a **declared** provider, and then ``baseUrl`` and ``api``
  are required, because nothing else can say where it is or what it speaks.

The tables here are a snapshot of what pi-ai ships, taken from
``builtinProviders()`` and from the model service's own ``APIS`` map. They are
static because the config is validated offline -- before any service is
started, and by ``ddeharness doctor`` on a machine with no Node at all. The
service is still the authority at run time: a built-in id this table has not
caught up with is refused here, and an ``api`` the service does not implement
is refused there, by name, with the list it does implement.

Regenerate with::

    node --input-type=module -e "
    import {builtinProviders} from './node_modules/@earendil-works/pi-ai/dist/providers/all.js';
    for (const p of builtinProviders()) console.log(p.id, p.name, Object.keys(p.auth))"

run from ``ui-tui/``.
"""

from __future__ import annotations

#: pi's built-in providers: id -> the name pi displays for it. Ordered as
#: ``builtinProviders()`` returns them, which is alphabetical by id.
BUILTIN: dict[str, str] = {
    "amazon-bedrock": "Amazon Bedrock",
    "ant-ling": "Ant Ling",
    "anthropic": "Anthropic",
    "azure-openai-responses": "Azure OpenAI",
    "baseten": "Baseten",
    "cerebras": "Cerebras",
    "cloudflare-ai-gateway": "Cloudflare AI Gateway",
    "cloudflare-workers-ai": "Cloudflare Workers AI",
    "deepseek": "DeepSeek",
    "fireworks": "Fireworks",
    "google": "Google",
    "google-vertex": "Google Vertex AI",
    "groq": "Groq",
    "huggingface": "Hugging Face",
    "kimi-coding": "Kimi For Coding",
    "minimax": "MiniMax",
    "minimax-cn": "MiniMax CN",
    "mistral": "Mistral",
    "moonshotai": "Moonshot AI",
    "moonshotai-cn": "Moonshot AI CN",
    "nvidia": "NVIDIA",
    "openai": "OpenAI",
    "openai-codex": "OpenAI Codex",
    "opencode": "OpenCode Zen",
    "opencode-go": "OpenCode Go",
    "openrouter": "OpenRouter",
    "qwen-token-plan": "Qwen Token Plan",
    "qwen-token-plan-cn": "Qwen Token Plan CN",
    "qwen-token-plan-individual": "Qwen Token Plan Individual",
    "radius": "Radius",
    "together": "Together",
    "vercel-ai-gateway": "Vercel AI Gateway",
    "xai": "xAI",
    "xiaomi": "Xiaomi",
    "xiaomi-token-plan-ams": "Xiaomi Token Plan AMS",
    "xiaomi-token-plan-cn": "Xiaomi Token Plan CN",
    "xiaomi-token-plan-sgp": "Xiaomi Token Plan SGP",
    "zai": "Z.AI",
    "zai-coding-cn": "Z.AI Coding CN",
}

#: The built-ins whose credential is a sign-in rather than a key, so
#: ``login: "oauth"`` is a thing to write under them. pi's ``github-copilot``
#: also offers one and is deliberately absent: see :data:`REMOVED`.
OAUTH: frozenset[str] = frozenset({"anthropic", "kimi-coding", "openai-codex", "openrouter", "radius", "xai"})

#: The wires the model service implements, which is the set an entry's ``api``
#: may name (``ui-tui/src/model-service/configure.ts``'s ``APIS``). Kept here so
#: a config naming one pi has but the service does not is refused where the
#: config is read, rather than at the first request.
APIS: tuple[str, ...] = (
    "anthropic-messages",
    "azure-openai-responses",
    "google-generative-ai",
    "mistral-conversations",
    "openai-codex-responses",
    "openai-completions",
    "openai-responses",
)

#: The environment variables pi reads a built-in's key from, mirrored from
#: pi-ai's own ``getApiKeyEnvVars``. Never used to *send* a key: pi resolves the
#: environment itself, and a copy of a key travelling through our payload would
#: be a second answer to a question pi already answers. Read only to report --
#: "this provider has no key in the config, but ANTHROPIC_API_KEY is set" is a
#: different status from "this provider has no key at all", and status, doctor
#: and the setup panel all have to say which.
ENV_KEYS: dict[str, tuple[str, ...]] = {
    "ant-ling": ("ANT_LING_API_KEY",),
    "anthropic": ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"),
    "azure-openai-responses": ("AZURE_OPENAI_API_KEY",),
    "baseten": ("BASETEN_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "cloudflare-ai-gateway": ("CLOUDFLARE_API_KEY",),
    "cloudflare-workers-ai": ("CLOUDFLARE_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "fireworks": ("FIREWORKS_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "google-vertex": ("GOOGLE_CLOUD_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "huggingface": ("HF_TOKEN",),
    "kimi-coding": ("KIMI_API_KEY",),
    "minimax": ("MINIMAX_API_KEY",),
    "minimax-cn": ("MINIMAX_CN_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "moonshotai": ("MOONSHOT_API_KEY",),
    "moonshotai-cn": ("MOONSHOT_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "opencode": ("OPENCODE_API_KEY",),
    "opencode-go": ("OPENCODE_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "qwen-token-plan": ("QWEN_TOKEN_PLAN_API_KEY",),
    "qwen-token-plan-cn": ("QWEN_TOKEN_PLAN_CN_API_KEY",),
    "qwen-token-plan-individual": ("QWEN_TOKEN_PLAN_API_KEY",),
    "radius": ("RADIUS_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "vercel-ai-gateway": ("AI_GATEWAY_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "xiaomi": ("XIAOMI_API_KEY",),
    "xiaomi-token-plan-ams": ("XIAOMI_TOKEN_PLAN_AMS_API_KEY",),
    "xiaomi-token-plan-cn": ("XIAOMI_TOKEN_PLAN_CN_API_KEY",),
    "xiaomi-token-plan-sgp": ("XIAOMI_TOKEN_PLAN_SGP_API_KEY",),
    "zai": ("ZAI_API_KEY",),
    "zai-coding-cn": ("ZAI_CODING_CN_API_KEY",),
}

#: Built-ins whose credential the environment supplies as a whole chain rather
#: than as one variable. Reported as configured when the chain is there, and
#: never asked for a key -- the user does not have one in that form.
AMBIENT: frozenset[str] = frozenset({"amazon-bedrock", "google-vertex"})

#: Built-ins pi ships the id of but cannot ship the provider for. An Azure
#: endpoint is one tenant's own address serving deployment names only they
#: have, so what goes under this id is a declaration -- an address, the wire and
#: the deployments as its models, written with ``ddeharness provider set`` --
#: and a key alone reaches nothing. A picker that offers it for a key offers a
#: row that cannot be finished, so the pickers leave it out until it is declared.
DECLARED_BUILTINS: frozenset[str] = frozenset({"azure-openai-responses"})

#: Names people type that are not pi ids, and the pi id each one means. A
#: near-miss is a dead end otherwise: it is not a built-in, so it reads as a
#: provider this config declares, and it is then refused for having no address
#: -- which is a true sentence about the wrong problem.
NEAR_MISS: dict[str, str] = {
    "azure": "azure-openai-responses",
    "azure_openai": "azure-openai-responses",
    "azure-openai": "azure-openai-responses",
    "bedrock": "amazon-bedrock",
    "chatgpt": "openai-codex",
    "claude": "anthropic",
    "cloudflare": "cloudflare-workers-ai",
    "codex": "openai-codex",
    "gemini": "google",
    "google-gemini": "google",
    "kimi": "moonshotai",
    "minimax-global": "minimax",
    "moonshot": "moonshotai",
    "qwen": "qwen-token-plan",
    "vertex": "google-vertex",
    "vertex_ai": "google-vertex",
    "vertex-ai": "google-vertex",
    "zhipu": "zai",
}


def suggestion(provider: str | None) -> str | None:
    """The pi id this near-miss means, or None when the name is not one."""
    return NEAR_MISS.get((provider or "").strip())


#: What a declared provider speaks unless it says otherwise. Chat Completions
#: is what an arbitrary relay or a self-hosted server implements.
DEFAULT_API = "openai-completions"

#: Providers this project supported and removed, with what to tell whoever
#: still has one configured. pi carries ``github-copilot``, so dropping our own
#: section is not enough to refuse it: without this entry the pi id would be
#: accepted as any other built-in. A removal is not a rename -- nothing is
#: folded into another entry, because the credential, the endpoint and the
#: models all belonged to the product that went away.
REMOVED: dict[str, str] = {
    "github-copilot": (
        "GitHub Copilot support was removed from OpenDDE Harness. Delete the providers.github-copilot "
        "entry, and any github-copilot/... model ids, from your config; there is no replacement to move "
        "them to. Run `ddeharness onboard` to pick another provider."
    ),
}

#: Spellings of a removed provider that are not its pi id, so a config or a
#: command line naming one is told about the removal rather than read as a
#: provider nobody has ever heard of.
_REMOVED_ALIASES: dict[str, str] = {
    "copilot": "github-copilot",
    "github_copilot": "github-copilot",
    "githubCopilot": "github-copilot",
}


def is_builtin(provider: str) -> bool:
    """Does pi ship this provider? Exact id, because pi ids have no spellings."""
    return provider in BUILTIN


def display_name(provider: str, declared: str = "") -> str:
    """What to show for this provider: its own name, else pi's, else the id."""
    return declared or BUILTIN.get(provider, "") or provider


def removed_message(provider: str | None) -> str | None:
    """Why this name no longer names a provider, or ``None`` if it still does.

    Asked wherever a provider name arrives -- the config file, the write paths,
    a model id on its way to the service -- so naming the removed product is
    answered with the removal and never with "no such section".
    """
    name = (provider or "").strip()
    return REMOVED.get(_REMOVED_ALIASES.get(name, name))


def refuse_removed(*providers: str | None) -> None:
    """Stop here if any of these names a provider that was removed.

    Nothing is constructed and nothing is sent: there is no configuration that
    makes this work, so the removal message is the whole answer.
    """
    for provider in providers:
        message = removed_message(provider)
        if message:
            raise ValueError(message)


__all__ = [
    "AMBIENT",
    "APIS",
    "BUILTIN",
    "DECLARED_BUILTINS",
    "ENV_KEYS",
    "NEAR_MISS",
    "DEFAULT_API",
    "OAUTH",
    "REMOVED",
    "display_name",
    "is_builtin",
    "refuse_removed",
    "removed_message",
    "suggestion",
]
