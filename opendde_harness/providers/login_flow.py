"""pi's ``/login``, described once for the terminal side.

The TUI's ``/login`` (``ui-tui/src/selectors/authSelector.ts`` and
``modelStages.ts``) is pi's own sign-in, step for step and string for string:
the authentication method, then the provider, then the sign-in or the key --
with one addition of ours, an OpenAI-compatible endpoint under the key method.
The setup wizard's first step is the same flow at a terminal, and this module
is where its words and its rules live, so the wizard cannot drift from the TUI
by editing a string of its own.

TypeScript cannot import this and Python cannot import the selector, so the
English column here is restated from the TUI's sources, and
``tests/test_login_flow.py`` reads those sources and fails the moment one side
changes a word the other still shows. The Chinese column is the wizard's own:
the TUI speaks English only.

The rules below are the TUI's, in the same order the TUI applies them:
:func:`status_marker` is pi's own marker for a provider row
(``authSelector.statusMarker``), :func:`login_methods` is which ways in a row
offers (``authSelector.loginPairs``), and :func:`offered` is which rows the
wizard puts in front of a first run.
"""

from __future__ import annotations

from typing import Any, Sequence

from opendde_harness.providers.auth import CRED_KEY, CRED_OAUTH

#: An English sentence and its Chinese one, in that order. The wizard picks.
Text = tuple[str, str]

# ── pi's own two doors, in pi's own order: the subscription first ──────────

#: pi's generic label for the subscription option, where the provider carries
#: no ``loginLabel`` of its own.
SIGN_IN_WITH_ACCOUNT: Text = ("Sign in with an account", "使用账号登录")
#: pi's label for the key option. Not per provider: pi uses this one string.
SIGN_IN_WITH_API_KEY: Text = ("Sign in with an API key", "使用 API Key 登录")

#: The methods as the gateway's rows spell them, each with pi's label.
METHODS: tuple[tuple[str, Text], ...] = ((CRED_OAUTH, SIGN_IN_WITH_ACCOUNT), (CRED_KEY, SIGN_IN_WITH_API_KEY))

# ── the titles of each stage, pi's ─────────────────────────────────────────

SELECT_METHOD: Text = ("Select authentication method:", "选择登录方式：")
SELECT_PROVIDER: Text = ("Select provider to configure:", "选择要配置的服务商：")
#: pi titles the method step with the provider's name once it knows which one.
SELECT_METHOD_FOR: Text = ("Select authentication method for {name}:", "选择 {name} 的登录方式：")
SIGN_IN_TO: Text = ("Sign in to {name}", "登录 {name}")
CONNECT: Text = ("Connect {name}", "连接 {name}")
ADD_ENDPOINT: Text = ("Add an OpenAI-compatible endpoint", "添加 OpenAI 兼容端点")

#: What each stage says under its title.
SIGN_IN_SUBTITLE: Text = (
    "this signs in for the whole machine, not just this conversation",
    "登录对整台机器生效，不只是这一次对话",
)
KEY_SUBTITLE: Text = ("the key is stored by the gateway, for this provider", "Key 由网关保存，仅用于该服务商")
KEY_SUBTITLE_REPLACES: Text = (
    "this provider already has a key; saving replaces it",
    "该服务商已有 Key，保存会替换它",
)
ENDPOINT_SUBTITLE: Text = (
    "a relay, a gateway or a server you run yourself, speaking Chat Completions — like OpenRouter, "
    "with your address and key",
    "中转、网关或自建服务，讲 Chat Completions —— 像 OpenRouter 一样，填你的地址和 Key",
)

# ── what each stage settles ────────────────────────────────────────────────

#: pi's own sentence once a credential is stored.
LOGGED_IN: Text = ("Logged in to {name}", "已登录 {name}")
#: The endpoint's own version, with what it was found to serve.
LOGGED_IN_ENDPOINT: Text = (
    "Logged in to {name} ({count} model{plural} from {url})",
    "已登录 {name}（{url} 提供 {count} 个模型）",
)
#: pi's own sentence for a sign-in that did not finish, with the reason after it.
FAILED_LOGIN: Text = ("Failed to login to {name}: {why}", "登录 {name} 失败：{why}")

# ── the key form ───────────────────────────────────────────────────────────

API_KEY: Text = ("API key", "API Key")
KEY_HINT_ENV: Text = ("{env} is already set — leave this blank to use it", "{env} 已设置 —— 留空即使用它")
KEY_HINT_REQUIRED: Text = ("required for this provider", "该服务商必填")
KEY_HINT_OPTIONAL: Text = ("optional for this provider", "该服务商可不填")
KEY_REQUIRED: Text = ("this provider needs an API key", "该服务商需要 API Key")
KEY_REFUSED: Text = ("the gateway did not accept that credential", "网关没有接受这份凭据")
CONTROL_CHARACTERS: Text = ("that value contains control characters", "这个值含有控制字符")

# ── the endpoint form ──────────────────────────────────────────────────────

#: The provider list's last row under the key method: ours, not pi's.
ENDPOINT_ROW: Text = ("OpenAI Compatible", "OpenAI Compatible")

PROVIDER_ID: Text = ("Provider id", "服务商 id")
PROVIDER_ID_HINT: Text = (
    "what the entry is filed under, and the prefix of every model id it serves",
    "配置项的键名，也是它所有模型 id 的前缀",
)
PROVIDER_ID_PLACEHOLDER = "my-vllm"
BASE_URL: Text = ("Base URL", "Base URL")
BASE_URL_HINT: Text = ("where the endpoint lives", "端点的地址")
BASE_URL_PLACEHOLDER = "http://127.0.0.1:8000/v1"
ENDPOINT_KEY_HINT: Text = (
    "optional: a server you run yourself usually wants none",
    "可不填：自建服务通常不需要",
)
MODEL_IDS: Text = ("Model ids", "模型 id")
MODEL_IDS_HINT: Text = (
    "optional: leave empty to read the endpoint’s own list; comma-separated otherwise",
    "可不填：留空则读取端点自己的列表；多个用英文逗号分隔",
)
MODEL_IDS_PLACEHOLDER: Text = ("discovered from GET /models", "从 GET /models 读取")

# ── the sign-in's own lines, pi's ──────────────────────────────────────────

LOADING_PROVIDERS: Text = ("loading providers…", "正在读取服务商…")
STARTING_SIGN_IN: Text = ("starting the sign-in…", "正在开始登录…")
WORKING: Text = ("working…", "处理中…")
ENTER_CODE: Text = ("Enter code: {code}", "输入验证码：{code}")
WAITING_FOR_AUTHENTICATION: Text = ("Waiting for authentication...", "等待授权完成...")
CLICK_TO_OPEN: Text = ("Ctrl+click to open", "Ctrl+点击打开")

# ── pi's status marker for one row ─────────────────────────────────────────

UNCONFIGURED: Text = ("• unconfigured", "• 未配置")
CONFIGURED: Text = ("✓ configured", "✓ 已配置")
CONFIGURED_ENV: Text = ("✓ env: {env}", "✓ 环境变量：{env}")
SUBSCRIPTION_CONFIGURED: Text = ("• subscription configured", "• 已配置订阅")
API_KEY_CONFIGURED: Text = ("• API key configured", "• 已配置 API Key")


def status_marker(row: dict[str, Any], method: str) -> Text:
    """pi's own marker for one provider row under one method.

    The words are pi's (``oauth-selector.formatStatusIndicator``): nothing
    stored is unconfigured, a credential of this very method is configured (or
    the variable that supplies it), and a provider configured by its *other*
    method says which -- signing in to a provider that already holds a key is a
    real thing to do, and the row has to say what is there now.
    """
    if not row.get("authenticated"):
        return UNCONFIGURED
    if row.get("auth_type") != method:
        return SUBSCRIPTION_CONFIGURED if row.get("auth_type") == CRED_OAUTH else API_KEY_CONFIGURED
    env = row.get("key_env")
    if env:
        return CONFIGURED_ENV[0].format(env=env), CONFIGURED_ENV[1].format(env=env)
    return CONFIGURED


def login_methods(row: dict[str, Any]) -> list[str]:
    """Which ways in this row offers, in pi's order.

    ``auth_methods`` is pi's own list, read off its provider objects by the
    model service. A gateway that could not ask it sends none, and then the
    row's own ``auth_type`` decides -- the one way in the gate would judge a
    submission by -- so a machine whose model service is not runnable yet can
    still be handed a key.
    """
    methods = [method for method in row.get("auth_methods") or () if method in (CRED_OAUTH, CRED_KEY)]
    if methods:
        return methods
    own = row.get("auth_type")
    return [own] if own in (CRED_OAUTH, CRED_KEY) else []


#: The built-ins a first run is offered, in this order. The TUI's list is
#: every provider pi ships, forty rows behind a filter; a wizard screen is read
#: top to bottom, and forty rows there bury the handful almost everybody picks.
#: So the wizard offers these and the endpoint row, and says where the rest
#: are: ``/login`` in the TUI lists them all.
FEATURED: tuple[str, ...] = (
    "openai-codex",
    "openai",
    "anthropic",
    "google",
    "openrouter",
    "deepseek",
    "moonshotai",
    "zai",
    "xai",
)


def offered(rows: Sequence[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    """The featured rows that offer ``method``, in the featured order."""
    return [row for slug in FEATURED for row in rows if row.get("slug") == slug and method in login_methods(row)]


__all__ = [
    "ADD_ENDPOINT",
    "CONNECT",
    "ENDPOINT_ROW",
    "FAILED_LOGIN",
    "FEATURED",
    "LOGGED_IN",
    "METHODS",
    "SELECT_METHOD",
    "SELECT_METHOD_FOR",
    "SELECT_PROVIDER",
    "SIGN_IN_TO",
    "SIGN_IN_WITH_ACCOUNT",
    "SIGN_IN_WITH_API_KEY",
    "Text",
    "login_methods",
    "offered",
    "status_marker",
]
