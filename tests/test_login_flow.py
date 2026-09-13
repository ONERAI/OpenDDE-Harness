"""pi's ``/login`` is described once for the terminal side, and held to the TUI's.

The TUI restates pi's strings in TypeScript and the wizard reads them from
``providers.login_flow``; nothing can import across, so the English column is
checked against the selector's own sources here. A word changed on one side
fails this until the other side says the same.
"""

import re
from pathlib import Path

import pytest

from opendde_harness.providers import login_flow

TUI = Path(__file__).resolve().parent.parent / "ui-tui" / "src"

#: Where the TUI writes each sentence: pi's own strings live in the selector
#: and its stages, the secret's verdict in the input it is typed into.
SOURCES = (
    "selectors/authSelector.ts",
    "selectors/modelStages.ts",
    "components/loginView.ts",
    "components/secretInput.ts",
)


def _texts() -> dict[str, login_flow.Text]:
    return {
        name: value
        for name, value in vars(login_flow).items()
        if name.isupper() and isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, str) for v in value)
    }


def _pattern(english: str) -> re.Pattern[str]:
    """The sentence as a pattern: a ``{field}`` matches whatever the TUI splices there."""
    return re.compile(".*?".join(re.escape(part) for part in re.split(r"\{[a-z_]+\}", english)))


@pytest.mark.parametrize("name", sorted(_texts()))
def test_every_sentence_of_the_flow_is_the_tuis_own(name):
    english, chinese = _texts()[name]
    sources = "\n".join((TUI / source).read_text() for source in SOURCES)

    assert _pattern(english).search(sources), f"{name}: the TUI does not say {english!r}"
    assert chinese, f"{name} has no Chinese sentence"


@pytest.mark.parametrize("name", sorted(_texts()))
def test_a_chinese_sentence_is_punctuated_in_chinese(name):
    """A Chinese line with an ASCII colon, comma or bracket reads as half-translated."""
    _english, chinese = _texts()[name]
    bare = re.sub(r"\{[a-z_]+\}", "", chinese)
    # ASCII punctuation is at home in a code token ("GET /models", "Ctrl+"), not after a Chinese word.
    stray = re.findall(r"[一-鿿][:,;!?()]", bare)
    assert stray == [], f"{name}: {chinese!r} uses ASCII punctuation after a Chinese word"


def test_pis_two_doors_come_in_pis_order():
    assert [method for method, _ in login_flow.METHODS] == ["oauth", "key"]
    assert [label for _, label in login_flow.METHODS] == [
        login_flow.SIGN_IN_WITH_ACCOUNT,
        login_flow.SIGN_IN_WITH_API_KEY,
    ]


def test_the_status_marker_is_pis_own():
    """Nothing stored is unconfigured; this method's credential is configured,
    or the variable supplying it; the other method's credential says which."""
    marker = login_flow.status_marker
    assert marker({"authenticated": False, "auth_type": "key"}, "key") == login_flow.UNCONFIGURED
    assert marker({"authenticated": True, "auth_type": "key", "key_env": None}, "key") == login_flow.CONFIGURED
    assert marker({"authenticated": True, "auth_type": "key", "key_env": "X_KEY"}, "key") == (
        "✓ env: X_KEY",
        "✓ 环境变量：X_KEY",
    )
    assert marker({"authenticated": True, "auth_type": "oauth"}, "key") == login_flow.SUBSCRIPTION_CONFIGURED
    assert marker({"authenticated": True, "auth_type": "key"}, "oauth") == login_flow.API_KEY_CONFIGURED


def test_a_row_the_gateway_could_not_ask_pi_about_offers_its_own_way_in():
    """pi's list when the service answered; the row's own kind when it could not,
    so a machine without a runnable model service can still be handed a key --
    the rule the TUI's list applies too."""
    assert login_flow.login_methods({"auth_methods": ["oauth", "key"], "auth_type": "key"}) == ["oauth", "key"]
    assert login_flow.login_methods({"auth_methods": [], "auth_type": "key"}) == ["key"]
    assert login_flow.login_methods({"auth_methods": [], "auth_type": "oauth"}) == ["oauth"]
    # A provider this config declares is reached by its address; it is not a row of the flow's.
    assert login_flow.login_methods({"auth_methods": [], "auth_type": "endpoint"}) == []


def test_a_first_run_is_offered_the_featured_few_in_the_featured_order():
    rows = [
        {"slug": "zai", "name": "Z.AI", "auth_methods": ["key"]},
        {"slug": "baseten", "name": "Baseten", "auth_methods": ["key"]},
        {"slug": "openai-codex", "name": "OpenAI Codex", "auth_methods": ["oauth"]},
        {"slug": "anthropic", "name": "Anthropic", "auth_methods": ["oauth", "key"]},
        {"slug": "ant-ling", "name": "Ant Ling", "auth_methods": ["key"]},
    ]
    assert [row["slug"] for row in login_flow.offered(rows, "key")] == ["anthropic", "zai"]
    assert [row["slug"] for row in login_flow.offered(rows, "oauth")] == ["openai-codex", "anthropic"]
    assert login_flow.FEATURED == (
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
