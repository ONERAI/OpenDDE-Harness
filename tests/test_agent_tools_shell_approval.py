"""What the shell tool says when an approval does not come back approved.

Reporting every refusal as "denied or the request expired" told the model, and
whoever read the transcript, that the UI could not tell the two apart -- when
the broker had recorded exactly which one happened. These tests pin one message
per outcome.

No command is ever executed here: every case ends before the executor runs.
"""

import pytest

from opendde_harness.agent.tools.approval import ApprovalDecision
from opendde_harness.agent.tools.shell import ExecTool

DELETE_COMMAND = "rm /tmp/opendde-approval-fixture"


class _ScriptedResponder:
    """An approval transport that answers with a fixed decision."""

    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.calls = 0

    async def await_approval(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        tool_call_id: str,
        command: str,
        description: str,
    ) -> ApprovalDecision:
        self.calls += 1
        return self.decision


def _tool(decision: ApprovalDecision) -> tuple[ExecTool, _ScriptedResponder]:
    tool = ExecTool()
    responder = _ScriptedResponder(decision)

    tool.start_approval_turn(responder, conversation_id="tui:test", turn_id="turn-1")
    tool.set_tool_call_id("tool-1")
    return tool, responder


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("deny", "Error: User denied this command"),
        ("timeout", "Error: The approval request expired before it was answered"),
        ("cancelled", "Error: The approval request was cancelled before it was answered"),
        ("error", "Error: The approval request could not be completed, so the command did not run"),
    ],
)
async def test_each_refusal_names_what_happened(reason, expected):
    tool, responder = _tool(ApprovalDecision(False, reason))

    result = await tool.execute(command=DELETE_COMMAND)

    assert responder.calls == 1
    assert result.model_text.startswith(expected)
    # Every refusal still stops the loop rather than inviting a retry.
    assert result.retryable is False
    assert result.abort_action is True


async def test_a_denial_does_not_read_as_an_expiry():
    tool, _ = _tool(ApprovalDecision(False, "deny"))

    result = await tool.execute(command=DELETE_COMMAND)

    assert "expired" not in result.model_text


async def test_a_repeat_of_a_refused_command_does_not_claim_it_was_denied():
    # The digest set holds every refusal, not denials alone, so the second
    # message cannot say the user denied anything.
    tool, responder = _tool(ApprovalDecision(False, "timeout"))

    await tool.execute(command=DELETE_COMMAND)
    repeat = await tool.execute(command=DELETE_COMMAND)

    assert responder.calls == 1
    assert repeat.model_text.startswith("Error: This command was already refused earlier in the current turn")


async def test_an_approval_lets_the_command_through(tmp_path):
    tool, responder = _tool(ApprovalDecision(True, "allow"))
    target = tmp_path / "fixture.txt"

    target.write_text("delete me")

    result = await tool.execute(command=f"rm {target}")

    assert responder.calls == 1
    # It reached the executor rather than a refusal message, and did the work.
    assert not target.exists()
    assert not str(result).startswith("Error:")


async def test_a_turn_without_a_responder_cannot_approve():
    tool = ExecTool()

    tool.start_approval_turn(None, conversation_id="tui:test", turn_id="turn-1")

    result = await tool.execute(command=DELETE_COMMAND)

    assert result.model_text.startswith("Error: Command requires user approval")
