"""Reading a session file written before the records were pi messages.

The fixture (``fixtures/session_pre_pi_shape.jsonl``) is a real file in the old
shape: all four roles, an assistant turn with a verbatim ``pi`` copy, a
multimodal user message, a compaction marker, and lifecycle records between
them. Loading it must produce pi messages and must leave the bytes on disk
alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import COMPACTION_KEY
from opendde_harness.session.legacy import read_records
from opendde_harness.session.manager import SessionManager

FIXTURE = Path(__file__).parent / "fixtures" / "session_pre_pi_shape.jsonl"


def _load(tmp_path: Path):
    """The fixture, loaded through the real ``SessionManager``."""
    manager = SessionManager(tmp_path)
    path = manager.sessions_dir / "cli" / "legacy.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    return manager, path, manager.get_or_create("cli:legacy")


def test_every_old_role_is_read_as_the_shape_the_harness_now_writes(tmp_path):
    _, _, session = _load(tmp_path)
    assert [m["role"] for m in session.messages] == [
        "system",
        "user",
        "assistant",
        "toolResult",
        "assistant",
        "user",
        "assistant",
    ]


def test_an_old_assistant_turn_keeps_its_calls_its_reasoning_and_its_ids(tmp_path):
    _, _, session = _load(tmp_path)
    turn = session.messages[2]
    assert msg.text_of(turn) == "reading"
    assert msg.thinking_of(turn) == "the file first"
    assert msg.tool_calls_of(turn) == [msg.tool_call_block("call_1", "read", {"path": "a.txt"})]
    assert turn["stopReason"] == "toolUse"
    # Nothing in the record says which wire wrote it, so pi is told as much.
    assert turn["api"] == msg.REPLAY_API
    assert (turn["id"], turn["turn_id"]) == ("r2", "t1")


def test_an_old_tool_result_takes_its_name_from_the_call_it_answers(tmp_path):
    _, _, session = _load(tmp_path)
    result = session.messages[3]
    assert result["toolCallId"] == "call_1"
    assert result["toolName"] == "read"
    assert result["content"] == [msg.text_block("the file body")]
    assert result["isError"] is False


def test_a_stored_pi_copy_becomes_the_record_verbatim(tmp_path):
    _, _, session = _load(tmp_path)
    answer = session.messages[4]
    assert answer["api"] == "openai-responses"
    assert answer["responseId"] == "resp_1"
    assert answer["content"][0]["thinkingSignature"] == "sig-1"
    assert msg.text_of(answer) == "it says hello"
    assert "pi" not in answer
    assert answer["id"] == "r4"


def test_a_stored_pi_copy_without_a_usage_gains_the_zero_one(tmp_path):
    """The copy was stored without it, and pi reads ``usage.totalTokens`` off
    every message it replays without checking the field is there. Zeroes rather
    than a guess: what the turn cost was not stored beside the copy, and no
    adapter bills a message it is replaying."""
    _, _, session = _load(tmp_path)
    answer = session.messages[4]

    assert answer["usage"] == {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0},
    }
    # A copy that did carry one keeps its own figures.
    kept = read_records([{"role": "assistant", "content": "x", "pi": {**answer, "usage": {"totalTokens": 99}}}])
    assert kept[0]["usage"] == {"totalTokens": 99}


def test_an_old_inline_image_becomes_a_pi_image_block(tmp_path):
    _, _, session = _load(tmp_path)
    assert session.messages[5]["content"] == [
        msg.text_block("and this picture?"),
        msg.image_block("iVBORw0KGgoAAAANSUhEUg==", "image/png"),
    ]


def test_a_compaction_marker_keeps_its_boundary_and_gains_a_text_block(tmp_path):
    _, _, session = _load(tmp_path)
    marker = session.messages[6]
    assert marker[COMPACTION_KEY]["model"] == "openai-codex/gpt-5"
    assert marker[COMPACTION_KEY]["items"][0]["content"][0]["text"] == "the user's own recent words"
    assert msg.text_of(marker).startswith("[Earlier conversation compacted")


def test_the_lifecycle_records_survive_the_read(tmp_path):
    _, _, session = _load(tmp_path)
    assert session.turn_status() == {"t1": "completed"}
    # The tool the journal started was answered, so nothing is uncertain.
    assert session.uncertain_tool_calls() == []


def test_reading_an_old_file_never_rewrites_it(tmp_path):
    manager, path, session = _load(tmp_path)
    before = path.read_text(encoding="utf-8")
    manager.flush("cli:legacy")
    assert path.read_text(encoding="utf-8") == before
    # And a new turn appends to it without touching what was there.
    session.record(msg.user_message("one more"))
    manager.save(session)
    after = path.read_text(encoding="utf-8")
    assert after.startswith(before)
    appended = [json.loads(line) for line in after[len(before) :].splitlines() if line.strip()]
    assert [r.get("role") for r in appended if r.get("_type") != "metadata"] == ["user"]


def test_a_record_already_in_the_new_shape_is_left_exactly_as_it_is():
    records = [
        msg.user_message("hi"),
        {
            "role": "assistant",
            "content": [msg.text_block("there"), msg.tool_call_block("c1", "read", {})],
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "claude-opus-5",
            "stopReason": "toolUse",
            "timestamp": 12,
        },
        msg.tool_result_message("c1", "read", "body"),
    ]
    assert read_records([dict(r) for r in records]) == records


def test_an_old_record_with_no_content_at_all_still_reads(tmp_path):
    """A turn that asked for a tool and said nothing: the calls are the content."""
    converted = read_records(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": ""}}],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "read", "content": ""},
        ]
    )
    assert msg.tool_calls_of(converted[0]) == [msg.tool_call_block("c1", "read", {})]
    assert msg.text_of(converted[0]) == ""
    assert converted[1]["content"] == []
