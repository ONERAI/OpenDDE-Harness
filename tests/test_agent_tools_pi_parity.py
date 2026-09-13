"""The file, search and shell tools answer to pi's names and pi's schemas.

The point of the alignment is that a prompt, a skill or a habit written for pi's
built-ins works here unchanged, and that only holds if the *advertised* names
and parameter names match pi's to the letter. ``PI_SCHEMAS`` is that letter: it
is transcribed from pi's own tool definitions in
``packages/coding-agent/src/core/tools/{bash,read,write,edit,ls,grep,find}.ts``,
so a drift in either direction fails here rather than in a user's prompt.

The behaviour underneath stays the harness's own -- the permitted-directory
boundary, the standing-instructions approval gate, the result caps -- so the
rest of the file covers the semantics the new schemas introduced: ``edit``
taking several replacements at once, and ``read``'s 1-indexed pagination.
"""

import asyncio
import shutil
from pathlib import Path

import pytest

from opendde_harness.agent.tools.file_search import FindTool, GrepTool
from opendde_harness.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from opendde_harness.agent.tools.shell import ExecTool

# ---------------------------------------------------------------------------
# pi's own definitions, transcribed
# ---------------------------------------------------------------------------

#: ``tool name -> (every property, the required ones)``, in pi's spelling.
#: camelCase included: ``ignoreCase`` and ``oldText``/``newText`` are pi's, and
#: a model that has learned ``ignore_case`` from us would not be calling pi's.
PI_SCHEMAS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "bash": (frozenset({"command", "timeout"}), frozenset({"command"})),
    "read": (frozenset({"path", "offset", "limit"}), frozenset({"path"})),
    "write": (frozenset({"path", "content"}), frozenset({"path", "content"})),
    "edit": (frozenset({"path", "edits"}), frozenset({"path", "edits"})),
    "ls": (frozenset({"path", "limit"}), frozenset()),
    "grep": (
        frozenset({"pattern", "path", "glob", "ignoreCase", "literal", "context", "limit"}),
        frozenset({"pattern"}),
    ),
    "find": (frozenset({"pattern", "path", "limit"}), frozenset({"pattern"})),
}

#: pi's ``edits[]`` item, which has no counterpart anywhere else in the set.
PI_EDIT_ITEM: tuple[frozenset[str], frozenset[str]] = (
    frozenset({"oldText", "newText"}),
    frozenset({"oldText", "newText"}),
)


def _tool(cls, tmp_path: Path):
    return ExecTool() if cls is ExecTool else cls(workspace=tmp_path, allowed_dir=tmp_path)


@pytest.mark.parametrize(
    "cls",
    [ExecTool, ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, GrepTool, FindTool],
)
def test_each_tool_advertises_pis_name_and_pis_parameter_names(cls, tmp_path: Path) -> None:
    tool = _tool(cls, tmp_path)

    assert tool.name in PI_SCHEMAS, f"{cls.__name__} answers to {tool.name!r}, which pi has no tool for"
    properties, required = PI_SCHEMAS[tool.name]
    schema = tool.parameters

    assert schema["type"] == "object"
    assert set(schema["properties"]) == set(properties)
    assert set(schema.get("required") or ()) == set(required)


def test_the_seven_aligned_tools_are_all_covered(tmp_path: Path) -> None:
    """The table is the contract, so nothing in it may go unclaimed."""
    claimed = {
        _tool(cls, tmp_path).name
        for cls in (ExecTool, ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, GrepTool, FindTool)
    }

    assert claimed == set(PI_SCHEMAS)


def test_an_edits_entry_carries_pis_oldtext_and_newtext(tmp_path: Path) -> None:
    item = EditFileTool(workspace=tmp_path, allowed_dir=tmp_path).parameters["properties"]["edits"]["items"]
    properties, required = PI_EDIT_ITEM

    assert item["type"] == "object"
    assert set(item["properties"]) == set(properties)
    assert set(item["required"]) == set(required)


@pytest.mark.parametrize(
    "cls",
    [ExecTool, ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, GrepTool, FindTool],
)
def test_every_parameter_is_described(cls, tmp_path: Path) -> None:
    """A renamed parameter with pi's old description is a worse tool than either."""
    for name, spec in _tool(cls, tmp_path).parameters["properties"].items():
        assert spec.get("description"), f"{name} has no description"


# ---------------------------------------------------------------------------
# edit: several exact replacements, all or nothing
# ---------------------------------------------------------------------------


@pytest.fixture
def edit(tmp_path: Path) -> EditFileTool:
    return EditFileTool(workspace=tmp_path, allowed_dir=tmp_path)


def test_several_edits_land_in_one_call(edit: EditFileTool, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha\nbeta\ngamma\n")

    answer = asyncio.run(
        edit.execute(
            path="a.py",
            edits=[{"oldText": "alpha", "newText": "ALPHA"}, {"oldText": "gamma", "newText": "GAMMA"}],
        )
    )

    assert "replaced 2 block(s)" in answer
    assert (tmp_path / "a.py").read_text() == "ALPHA\nbeta\nGAMMA\n"


def test_edits_are_matched_against_the_original_not_against_each_other(edit: EditFileTool, tmp_path: Path) -> None:
    """The first edit's replacement is the second edit's anchor text.

    Applied one after another, the first turns ``alpha`` into ``beta`` and the
    second then finds ``beta`` twice and refuses. Locating every edit in the
    original up front, then replacing back to front, is what lets the model
    name regions it can see in the file it read.
    """
    (tmp_path / "a.py").write_text("alpha\nbeta\n")

    answer = asyncio.run(
        edit.execute(
            path="a.py",
            edits=[{"oldText": "alpha", "newText": "beta"}, {"oldText": "beta", "newText": "gamma"}],
        )
    )

    assert "replaced 2 block(s)" in answer
    assert (tmp_path / "a.py").read_text() == "beta\ngamma\n"


def test_an_oldtext_matching_twice_is_refused_rather_than_guessed(edit: EditFileTool, tmp_path: Path) -> None:
    """``replace_all`` is gone: an ambiguous anchor asks for more context.

    Applying an ambiguous edit everywhere and reporting success is the failure
    that cannot be seen from the result -- the model that meant one of the two
    has no way to tell which it got.
    """
    (tmp_path / "a.py").write_text("x = 1\nx = 1\n")

    answer = asyncio.run(edit.execute(path="a.py", edits=[{"oldText": "x = 1", "newText": "x = 2"}]))

    assert answer.startswith("Error: Found 2 occurrences")
    assert "unique" in answer
    assert (tmp_path / "a.py").read_text() == "x = 1\nx = 1\n", "nothing was written"


def test_a_missing_oldtext_names_the_entry_that_failed(edit: EditFileTool, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha\nbeta\n")

    answer = asyncio.run(
        edit.execute(
            path="a.py",
            edits=[{"oldText": "alpha", "newText": "ALPHA"}, {"oldText": "nowhere", "newText": "x"}],
        )
    )

    assert "edits[1]" in answer
    assert (tmp_path / "a.py").read_text() == "alpha\nbeta\n", "the first edit did not land either"


def test_a_lone_failing_edit_is_not_reported_by_index(edit: EditFileTool, tmp_path: Path) -> None:
    """A model told about ``edits[0]`` when it sent one edit looks for others."""
    (tmp_path / "a.py").write_text("alpha\n")

    answer = asyncio.run(edit.execute(path="a.py", edits=[{"oldText": "nowhere", "newText": "x"}]))

    assert "edits[" not in answer
    assert "Could not find the text" in answer


def test_overlapping_edits_are_refused(edit: EditFileTool, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("abcdef\n")

    answer = asyncio.run(
        edit.execute(
            path="a.py",
            edits=[{"oldText": "abcd", "newText": "Z"}, {"oldText": "cdef", "newText": "Y"}],
        )
    )

    assert "overlap" in answer
    assert (tmp_path / "a.py").read_text() == "abcdef\n"


def test_an_empty_oldtext_is_refused(edit: EditFileTool, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha\n")

    answer = asyncio.run(edit.execute(path="a.py", edits=[{"oldText": "", "newText": "x"}]))

    assert "must not be empty" in answer


def test_an_empty_edits_list_is_refused(edit: EditFileTool, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha\n")

    assert "at least one replacement" in asyncio.run(edit.execute(path="a.py", edits=[]))


def test_a_replacement_that_changes_nothing_says_so(edit: EditFileTool, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha\n")

    answer = asyncio.run(edit.execute(path="a.py", edits=[{"oldText": "alpha", "newText": "alpha"}]))

    assert "No changes made" in answer


def test_crlf_line_endings_survive_an_edit(edit: EditFileTool, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_bytes(b"alpha\r\nbeta\r\n")

    asyncio.run(edit.execute(path="a.py", edits=[{"oldText": "beta", "newText": "BETA"}]))

    assert (tmp_path / "a.py").read_bytes() == b"alpha\r\nBETA\r\n"


@pytest.mark.parametrize(
    "sent",
    [
        {"oldText": "alpha", "newText": "ALPHA"},
        {"edits": {"oldText": "alpha", "newText": "ALPHA"}},
        {"edits": '[{"oldText": "alpha", "newText": "ALPHA"}]'},
        {"edits": '{"oldText": "alpha", "newText": "ALPHA"}'},
    ],
    ids=["top-level pair", "single object", "json array string", "json object string"],
)
def test_the_shapes_models_send_instead_of_edits_are_repaired(edit: EditFileTool, tmp_path: Path, sent: dict) -> None:
    """pi repairs these before dispatch; so does ``cast_params``, which runs
    before validation, so a recoverable shape costs no round trip."""
    (tmp_path / "a.py").write_text("alpha\n")

    params = edit.cast_params({"path": "a.py", **sent})

    assert params["edits"] == [{"oldText": "alpha", "newText": "ALPHA"}]
    assert edit.validate_params(params) == []
    assert "replaced 1 block(s)" in asyncio.run(edit.execute(**params))


def test_an_edit_outside_the_permitted_directory_is_refused(tmp_path: Path) -> None:
    """The boundary pi's own tools do not carry."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_text("alpha\n")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tool = EditFileTool(workspace=workspace, allowed_dir=workspace)

    answer = asyncio.run(tool.execute(path=str(outside / "a.py"), edits=[{"oldText": "alpha", "newText": "x"}]))

    assert "outside allowed directory" in answer
    assert (outside / "a.py").read_text() == "alpha\n"


# ---------------------------------------------------------------------------
# read: 1-indexed offset, limit, and where it says to continue
# ---------------------------------------------------------------------------


@pytest.fixture
def ten_lines(tmp_path: Path) -> ReadFileTool:
    (tmp_path / "many.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    return ReadFileTool(workspace=tmp_path, allowed_dir=tmp_path)


def test_offset_is_one_indexed_as_in_pi(ten_lines: ReadFileTool) -> None:
    """``offset=1`` is the first line, not the second."""
    body = asyncio.run(ten_lines.execute(path="many.txt", offset=1, limit=1))

    assert body.startswith("1| line1")
    assert "line2" not in body


def test_limit_bounds_the_window_and_the_result_says_how_to_continue(ten_lines: ReadFileTool) -> None:
    body = asyncio.run(ten_lines.execute(path="many.txt", offset=3, limit=2))

    assert "3| line3" in body
    assert "4| line4" in body
    assert "line5" not in body
    assert "Use offset=5 to continue" in body


def test_the_last_window_says_it_reached_the_end(ten_lines: ReadFileTool) -> None:
    body = asyncio.run(ten_lines.execute(path="many.txt", offset=9))

    assert "End of file — 10 lines total" in body
    assert "continue" not in body


def test_an_offset_past_the_end_is_an_error_not_an_empty_read(ten_lines: ReadFileTool) -> None:
    assert "beyond end of file" in asyncio.run(ten_lines.execute(path="many.txt", offset=99))


def test_omitting_offset_reads_from_the_top(ten_lines: ReadFileTool) -> None:
    body = asyncio.run(ten_lines.execute(path="many.txt"))

    assert body.startswith("1| line1")
    assert "10| line10" in body


# ---------------------------------------------------------------------------
# write, ls, grep, find: what the new schemas changed
# ---------------------------------------------------------------------------


def test_write_creates_parent_directories_and_overwrites(tmp_path: Path) -> None:
    tool = WriteFileTool(workspace=tmp_path, allowed_dir=tmp_path)

    assert "Successfully wrote" in asyncio.run(tool.execute(path="deep/nest/a.txt", content="one"))
    asyncio.run(tool.execute(path="deep/nest/a.txt", content="two"))

    assert (tmp_path / "deep/nest/a.txt").read_text() == "two"


def test_the_write_hints_point_at_edit_now_that_there_is_no_append_mode(tmp_path: Path) -> None:
    """Both hints have to name a next action that the schema still offers."""
    tool = WriteFileTool(workspace=tmp_path, allowed_dir=tmp_path)

    for hint in (tool.truncation_hint, tool.incomplete_hint):
        assert "edit" in hint
        assert "append" not in hint


def test_ls_defaults_to_the_workspace_and_marks_directories(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / ".dotfile").write_text("")
    (tmp_path / "a.py").write_text("")
    (tmp_path / "node_modules").mkdir()

    body = asyncio.run(ListDirTool(workspace=tmp_path, allowed_dir=tmp_path).execute())

    assert body.splitlines() == [".dotfile", "a.py", "sub/"], "sorted, dotfiles kept, build dirs skipped"


def test_ls_limit_says_what_it_held_back(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("")

    body = asyncio.run(ListDirTool(workspace=tmp_path, allowed_dir=tmp_path).execute(limit=2))

    assert "showing first 2 of 5 entries" in body


def test_grep_takes_pis_ignorecase_and_literal(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("Alpha\na[b]c\n")
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    assert "Alpha" in asyncio.run(tool.execute(pattern="alpha", ignoreCase=True))
    assert asyncio.run(tool.execute(pattern="alpha")) == "No matches found."
    assert "a[b]c" in asyncio.run(tool.execute(pattern="a[b]", literal=True))
    assert asyncio.run(tool.execute(pattern="a[b]")) == "No matches found."


def test_grep_only_rejects_a_bad_regex_when_the_pattern_is_one(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("a[\n")
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    assert "invalid regular expression" in asyncio.run(tool.execute(pattern="a["))
    assert "a[" in asyncio.run(tool.execute(pattern="a[", literal=True))


def test_the_python_fallback_answers_the_same_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ripgrep is an optimisation, not a dependency, so the fallback takes the
    same two flags rather than quietly ignoring them."""
    (tmp_path / "x.txt").write_text("Alpha\na[b]c\n")
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    assert "Alpha" in asyncio.run(tool.execute(pattern="alpha", ignoreCase=True))
    assert asyncio.run(tool.execute(pattern="alpha")) == "No matches found."
    assert "a[b]c" in asyncio.run(tool.execute(pattern="a[b]", literal=True))


def test_find_returns_paths_relative_to_the_search_root(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("")
    (tmp_path / "src/b.ts").write_text("")

    body = asyncio.run(FindTool(workspace=tmp_path, allowed_dir=tmp_path).execute(pattern="*.py"))

    assert body == "src/a.py"


# ---------------------------------------------------------------------------
# The retired names, in sessions already on disk
# ---------------------------------------------------------------------------


def test_a_session_stored_under_the_retired_tool_names_still_loads(tmp_path: Path) -> None:
    """A ``toolName`` is a string in a file, not a lookup into the registry.

    Sessions written before the rename name ``read_file`` and ``exec``, and
    those tools no longer exist. Loading one must still produce the calls and
    the results it holds: the names are carried through as recorded, so the
    transcript reads the way it did when it was written. The renderer has no
    table to miss either -- a tool it does not recognise is headed by its own
    name.
    """
    import json

    from opendde_harness.providers import messages as msg
    from opendde_harness.session.manager import SessionManager

    records = [
        {"_type": "metadata", "key": "cli:old", "created_at": "2026-09-01T10:00:00"},
        {"role": "user", "content": "read a.txt", "id": "r1", "turn_id": "t1"},
        {
            "role": "assistant",
            "content": "reading",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}
            ],
            "id": "r2",
            "turn_id": "t1",
        },
        {"role": "tool", "tool_call_id": "c1", "content": "the file body", "id": "r3", "turn_id": "t1"},
        {
            "role": "assistant",
            "content": "running it",
            "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "exec", "arguments": '{"command": "ls"}'}}
            ],
            "id": "r4",
            "turn_id": "t1",
        },
        {"role": "tool", "tool_call_id": "c2", "content": "a.txt", "id": "r5", "turn_id": "t1"},
    ]
    manager = SessionManager(tmp_path)
    path = manager.sessions_dir / "cli" / "old.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    session = manager.get_or_create("cli:old")

    assert msg.tool_calls_of(session.messages[1]) == [msg.tool_call_block("c1", "read_file", {"path": "a.txt"})]
    assert session.messages[2]["toolName"] == "read_file"
    assert msg.tool_calls_of(session.messages[3]) == [msg.tool_call_block("c2", "exec", {"command": "ls"})]
    assert session.messages[4]["toolName"] == "exec"


def test_the_untrusted_fence_still_unwraps_a_retired_tools_name() -> None:
    """The fence carries the tool name it was written with, and the recognizer
    is the sentence, not the name -- so a stored result fenced as ``exec`` is
    still stripped before the person sees it."""
    from opendde_harness.security.trust import unwrap_untrusted, wrap_untrusted

    stored = wrap_untrusted("PDB 3RRQ", source="exec")

    assert unwrap_untrusted(stored) == "PDB 3RRQ"
