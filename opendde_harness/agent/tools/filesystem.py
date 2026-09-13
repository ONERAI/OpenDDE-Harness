"""File system tools: ``read``, ``write``, ``edit`` and ``ls``.

The names and parameter schemas are pi's, so a prompt, skill or habit written
for pi's built-ins calls these unchanged. The implementations are the harness's
own: they carry the permitted-directory boundary, the standing-instructions
approval gate, and the result caps and hints that the agent loop reads.
"""

import base64
import difflib
import json
import mimetypes
from pathlib import Path
from typing import Any

from opendde_harness.agent.tools.approval import (
    ALREADY_REFUSED,
    NOT_INTERACTIVE,
    ApprovalGate,
    ApprovalResponder,
)
from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.context_engine.project_instructions import is_instruction_file
from opendde_harness.utils.helpers import detect_image_mime


def _resolve_path(path: str, workspace: Path | None = None, allowed_dir: Path | None = None) -> Path:
    """Resolve path against workspace (if relative) and enforce directory restriction."""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = workspace / p
    resolved = p.resolve()
    if allowed_dir:
        try:
            resolved.relative_to(allowed_dir.resolve())
        except ValueError:
            raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


class _FsTool(Tool):
    """Shared base for filesystem tools — common init and path resolution."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    def _resolve(self, path: str) -> Path:
        return _resolve_path(path, self._workspace, self._allowed_dir)


class _WritingFsTool(_FsTool):
    """A filesystem tool that changes a file, and so can change its own orders.

    ``AGENTS.md`` and ``ODH.md`` are read into the system prompt as the user's
    own standing instructions, unfenced, because that is what they are for. A
    directory being writable is authority to write files in it; it is not the
    user adopting whatever lands there as instructions. Without this, an agent
    that could write a file could write its own standing instructions and be
    following them on the next turn, and the only evidence would be a heading
    in a prompt nobody reads.

    So a write or an edit to one of those names is asked about, every time,
    even where the path is inside the permitted directory and an ordinary write
    there would need no permission at all. The prompt says what the file
    becomes, because "write AGENTS.md" does not look like a privileged act.
    """

    external_effects = True

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        super().__init__(workspace, allowed_dir)
        self._approval = ApprovalGate(f"{type(self).__name__}_approval_turn")

    def start_approval_turn(
        self,
        responder: ApprovalResponder | None,
        *,
        conversation_id: str,
        turn_id: str,
    ) -> None:
        """Bind or revoke interactive approval capability for the current turn."""
        self._approval.start_approval_turn(responder, conversation_id=conversation_id, turn_id=turn_id)

    def set_tool_call_id(self, tool_call_id: str) -> None:
        """Attach the provider call ID so approval is auditable end to end."""
        self._approval.set_tool_call_id(tool_call_id)

    async def _approve_instruction_change(self, fp: Path, what: str) -> str | None:
        """None when this change may go ahead, else what to tell the model.

        Fails closed. A turn with no way to ask -- a background run, a channel
        with no approval transport -- cannot write standing instructions, which
        is the point: those are the turns nobody is watching.
        """
        if not is_instruction_file(fp):
            return None

        reason = await self._approval.ask(
            subject=f"{what} {fp}",
            description=(
                f"{what.capitalize()} {fp.name}, which the agent reads on every turn as your own standing instructions"
            ),
        )

        if reason is None:
            return None

        return self._REFUSALS.get(reason, self._REFUSAL_UNANSWERED).format(name=fp.name)

    _REFUSAL_UNANSWERED = "Error: the approval request could not be completed, so {name} was not written"
    _REFUSALS = {
        ALREADY_REFUSED: "Error: this change to {name} was already refused earlier in the current turn",
        NOT_INTERACTIVE: (
            "Error: {name} is read as the user's standing instructions, so changing it needs their "
            "approval, and this turn has no way to ask. Tell the user what you would put in it."
        ),
        "deny": "Error: the user denied this change to {name}",
        "timeout": "Error: the approval request expired before it was answered, so {name} was not written",
        "cancelled": "Error: the approval request was cancelled before it was answered, so {name} was not written",
    }


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


class ReadFileTool(_FsTool):
    """Read file contents with optional line-based pagination.

    Named ``read`` on the wire, with pi's ``{path, offset, limit}`` schema and
    its 1-indexed ``offset``, so a prompt or habit written for pi's ``read``
    works here unchanged.
    """

    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2000

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
            "Images are sent as attachments, downscaled if large. For text files, output returns "
            f"numbered lines, truncated to {self._DEFAULT_LIMIT} lines or {self._MAX_CHARS // 1000}KB "
            "(whichever is hit first). Use offset/limit for large files. When you need the full file, "
            "continue with offset until complete. Reads are confined to the permitted directory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed)",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        }

    # Sniffed from the first bytes, not the extension — a .txt that is really a
    # PNG must not be decoded as UTF-8, and vice versa.
    _MAGIC_SNIFF_BYTES = 32

    def _read_image(self, fp: Path, mime: str) -> ToolResult:
        """Image branch: preprocess, then hand back text *and* an image block.

        ``model_text`` is deliberately self-sufficient. Providers that cannot put
        an image in a tool result never see ``blocks``, and the same text is what
        lands in session history, so it carries the metadata and the path rather
        than pointing at a picture that may not be there.
        """
        from opendde_harness.agent.tools import media

        payload, out_mime, meta = media.prepare_image(fp.read_bytes(), mime)
        summary = media.describe_image(fp, meta)
        return ToolResult(
            model_text=summary,
            display_text=f"{fp.name} ({meta['width']}x{meta['height']}, ~{meta['tokens']} tok)",
            blocks=[
                media.text_block(summary),
                media.image_block(base64.b64encode(payload).decode("ascii"), out_mime),
            ],
        )

    async def execute(self, path: str, offset: int = 1, limit: int | None = None, **kwargs: Any) -> str | ToolResult:
        try:
            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"
            if not fp.is_file():
                return f"Error: Not a file: {path}"

            with fp.open("rb") as fh:
                head = fh.read(self._MAGIC_SNIFF_BYTES)
            sniffed = detect_image_mime(head)
            mime = sniffed
            if mime is None:
                guessed = mimetypes.guess_type(fp.name)[0]
                # Magic bytes only cover the four formats every target inlines, so
                # a raster format Pillow can convert (BMP/TIFF/ICO) reaches this
                # branch as an extension guess. A guess can be wrong both ways --
                # .svg is XML with no Pillow decoder, a .png stub may hold text --
                # so it is trusted provisionally and a decode failure falls back
                # to the text path below rather than refusing to read the file.
                if guessed and guessed.startswith("image/"):
                    mime = guessed
            if mime is not None:
                from opendde_harness.agent.tools.media import ImageTooLargeError

                try:
                    return self._read_image(fp, mime)
                except ImageTooLargeError as e:
                    return f"Error: {e}"
                except Exception as e:
                    if sniffed is not None:
                        return f"Error decoding image {path}: {e}"
                    # An extension-only guess that would not decode: fall through.

            all_lines = fp.read_text(encoding="utf-8").splitlines()
            total = len(all_lines)

            if offset < 1:
                offset = 1
            if total == 0:
                return f"(Empty file: {path})"
            if offset > total:
                return f"Error: offset {offset} is beyond end of file ({total} lines)"

            start = offset - 1
            end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(all_lines[start:end])]
            result = "\n".join(numbered)

            if len(result) > self._MAX_CHARS:
                trimmed, chars = [], 0
                for line in numbered:
                    chars += len(line) + 1
                    if chars > self._MAX_CHARS:
                        break
                    trimmed.append(line)
                end = start + len(trimmed)
                result = "\n".join(trimmed)

            if end < total:
                result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
            else:
                result += f"\n\n(End of file — {total} lines total)"
            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {e}"


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


class WriteFileTool(_WritingFsTool):
    """Write content to a file.

    Named ``write`` with pi's ``{path, content}`` schema. There is no append
    mode: pi's ``write`` has none, and a second parameter that changes whether
    a call replaces or extends a file is the one thing a truncated call must
    never get wrong (a cut before ``mode`` arrived turned an append into a
    silent overwrite). A file too long for one reply is started with ``write``
    and extended with ``edit``, which names the text it is anchored to and so
    cannot mean two different things.
    """

    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories. Writes are confined to the permitted "
            "directory, and a file the agent reads as your standing instructions needs your approval."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        }

    @property
    def truncation_hint(self) -> str:
        return (
            "Arguments cut off by that limit are discarded whole rather than partly saved, so "
            "build the file up instead: write the first part, then extend it with edit, whose "
            "oldText anchors on text already in the file."
        )

    @property
    def incomplete_hint(self) -> str:
        return (
            "arguments cut off that way are discarded whole rather than partly saved, so build "
            "the file up instead -- write the first part, then extend it with edit, whose oldText "
            "anchors on text already in the file."
        )

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            fp = self._resolve(path)
            refusal = await self._approve_instruction_change(fp, "write")
            if refusal:
                return refusal
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {fp}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {e}"


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


def _find_match(content: str, old_text: str) -> tuple[str | None, int]:
    """Locate old_text in content: exact first, then line-trimmed sliding window.

    Both inputs should use LF line endings (caller normalises CRLF).
    Returns (matched_fragment, count) or (None, 0).
    """
    if old_text in content:
        return old_text, content.count(old_text)

    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0
    stripped_old = [line.strip() for line in old_lines]
    content_lines = content.splitlines()

    candidates = []
    for i in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[i : i + len(stripped_old)]
        if [line.strip() for line in window] == stripped_old:
            candidates.append("\n".join(window))

    if candidates:
        return candidates[0], len(candidates)
    return None, 0


class EditFileTool(_WritingFsTool):
    """Edit one file by exact text replacement, one or more places at a time.

    Named ``edit`` with pi's schema: ``{path, edits: [{oldText, newText}]}``.
    Every ``oldText`` is matched against the *original* file, not against the
    result of the edits before it, so the model does not have to track offsets
    it cannot see. Each must match exactly once and the matched regions must not
    overlap; otherwise nothing is written and the error names the entry that
    failed.

    ``replace_all`` is gone. It existed to resolve an ambiguous ``oldText`` by
    applying it everywhere, which is the opposite of what a unique anchor is
    for: a model that meant one of five occurrences and got all five has no way
    to tell from the success message. Several occurrences now ask for more
    context instead.
    """

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return (
            "Edit a single file using exact text replacement. Every edits[].oldText must match a "
            "unique, non-overlapping region of the original file. If two changes affect the same "
            "block or nearby lines, merge them into one edit instead of emitting overlapping edits. "
            "Do not include large unchanged regions just to connect distant changes. Edits are "
            "confined to the permitted directory, and a file the agent reads as your standing "
            "instructions needs your approval."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                "edits": {
                    "type": "array",
                    "description": (
                        "One or more targeted replacements. Each edit is matched against the "
                        "original file, not incrementally. Do not include overlapping or nested "
                        "edits. If two changes touch the same block or nearby lines, merge them "
                        "into one edit instead."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": (
                                    "Exact text for one targeted replacement. It must be unique in "
                                    "the original file and must not overlap with any other "
                                    "edits[].oldText in the same call."
                                ),
                            },
                            "newText": {
                                "type": "string",
                                "description": "Replacement text for this targeted edit.",
                            },
                        },
                        "required": ["oldText", "newText"],
                    },
                },
            },
            "required": ["path", "edits"],
        }

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Normalise the shapes models send instead of ``edits`` before validating.

        Some models send ``edits`` as a JSON *string*, some send a single edit
        object where an array belongs, and some put ``oldText``/``newText`` at
        the top level as if the old single-edit tool were still here. pi repairs
        all three before dispatch; this is the same repair at the one hook that
        runs before :meth:`validate_params`, so a recoverable shape costs the
        model no round trip.
        """
        if not isinstance(params, dict):
            return super().cast_params(params)

        fixed = dict(params)
        edits = fixed.get("edits")
        if isinstance(edits, str):
            try:
                parsed = json.loads(edits)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                fixed["edits"] = parsed
            elif _is_single_edit(parsed):
                fixed["edits"] = [parsed]
        elif _is_single_edit(edits):
            fixed["edits"] = [edits]

        legacy = {"oldText": fixed.pop("oldText", None), "newText": fixed.pop("newText", None)}
        if isinstance(legacy["oldText"], str) and isinstance(legacy["newText"], str):
            existing = fixed.get("edits")
            fixed["edits"] = [*(existing if isinstance(existing, list) else []), legacy]

        return super().cast_params(fixed)

    async def execute(self, path: str, edits: list[dict[str, Any]], **kwargs: Any) -> str:
        if not isinstance(edits, list) or not edits:
            return "Error: edits must contain at least one replacement."
        total = len(edits)
        try:
            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"

            refusal = await self._approve_instruction_change(fp, "edit")
            if refusal:
                return refusal

            raw = fp.read_bytes()
            uses_crlf = b"\r\n" in raw
            content = raw.decode("utf-8").replace("\r\n", "\n")

            # Every edit is located in the *original* content before any of them
            # is applied, so an earlier replacement cannot move or destroy a
            # later one's anchor. Nothing is written unless all of them resolve.
            located: list[tuple[int, int, int, str]] = []  # (start, end, edit index, new text)
            for index, edit in enumerate(edits):
                old_text = edit.get("oldText")
                new_text = edit.get("newText")
                if not isinstance(old_text, str) or not isinstance(new_text, str):
                    return f"Error: {_ref(index, total)} needs both oldText and newText as strings."
                if not old_text:
                    empty = "oldText" if total == 1 else f"edits[{index}].oldText"
                    return f"Error: {empty} must not be empty in {path}."

                match, count = _find_match(content, old_text.replace("\r\n", "\n"))
                if match is None:
                    return self._not_found_msg(old_text, content, path, index, total)
                if count > 1:
                    unique = "The text must be unique." if total == 1 else "Each oldText must be unique."
                    return (
                        f"Error: Found {count} occurrences of {_ref(index, total)} in {path}. "
                        f"{unique} Please provide more context to make it unique."
                    )
                begin = content.index(match)
                located.append((begin, begin + len(match), index, new_text.replace("\r\n", "\n")))

            located.sort()
            for (_, prev_end, prev_index, _), (next_begin, _, next_index, _) in zip(located, located[1:], strict=False):
                if prev_end > next_begin:
                    return (
                        f"Error: edits[{prev_index}] and edits[{next_index}] overlap in {path}. "
                        "Merge them into one edit or target disjoint regions."
                    )

            # Applied back to front so each replacement leaves the offsets of
            # the ones before it untouched.
            new_content = content
            for begin, finish, _, new_text in reversed(located):
                new_content = new_content[:begin] + new_text + new_content[finish:]

            if new_content == content:
                if total == 1:
                    return (
                        f"Error: No changes made to {path}. The replacement produced identical content. "
                        "This might indicate an issue with special characters or the text not existing "
                        "as expected."
                    )
                return f"Error: No changes made to {path}. The replacements produced identical content."

            if uses_crlf:
                new_content = new_content.replace("\n", "\r\n")

            fp.write_bytes(new_content.encode("utf-8"))
            return f"Successfully replaced {total} block(s) in {fp}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {e}"

    @staticmethod
    def _not_found_msg(old_text: str, content: str, path: str, index: int, total: int) -> str:
        lines = content.splitlines(keepends=True)
        old_lines = old_text.splitlines(keepends=True)
        window = len(old_lines)

        best_ratio, best_start = 0.0, 0
        for i in range(max(1, len(lines) - window + 1)):
            ratio = difflib.SequenceMatcher(None, old_lines, lines[i : i + window]).ratio()
            if ratio > best_ratio:
                best_ratio, best_start = ratio, i

        exactly = "The old text" if total == 1 else "The oldText"
        head = (
            f"Error: Could not find {_ref(index, total)} in {path}. "
            f"{exactly} must match exactly including all whitespace and newlines."
        )
        if best_ratio > 0.5:
            diff = "\n".join(
                difflib.unified_diff(
                    old_lines,
                    lines[best_start : best_start + window],
                    fromfile="oldText (provided)",
                    tofile=f"{path} (actual, line {best_start + 1})",
                    lineterm="",
                )
            )
            return f"{head}\nBest match ({best_ratio:.0%} similar) at line {best_start + 1}:\n{diff}"
        return f"{head} No similar text found — verify the file content."


def _is_single_edit(value: Any) -> bool:
    """Whether ``value`` is one ``{oldText, newText}`` pair rather than a list."""
    return isinstance(value, dict) and isinstance(value.get("oldText"), str) and isinstance(value.get("newText"), str)


def _ref(index: int, total: int) -> str:
    """``the text`` for a lone edit, ``edits[i]`` when there are several.

    A single-edit call has no index worth quoting, and pi's own errors say so
    differently for the two cases; a model told about ``edits[0]`` when it sent
    one edit goes looking for the other entries it never wrote.
    """
    return "the text" if total == 1 else f"edits[{index}]"


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


class ListDirTool(_FsTool):
    """List one directory.

    Named ``ls`` with pi's ``{path, limit}`` schema, both optional. Neither
    ``recursive`` nor a per-call ignore list survives the alignment: a recursive
    listing of an unknown tree is how a single call returns tens of thousands of
    paths, and ``find`` already answers "what is under here" with a pattern and
    a cap.
    """

    _DEFAULT_LIMIT = 500
    _IGNORE_DIRS = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".coverage",
        "htmlcov",
    }

    @property
    def name(self) -> str:
        return "ls"

    @property
    def description(self) -> str:
        return (
            "List directory contents. Returns entries sorted alphabetically, with '/' suffix for "
            f"directories. Includes dotfiles. Output is truncated to {self._DEFAULT_LIMIT} entries. "
            "Build directories and caches (.git, node_modules, __pycache__, and the like) are "
            "skipped, and listing is confined to the permitted directory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list (default: current directory)"},
                "limit": {
                    "type": "integer",
                    "description": f"Maximum number of entries to return (default: {self._DEFAULT_LIMIT})",
                    "minimum": 1,
                },
            },
            "required": [],
        }

    async def execute(self, path: str = ".", limit: int | None = None, **kwargs: Any) -> str:
        try:
            dp = self._resolve(path)
            if not dp.exists():
                return f"Error: Directory not found: {path}"
            if not dp.is_dir():
                return f"Error: Not a directory: {path}"

            cap = limit or self._DEFAULT_LIMIT
            items: list[str] = []
            total = 0

            for item in sorted(dp.iterdir()):
                if item.name in self._IGNORE_DIRS:
                    continue
                total += 1
                if len(items) < cap:
                    items.append(f"{item.name}/" if item.is_dir() else item.name)

            if not items and total == 0:
                return f"Directory {path} is empty"

            result = "\n".join(items)
            if total > cap:
                result += f"\n\n(truncated, showing first {cap} of {total} entries)"
            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {e}"
