"""The user's own instruction files: ``AGENTS.md`` and ``ODH.md``.

Modelled on Claude Code's ``CLAUDE.md``. Two names are read in every directory
searched: ``AGENTS.md``, the convention other tools read too, and ``ODH.md``,
ours. Where a directory holds both, AGENTS.md is rendered first and ODH.md
after it, so the more specific file has the last word.

The search runs from the outermost scope inward, so the nearest instructions
win: the user scope beside the config, then the repository root, then every
directory from there down to the one the tool was launched in. Outside a git
work tree only the launch directory is read.

Two boundaries bound the walk, and both are about not reading somebody else's
files. The walk stops at the user's home when home is above the launch
directory, so a repository checked out under home cannot pull in a file from
home's parent; a repository that legitimately lives outside home is walked to
its own root, which is where it ends anyway. And a file is only accepted when
its *resolved* path is inside the scope it was found in, so a symlink named
``AGENTS.md`` cannot serve bytes from outside the repository under an
in-repository heading.

These are the user writing to their own agent, so they are **not** wrapped as
untrusted data -- unlike a tool result or a recalled memory, which are. That is
the whole distinction the fence exists to draw, and it is also why the agent's
own tools may not quietly write one: see ``is_instruction_file``, which the
write and edit tools use to route such a change through the approval prompt.
Discovering a filename is not evidence of authorship.

Caps, because this text rides on every request: 32 KiB a file and 128 KiB in
total, counted against the bytes actually accepted, headings included. A file
is read up to its allowance and one byte further -- enough to know it was cut,
not enough for an 8 MB file to cost 8 MB on the prompt path. The budget and
the rendering both work from that one read, so they cannot disagree about a
file that changed between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path

from loguru import logger

#: Read in this order within one directory: the shared convention, then ours.
FILENAMES = ("AGENTS.md", "ODH.md")

#: Where the user-scope pair lives, beside config.json.
USER_SCOPE_DIRNAME = ".opendde_harness"

#: What one file may contribute, and what all of them may contribute together.
MAX_FILE_BYTES = 32 * 1024
MAX_TOTAL_BYTES = 128 * 1024

#: Said in the rendered text itself, so the model knows it is reading a part.
TRUNCATION_NOTE = "\n\n[... truncated at {kib} KiB; the rest of this file was not sent]"

#: Said once above the files, so the model knows whose instructions these are.
PREAMBLE = (
    "The user keeps standing instructions for this work in the files below, "
    "nearest-directory last. Follow them as you would the user's own words."
)


def is_instruction_file(path: Path | str) -> bool:
    """Whether writing ``path`` would change what the agent is told to do.

    The name alone, because the name is what the loader looks for. A write to
    any file with one of these names becomes standing instructions the moment
    it lands somewhere the search reaches, and the tools that write files ask
    before doing that rather than after.
    """
    return Path(path).name in FILENAMES


@dataclass(frozen=True)
class InstructionFile:
    """One file the search found, what it contributes, and what became of it."""

    #: Absolute, symlinks resolved. The identity `/memory on|off` names and the
    #: one duplicates are collapsed on.
    path: str
    #: How it is shown and headed: relative to the work-tree root, `~/...` for
    #: the user scope, absolute otherwise.
    display: str
    #: Bytes on disk when it was read.
    size: int
    #: What it contributes, already capped and with the truncation note. Empty
    #: for a file that is switched off or that the total budget refused.
    text: str = ""
    #: sha256 of ``text``. What "changed since this session started" compares.
    digest: str = ""
    #: It is longer than the per-file cap, so only its head is sent.
    truncated: bool = False
    #: Found and not sent: the total was already spent by nearer files.
    skipped: bool = False
    #: The session has this one switched on.
    enabled: bool = True
    #: Its content differs from the first thing this session saw in it.
    changed: bool = False


# ---------------------------------------------------------------------------
# Where to look
# ---------------------------------------------------------------------------


def _git_root(start: Path, home: Path | None) -> Path | None:
    """The work tree ``start`` is in, or None.

    Walks for a ``.git`` entry rather than running git: this is on the prompt
    path, and a subprocess per turn to learn something a directory entry
    already says would be a strange thing to pay for. A file named ``.git`` is
    a work tree link and counts.

    The walk stops at ``home`` when home is above ``start``. Without that stop
    a stray ``.git`` in home's parent -- or in ``/`` -- made every directory
    above home part of "the repository", and an AGENTS.md up there became
    standing instructions for a project that had never heard of it. A
    repository outside home keeps its full walk: requiring every project to
    live under home would be a worse rule than the one being fixed.
    """
    under_home = home is not None and (start == home or home in start.parents)

    for directory in (start, *start.parents):
        if (directory / ".git").exists():
            return directory

        if under_home and directory == home:
            return None

    return None


def _search_directories(cwd: Path, home: Path) -> tuple[list[Path], Path | None]:
    """Every directory to look in, outermost first, and the work-tree root.

    The user scope leads because it is the most general: a rule that applies to
    every project this person works on. Then the repository root, then each
    directory down to the launch directory, which is the most specific and so
    has the last word. Deduplicated, because the user scope can *be* the launch
    directory -- somebody working in `~/.opendde_harness` -- and a directory
    listed twice contributed its files twice and spent the budget twice.
    """
    places: list[Path] = [home / USER_SCOPE_DIRNAME]
    root = _git_root(cwd, home)
    chain = [cwd] if root is None else [root, *[p for p in reversed(cwd.parents) if root in p.parents], cwd]
    seen: set[Path] = set(places)

    for directory in chain:
        if directory not in seen:
            seen.add(directory)
            places.append(directory)

    return places, root


def _display_for(path: Path, root: Path | None, cwd: Path, home: Path) -> str:
    """The name a heading and the ``/memory`` list use.

    Short, because it is a column in a list and the thing someone types after
    ``/memory off``. Inside a work tree that is the path from its root: how
    people name a file when they talk about the repository, and stable
    whichever directory the tool was launched in. Failing that, relative to
    the launch directory, then ``~/`` for the user scope, then the whole path.
    """
    for base, prefix in ((root, ""), (cwd, ""), (home, "~/")):
        if base is None:
            continue

        try:
            return prefix + path.relative_to(base).as_posix()
        except ValueError:
            continue

    return path.as_posix()


def _within(path: Path, scopes: list[Path]) -> bool:
    """Whether ``path`` really lives inside one of the scopes searched.

    Checked against the *resolved* path, which is the point: a symlink named
    AGENTS.md in the repository is an in-repository name for bytes that can be
    anywhere. Rendering those under an in-repository heading says the project
    ships instructions it does not ship.
    """
    return any(path == scope or scope in path.parents for scope in scopes)


# ---------------------------------------------------------------------------
# What they contribute
# ---------------------------------------------------------------------------


def _read_capped(path: Path) -> tuple[str, bool] | None:
    """One file's contribution and whether it was cut, or None if unreadable.

    Reads the allowance and one byte more. The extra byte is how "exactly the
    cap" and "longer than the cap" are told apart; without it a file that
    happened to be exactly 32 KiB would claim to have been truncated. Nothing
    beyond that is ever in memory, which is the difference between a large
    instruction file costing its allowance and costing its size.

    Bytes, not characters: the cap is about what the request costs. The cut is
    decoded with ``ignore`` so a multi-byte character split by it is dropped
    rather than delivered as a replacement glyph.
    """
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        logger.debug("project instructions: {} could not be read ({})", path, exc)

        return None

    if len(raw) <= MAX_FILE_BYTES:
        return raw.decode("utf-8", "replace").strip(), False

    return raw[:MAX_FILE_BYTES].decode("utf-8", "ignore").strip() + TRUNCATION_NOTE.format(
        kib=MAX_FILE_BYTES // 1024
    ), True


def _section(display: str, text: str) -> str:
    """One file as it appears in the prompt."""
    return f"## {display}\n\n{text}"


def snapshot(
    cwd: Path,
    *,
    home: Path | None = None,
    disabled: set[str] | None = None,
    already_loaded: set[Path] | None = None,
) -> list[InstructionFile]:
    """Every instruction file in scope, read once, outermost first.

    One pass: find, read, cap, budget. The budget is spent on the bytes this
    call actually read, so it cannot be computed against one version of a file
    and rendered from another -- which is what let six files, each grown to the
    per-file cap after being measured small, render half as much again as the
    total allows.

    ``already_loaded`` is the bootstrap segment's own files: a workspace that
    lists one of these among them has had it rendered already, and must not
    have it rendered twice.
    """
    home = (home or Path.home()).expanduser()

    try:
        cwd = Path(cwd).resolve()
        home = home.resolve()
    except OSError:
        return []

    places, root = _search_directories(cwd, home)
    # A file has to resolve inside one of the places searched. The user scope
    # and the work tree are the two scopes that exist; outside a work tree the
    # launch directory is its own.
    scopes = [home / USER_SCOPE_DIRNAME, root or cwd]
    already = set()

    for loaded in already_loaded or set():
        try:
            already.add(loaded.resolve())
        except OSError:
            continue

    found: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    for directory in places:
        for name in FILENAMES:
            candidate = directory / name

            try:
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
            except OSError:
                continue

            if resolved in already or resolved in seen:
                # The same bytes under a second name -- an aliased directory, a
                # symlink, the user scope reached twice -- is one instruction,
                # not two. Kept at the first place it was found, which is the
                # most general, so nearer files still override it.
                continue

            if not _within(resolved, scopes):
                logger.debug(
                    "project instructions: {} resolves outside this project ({}); not read",
                    candidate,
                    resolved,
                )
                continue

            seen.add(resolved)
            found.append((resolved, _display_for(candidate, root, cwd, home)))

    return _budgeted(found, disabled or set())


def _budgeted(found: list[tuple[Path, str]], disabled: set[str]) -> list[InstructionFile]:
    """Read what was found, nearest first, until the total is spent.

    The nearest instructions are the specific ones somebody wrote for the
    directory they are working in, so they claim the budget first and whatever
    is left goes to the more general files above them. What each one costs is
    its rendered section, heading included: the cap is a promise about the
    request, and a promise that ignored the headings would be a slightly
    false one.
    """
    budget = MAX_TOTAL_BYTES
    accepted: dict[Path, InstructionFile] = {}

    for path, display in reversed(found):
        try:
            size = path.stat().st_size
        except OSError:
            continue

        if path.as_posix() in disabled:
            # Still listed, because `/memory` has to show what it would switch
            # back on, and it costs nothing so it claims nothing.
            accepted[path] = InstructionFile(path=path.as_posix(), display=display, size=size, enabled=False)
            continue

        read = _read_capped(path)

        if read is None:
            continue

        text, truncated = read
        cost = len(_section(display, text).encode()) if text else 0

        if cost > budget:
            accepted[path] = InstructionFile(
                path=path.as_posix(),
                display=display,
                size=size,
                truncated=truncated,
                skipped=True,
            )
            continue

        budget -= cost
        accepted[path] = InstructionFile(
            path=path.as_posix(),
            display=display,
            size=size,
            text=text,
            digest=sha256(text.encode()).hexdigest() if text else "",
            truncated=truncated,
        )

    return [accepted[path] for path, _display in found if path in accepted]


def render(files: list[InstructionFile]) -> str:
    """The system-prompt section, or ``""`` when there is nothing to say.

    One heading per file naming where it came from, so a repository-wide rule
    is distinguishable from one belonging to the directory in hand. Nothing is
    read here: the text is what :func:`snapshot` accepted, so what the budget
    counted and what the model receives are the same bytes.
    """
    parts = [_section(f.display, f.text) for f in files if f.text and f.enabled and not f.skipped]

    if not parts:
        return ""

    return "# Project instructions\n\n" + PREAMBLE + "\n\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# What a session knows about them
# ---------------------------------------------------------------------------


@dataclass
class _SessionState:
    """One conversation's view of the instruction files."""

    #: Paths this conversation has switched off.
    disabled: set[str] = field(default_factory=set)
    #: What each file held the first time this conversation saw it. Drift from
    #: this is what `/memory` reports and what the UI says a line about.
    baseline: dict[str, str] = field(default_factory=dict)


class SessionInstructions:
    """Per-conversation instruction state, held for the life of the gateway.

    Keyed by session, because ``/memory off`` says "not for this conversation"
    and a single set shared by the process made it mean "not for any
    conversation, including ones opened later". Nothing here is written to
    disk: a preference that outlived the terminal would be a surprise the next
    time somebody launched.

    The baseline is the other half of the same idea. The loader records what it
    rendered, by content hash, the first time a conversation sees each file;
    anything else afterwards is drift, and drift is reported rather than
    quietly followed. An agent that writes its own standing instructions has to
    get past the approval prompt to do it, and if it gets past that by some
    other route this is what makes the change visible.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}

    def _state(self, session_key: str) -> _SessionState:
        return self._sessions.setdefault(session_key, _SessionState())

    def disabled(self, session_key: str) -> set[str]:
        return set(self._state(session_key).disabled)

    def set(self, session_key: str, path: str, *, enabled: bool) -> None:
        state = self._state(session_key)

        if enabled:
            state.disabled.discard(path)
        else:
            state.disabled.add(path)

    def observe(self, session_key: str, files: list[InstructionFile]) -> list[InstructionFile]:
        """Record what this conversation is seeing, and flag what moved.

        First sight of a file sets its baseline, so a session that opens on an
        already-written file reports no drift -- the file is simply what this
        conversation started with. A switched-off file is left alone: it is
        contributing nothing, so there is nothing to have changed.
        """
        baseline = self._state(session_key).baseline
        marked: list[InstructionFile] = []

        for file in files:
            if not file.enabled or file.skipped or not file.digest:
                marked.append(file)
                continue

            known = baseline.setdefault(file.path, file.digest)
            marked.append(file if known == file.digest else replace(file, changed=True))

        return marked

    def forget(self, session_key: str) -> None:
        self._sessions.pop(session_key, None)

    def clear(self) -> None:
        self._sessions.clear()


#: The gateway's own. One process serves one person's conversations.
SESSIONS = SessionInstructions()


def current(
    cwd: Path | None = None,
    *,
    session_key: str = "",
    already_loaded: set[Path] | None = None,
) -> list[InstructionFile]:
    """What the next turn of ``session_key`` would send.

    This runs on every turn and on every ``session.info``, so it answers with
    nothing rather than raising: a launch directory that has since been deleted
    makes ``os.getcwd`` fail, and losing the prompt's instructions is a smaller
    failure than losing the turn.
    """
    try:
        files = snapshot(
            cwd or Path.cwd(),
            disabled=SESSIONS.disabled(session_key),
            already_loaded=already_loaded,
        )
    except OSError as exc:
        logger.debug("project instructions: the search could not run ({})", exc)

        return []

    return SESSIONS.observe(session_key, files)


def match(argument: str, files: list[InstructionFile]) -> InstructionFile | None:
    """The file a ``/memory on|off <path>`` argument names, or None.

    Someone reading the list sees ``AGENTS.md`` and types that, so a bare name
    resolves when it is unambiguous. An ambiguous one does not, because
    switching off the wrong file is silent and would be found out a turn later.
    """
    wanted = argument.strip()

    if not wanted:
        return None

    for file in files:
        if wanted in (file.path, file.display):
            return file

    expanded = Path(wanted).expanduser()

    try:
        resolved = expanded.resolve().as_posix()
    except OSError:
        resolved = expanded.as_posix()

    for file in files:
        if file.path == resolved:
            return file

    by_name = [f for f in files if Path(f.path).name == wanted]

    return by_name[0] if len(by_name) == 1 else None
