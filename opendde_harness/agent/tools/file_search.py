"""Search tools: ``grep`` (content search) and ``find`` (file lookup).

Both carry pi's names and parameter schemas, camelCase included, so a prompt or
habit written for pi's built-ins calls them unchanged.

Both run host-side and reuse ``_FsTool``'s workspace/allowed_dir resolution so
they share the exact same path boundary as read/write/ls — never the
SandboxExecutor (avoids shuttling large result sets across a VM edge).

``grep`` prefers the ``rg`` (ripgrep) binary when present on PATH for speed and
.gitignore awareness, and falls back to a pure-Python scan otherwise so opendde
keeps working with zero hard binary dependency.
"""

import asyncio
import fnmatch
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from opendde_harness.agent.tools.filesystem import _FsTool

# Noise directories skipped by the pure-Python fallback / find. ripgrep handles
# its own ignore logic via .gitignore, so this only gates the fallback path.
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

# Pseudo / system filesystem roots that must never be tree-walked. A model that
# runs `grep <pat> /` (or find over /) would otherwise traverse the entire host
# — including slow network mounts under /proc, /sys, or /mnt — and hang the whole
# run indefinitely (observed: a 47-min wedge in disk-sleep on a shared mount).
# Searches must name a real subtree, not a system root.
_DENY_TRAVERSAL_ROOTS = {Path(p) for p in ("/", "/proc", "/sys", "/dev", "/run", "/boot")}
# Wall-clock cap on the pure-Python os.walk fallback so an allowed-but-huge tree
# still cannot hang the loop. ripgrep already has its own _RG_TIMEOUT.
_WALK_DEADLINE_S = 20.0


def _denied_traversal_root(base: Path) -> bool:
    """True if ``base`` resolves to a system root that must not be tree-walked."""
    try:
        resolved = base.resolve()
    except OSError:
        return False
    # Any filesystem/drive root is its own parent — catches POSIX "/" and the
    # Windows drive/UNC roots ("C:\\", "\\\\server\\share") that the POSIX-only
    # _DENY_TRAVERSAL_ROOTS set misses (a search at C:\ would otherwise walk the
    # whole drive).
    if resolved.parent == resolved:
        return True
    return resolved in _DENY_TRAVERSAL_ROOTS


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


class GrepTool(_FsTool):
    """Search file contents by regex, ripgrep-backed with a pure-Python fallback.

    pi's schema exactly, which drops our ``output_mode``: a file list and a
    per-file count were two more shapes of the same answer, and ``limit`` plus a
    narrower ``glob`` reaches the same place through one output format the model
    already knows how to read.
    """

    _MAX_CHARS = 30_000
    _DEFAULT_LIMIT = 100
    _RG_TIMEOUT = 30

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents for a pattern. Returns matching lines with file paths and line "
            "numbers. Respects .gitignore. Output is truncated to "
            f"{self._DEFAULT_LIMIT} matches or {self._MAX_CHARS // 1000}KB (whichever is hit first). "
            "Prefer this over running grep or rg through bash, and use glob to restrict to file "
            "types (e.g. '*.py'). Searching is confined to the permitted directory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: current directory)",
                },
                "glob": {
                    "type": "string",
                    "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'",
                },
                "ignoreCase": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)",
                },
                "literal": {
                    "type": "boolean",
                    "description": "Treat pattern as literal string instead of regex (default: false)",
                },
                "context": {
                    "type": "integer",
                    "description": "Number of lines to show before and after each match (default: 0)",
                    "minimum": 0,
                    "maximum": 20,
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum number of matches to return (default: {self._DEFAULT_LIMIT})",
                    "minimum": 1,
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        ignoreCase: bool = False,  # noqa: N803 — pi's parameter name, verbatim
        literal: bool = False,
        context: int = 0,
        limit: int | None = None,
        **kwargs: Any,
    ) -> str:
        cap = limit or self._DEFAULT_LIMIT
        if not literal:
            try:
                re.compile(pattern)
            except re.error as e:
                return f"Error: invalid regular expression: {e}"
        try:
            base = self._resolve(path)
        except PermissionError as e:
            return f"Error: {e}"
        if not base.exists():
            return f"Error: path not found: {path}"
        if base.is_dir() and _denied_traversal_root(base):
            return (
                f"Error: refusing to search '{path}' — it resolves to a system root "
                f"({base.resolve()}). Searching the whole filesystem hangs the agent. "
                "Specify a narrower directory (e.g. the workspace or a project subtree)."
            )

        rg = shutil.which("rg")
        try:
            if rg:
                return await self._run_rg(rg, pattern, base, glob, ignoreCase, literal, context, cap)
            return self._run_python(pattern, base, glob, ignoreCase, literal, context, cap)
        except Exception as e:
            return f"Error running grep: {e}"

    # ── ripgrep backend ─────────────────────────────────────────────────

    async def _run_rg(
        self,
        rg: str,
        pattern: str,
        base: Path,
        glob: str | None,
        ignore_case: bool,
        literal: bool,
        context: int,
        cap: int,
    ) -> str:
        args = [rg, "--color=never"]
        if ignore_case:
            args.append("-i")
        if literal:
            args.append("--fixed-strings")
        if glob:
            args += ["-g", glob]
        # rg only skips noise dirs when a .gitignore says so; add explicit excludes
        # so it matches the pure-Python fallback regardless of repo state. These come
        # after any user glob so the excludes win on last-match-wins ordering.
        for d in _IGNORE_DIRS:
            args += ["-g", f"!{d}"]

        args += ["--line-number", "--no-heading", "--with-filename"]
        if context:
            args += ["-C", str(context)]
        # Force forward-slash separators in rg's output paths on every platform
        # so results read the same on Windows as POSIX (only affects path fields,
        # not matched content).
        args += ["--path-separator", "/"]
        args += ["-e", pattern, "--", str(base)]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self._RG_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: grep timed out after {self._RG_TIMEOUT}s"

        # rg exits 1 when there are no matches — that is a normal empty result.
        if proc.returncode not in (0, 1):
            return f"Error running rg: {err.decode('utf-8', 'replace').strip()}"

        text = out.decode("utf-8", "replace")
        # Make paths relative to the search root for compact, readable output.
        # rg emits forward-slash paths (--path-separator above); match that when
        # stripping the search-root prefix so the strip works on Windows too.
        base_str = str(base).replace(os.sep, "/")
        text = text.replace(base_str + "/", "").replace(base_str, base.name or ".")
        lines = [ln for ln in text.splitlines() if ln]
        if not lines:
            return "No matches found."

        return self._format_lines(lines, cap)

    # ── pure-Python fallback ────────────────────────────────────────────

    def _run_python(
        self,
        pattern: str,
        base: Path,
        glob: str | None,
        ignore_case: bool,
        literal: bool,
        context: int,
        cap: int,
    ) -> str:
        flags = re.IGNORECASE if ignore_case else 0
        rx = re.compile(re.escape(pattern) if literal else pattern, flags)
        files = self._iter_files(base, glob)

        content_lines: list[str] = []

        for fp in files:
            rel = self._relpath(fp, base)
            try:
                raw = fp.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:  # skip binary files
                continue
            text_lines = raw.decode("utf-8", "replace").splitlines()

            hits = [i for i, line in enumerate(text_lines) if rx.search(line)]
            if not hits:
                continue

            self._collect_content(content_lines, rel, text_lines, hits, context)

        return self._format_lines(content_lines, cap) if content_lines else "No matches found."

    @staticmethod
    def _collect_content(
        out: list[str],
        rel: str,
        text_lines: list[str],
        hits: list[int],
        context: int,
    ) -> None:
        emitted: set[int] = set()
        for h in hits:
            lo = max(0, h - context)
            hi = min(len(text_lines), h + context + 1)
            for i in range(lo, hi):
                if i in emitted:
                    continue
                emitted.add(i)
                sep = ":" if i == h or context == 0 else "-"
                out.append(f"{rel}{sep}{i + 1}{sep}{text_lines[i]}")

    def _iter_files(self, base: Path, glob: str | None):
        if base.is_file():
            yield base
            return
        deadline = time.monotonic() + _WALK_DEADLINE_S
        for root, dirs, names in os.walk(base):
            if time.monotonic() > deadline:
                # Stop rather than hang on an unexpectedly huge / slow tree.
                break
            dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
            for n in sorted(names):
                if glob and not fnmatch.fnmatch(n, glob):
                    continue
                yield Path(root) / n

    @staticmethod
    def _relpath(fp: Path, base: Path) -> str:
        try:
            return fp.relative_to(base if base.is_dir() else base.parent).as_posix()
        except ValueError:
            return fp.as_posix()

    def _format_lines(self, lines: list[str], cap: int) -> str:
        total = len(lines)
        shown = lines[:cap]
        result = "\n".join(shown)
        notes = []
        if total > cap:
            notes.append(
                f"showing first {cap} of {total} matching lines — {total - cap} more not shown; "
                "this is a PARTIAL result, do not treat it as the complete set or count from it. "
                "Narrow the pattern or the glob, or raise limit, to see the rest"
            )
        if len(result) > self._MAX_CHARS:
            result = result[: self._MAX_CHARS]
            notes.append(
                f"output truncated to {self._MAX_CHARS} chars — narrow the pattern or the glob "
                "rather than counting from this view"
            )
        if notes:
            result += f"\n\n(warning: {'; '.join(notes)})"
        return result


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------


class FindTool(_FsTool):
    """Find files by glob pattern, sorted by recency. Pure-Python (pathlib)."""

    _DEFAULT_LIMIT = 1000

    @property
    def name(self) -> str:
        return "find"

    @property
    def description(self) -> str:
        return (
            "Search for files by glob pattern. Returns matching file paths relative to the search "
            f"directory, most-recently-modified first. Respects .gitignore. Output is truncated to "
            f"{self._DEFAULT_LIMIT} results. Prefer this over running find or ls through bash. "
            "Searching is confined to the permitted directory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum number of results (default: {self._DEFAULT_LIMIT})",
                    "minimum": 1,
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        limit: int | None = None,
        **kwargs: Any,
    ) -> str:
        cap = limit or self._DEFAULT_LIMIT
        try:
            base = self._resolve(path)
        except PermissionError as e:
            return f"Error: {e}"
        if not base.exists():
            return f"Error: path not found: {path}"
        if not base.is_dir():
            return f"Error: not a directory: {path}"
        if _denied_traversal_root(base):
            return (
                f"Error: refusing to search '{path}' — it resolves to a system root "
                f"({base.resolve()}). Specify a narrower directory."
            )

        # A path-bearing pattern globs literally; a bare pattern matches basenames
        # recursively (fd-style), so 'foo.py' finds it at any depth.
        glob_expr = pattern if "/" in pattern else f"**/{pattern}"
        try:
            matches = [
                p for p in base.glob(glob_expr) if not any(part in _IGNORE_DIRS for part in p.relative_to(base).parts)
            ]
        except (ValueError, OSError) as e:
            return f"Error running find: {e}"
        if not matches:
            return "No files found matching pattern."

        matches.sort(key=lambda p: self._mtime(p), reverse=True)
        total = len(matches)
        shown = matches[:cap]
        lines = [f"{p.relative_to(base).as_posix()}/" if p.is_dir() else p.relative_to(base).as_posix() for p in shown]
        result = "\n".join(lines)
        if total > cap:
            result += f"\n\n(showing first {cap} of {total} results)"
        return result

    @staticmethod
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0
