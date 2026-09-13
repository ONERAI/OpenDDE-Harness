"""Custom Hatchling build hook: conditionally package the prebuilt JS bundles.

``npm run build`` produces one self-contained esbuild bundle per entry point in
``ui-tui/dist/`` (see BUNDLES): the terminal UI, and the pi-ai model service the
Python side spawns. We want a wheel to carry them so `pip`/`uv tool install`
yields a working `ddeharness tui` with no source checkout. But ``dist/`` is a
build artifact and is NOT committed (see .gitignore), so a clean checkout
legitimately lacks it.

A static ``[tool.hatch.build.targets.wheel.force-include]`` entry would make
hatchling hard-fail with "Forced include not found" whenever ``dist/`` is
absent — which would break the ordinary developer flow (`git clone && uv sync`
with no prior `npm run build`). So instead we add the bundles to the wheel's
force-include map *only when they exist*, and emit a warning otherwise.

Release builds run ``npm ci && npm run build`` first (see
.github/workflows/release.yml), so the published wheel always carries them; dev
builds without them simply fall back to the source tree at runtime (see
resolve_dist() in opendde_harness/node_runtime.py).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

#: Every bundle ``npm run build`` writes into ``ui-tui/dist/``. One list, used
#: both to decide the tree is stale and to decide the wheel may ship it: a
#: bundle missing from here is a bundle nobody notices is missing.
BUNDLES = ("entry.js", "model-service.js")


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        self._include_ui(build_data, "ui-tui", required=True)

    def _include_ui(self, build_data: dict, name: str, *, required: bool) -> None:
        ui = Path(self.root) / name
        if not required and not ui.is_dir():
            return
        dist = ui / "dist"
        _refresh_bundle(ui, dist, self.app)
        missing = [bundle for bundle in BUNDLES if not (dist / bundle).is_file()]
        if not missing:
            # Map source -> path inside the wheel's `opendde_harness` package.
            build_data.setdefault("force_include", {})[str(dist)] = f"opendde_harness/{name}/dist"
        else:
            self.app.display_warning(
                f"{name}/dist is missing {', '.join(missing)} — building WITHOUT the bundled TUI. "
                f"Run `npm --prefix {name} ci && npm --prefix {name} run build` before "
                "building a release wheel."
            )


def _refresh_bundle(ui: Path, dist: Path, app) -> None:
    """Rebuild the bundles from a source tree when any is missing or stale.

    A wheel built from a checkout must not ship a bundle older than the sources
    next to it; release builds run npm beforehand and are a no-op here.
    """
    import shutil
    import subprocess

    if not (ui / "src").is_dir():
        return
    if not bundle_is_stale(ui, dist):
        return

    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError(
            f"{ui.name}/dist is missing a bundle or older than {ui.name}/src and npm is not available; "
            f"run `npm --prefix {ui.name} ci && npm --prefix {ui.name} run build` before building the wheel."
        )

    with _build_lock(ui, app):
        # Asked again now that nobody else can be working: whoever held the
        # lock may have done exactly this job while we waited for it.
        if not bundle_is_stale(ui, dist):
            return

        app.display_info(f"{ui.name}/dist is stale or missing a bundle; running `npm run build` in {ui.name}/")
        if node_modules_is_stale(ui):
            app.display_info(f"{ui.name}/node_modules is missing or older than its lockfile; running `npm ci`")
            subprocess.run([npm, "ci"], cwd=str(ui), check=True)
        subprocess.run([npm, "run", "build"], cwd=str(ui), check=True)


def bundle_is_stale(ui: Path, dist: Path) -> bool:
    """Whether any built bundle is missing or older than what it is built from.

    One stale bundle is enough: ``npm run build`` writes all of them, so there
    is no such thing as refreshing half the tree.
    """
    sources = [*(ui / "src").rglob("*"), ui / "package.json", ui / "package-lock.json"]
    newest = max((f.stat().st_mtime for f in sources if f.is_file()), default=0.0)

    for name in BUNDLES:
        bundle = dist / name
        if not bundle.is_file() or bundle.stat().st_mtime < newest:
            return True

    return False


@contextmanager
def _build_lock(ui: Path, app):
    """Hold one checkout's refresh against every other process doing the same.

    Two builds of one checkout can both find the tree stale and both start
    installing, and ``npm ci`` empties ``node_modules`` before refilling it, so
    one install deletes what the other build is reading. Freshness answers a
    question about the past; it is not a mutex.

    The lock is held across the install *and* the build, because the overlap
    that corrupts is an install landing inside someone else's build. It lives
    beside the tree it protects, not inside ``node_modules``, which ``npm ci``
    deletes. Waiting is the right behavior: the loser's recheck then finds the
    work already done.
    """
    try:
        import fcntl
    except ImportError:  # Windows, which this project does not target.
        yield
        return

    path = ui / ".build-lock"
    with path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            app.display_info(f"another build is refreshing {ui.name}/; waiting for it")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def node_modules_is_stale(ui: Path) -> bool:
    """Whether the installed dependency tree predates the lockfile beside it.

    npm stamps ``node_modules/.package-lock.json`` on every install, so its
    mtime is when the tree was last installed for real. A checkout that pulled
    a new ``package-lock.json`` over an older install has a tree that cannot
    build the sources next to it, and npm does not notice on its own: it is
    only asked to build.

    Testing for the directory alone was not enough. Upgrading across the TUI
    rewrite is exactly this case -- the sources become the new UI's while
    ``node_modules`` is still the old UI's -- and it failed inside this hook
    with an esbuild resolution error that named neither npm nor this directory.
    """
    if not (ui / "node_modules").is_dir():
        return True

    installed = ui / "node_modules" / ".package-lock.json"
    lock = ui / "package-lock.json"

    if not installed.is_file() or not lock.is_file():
        return True

    return installed.stat().st_mtime < lock.stat().st_mtime
