# pi-tui end-to-end harness

These tests run the **real Python CLI launcher, authenticated TCP gateway,
dispatcher, AgentLoop, built-in tools, session persistence, and pi-tui frontend**
under a 120-column × 40-row pseudo-terminal. Only model generation is scripted.
No human terminal or model-provider credentials are needed.

## Run

From the repository root, with Node >=22.19, npx and the existing frontend
dependencies installed (`cd ui-tui && npm ci`):

```sh
PYTHONPATH=. /root/.venvs/opendde-harness-beta/bin/python \
  ui-tui/e2e/run.py --scenario tool --artifacts /tmp/tui-e2e-tool

PYTHONPATH=. /root/.venvs/opendde-harness-beta/bin/python \
  ui-tui/e2e/run.py --scenario all --artifacts /tmp/tui-e2e-all

PYTHONPATH=. /root/.venvs/opendde-harness-beta/bin/python -m pytest \
  tests/test_tui_e2e.py -m e2e -q -s --durations=12
```

`--scenario` choices are `boot`, `fullscreen`, `markdown`, `tool`, `thinking`,
`status`, `help`, `queue`, `design_tasks`, `escape_cancel`, `ctrl_c`, `resume`,
`approval`, `approval_allow`, `clarify`, and `error`.
Pytest parametrizes the same scenarios. The existing `e2e` marker excludes them
from the normal suite; no pyproject/conftest change is needed.

Python needs the project's dev dependencies, including pyte. The real token
estimator also needs its public `cl100k_base` vocabulary already cached. The
driver copies that asset from `$TIKTOKEN_CACHE_DIR`, `$DATA_GYM_CACHE_DIR`, or
`/tmp/data-gym-cache` into each fixture. It fails clearly if the asset is absent;
it never downloads it. This is tokenizer data, not an OAuth or model credential.
Persistent fixture files and logs stay in the chosen artifact directory. The default
is a newly created `/tmp/tui-e2e-*` directory. Tests retain artifacts on
success as well as failure.

Each PTY launch uses a separate short `TMPDIR` created as `/tmp/e2e-*`, independent
of the artifact path. tsx creates its IPC socket below TMPDIR, and deeply nested
pytest artifact paths can exceed the Unix socket address limit (108 bytes on
Linux). The harness removes this temporary directory after PTY teardown on both
success and failure; the retained terminal JSON records its path as `tmpdir`.

To validate a committed revision while other people edit the working tree:

```sh
git worktree add --detach /tmp/tui-e2e-head c4dd899
ln -s "$PWD/ui-tui/node_modules" /tmp/tui-e2e-head/ui-tui/node_modules

PYTHONPATH=. /root/.venvs/opendde-harness-beta/bin/python \
  ui-tui/e2e/run.py --repo /tmp/tui-e2e-head --scenario all

TUI_E2E_REPO=/tmp/tui-e2e-head PYTHONPATH=. \
  /root/.venvs/opendde-harness-beta/bin/python -m pytest \
  tests/test_tui_e2e.py -m e2e -q -s
```

`--repo` chooses the **product** source; the harness still comes from this
directory. Results record the commit, working-tree status, TUI source hash
before/after, and harness file hashes. A changed source hash identifies a run
that crossed concurrent edits. No checkout file is copied over or modified.

## Startup failures

`boot_pty.py` is a second, much smaller driver beside `run.py`. It launches
`src/entry.ts` alone, under its own pseudo-terminal, against a Unix socket that
answers JSON-RPC frames however the scenario wants — because what it covers is
a boot that fails *after* the renderer has taken the terminal, which the real
gateway has no reason to do. `boot_failure` refuses the handshake and expects
exit 3; `signal_during_boot` stalls it and sends SIGTERM once the alternate
screen is up, expecting exit 143. Both assert the same thing: the alternate
screen is left, autowrap is turned back on, and the canonical and echo flags
are as they were lent.

```sh
/root/.venvs/opendde-harness-beta/bin/python ui-tui/e2e/boot_pty.py

PYTHONPATH=. /root/.venvs/opendde-harness-beta/bin/python -m pytest \
  tests/test_tui_boot_pty.py -m e2e -q -s
```

## Screen mode

The product draws on the **alternate screen** by default. Scenario assertions
read the accumulated main-buffer transcript, which an alternate-screen session
does not produce, so every launch sets `OPENDDE_HARNESS_TUI_FULLSCREEN=0` and
gets the main screen. `Fixture.launch(fullscreen=True)` asks for the default
instead and sets it to `1`; only the `fullscreen` scenario does that.

That scenario is the one that covers the default: it checks the alternate
screen is entered at startup, that the reply is on the alternate screen while
the session runs, and that quitting leaves the alternate screen and writes the
conversation into the shell's own buffer. Boot and the exit each reset terminal
modes and emit a bare `1049l` of their own, so `PtySession.main_buffer_after_alt_exit`
takes its window from the **last** alternate-screen entry rather than from the
last exit.

## Isolation and injection

`run.py` creates a fresh HOME, config, workspace, gates and session store for
each scenario. It launches:

```text
<current Python> -m opendde_harness tui --dev
```

This is the console command's real entrypoint (`commands.run`). The loader has
no `OPENDDE_HARNESS_CONFIG` variable: the fixture uses the normal
`HOME/.opendde_harness/config.json` location and its explicit `set_config_path`
API in the test bootstrap.

For that child and its Python CLI workers only, PYTHONPATH includes `e2e/`.
Python loads `sitecustomize.py`, guarded by `OPENDDE_HARNESS_E2E_SCRIPT`. It:

1. Registers a test-only `scripted` ProviderSpec and supplies ProviderConfig to
   the CLI's schema lookup for that name. The latter is needed because the
   provider-list/startup gate otherwise rejects an extra provider unknown to
   LiteLLM, even if it exists in the runtime registry. Credential validation
   and the real startup gate still run; they are not stubbed to return true.
2. Replaces `cli._helpers.make_provider` with a factory
   that **only accepts the scripted route**. This also avoids unrelated
   LiteLLM prewarming. ProviderPool and the real AgentLoop use that factory.
3. Observes real dispatcher results and outgoing RPC frames in `journal.jsonl`;
   it does not fabricate gateway events or tool results.
4. Disables release-update polling and rejects non-loopback Python socket/DNS
   operations. A blocked attempt or unmatched script rule fails the scenario.

An injection error exits Python with code 90, rather than silently continuing
with a real provider. Normal processes do not import this module: `e2e/` is not
on their Python path, and the extra environment guard is absent.

The driver supplies a small environment rather than inheriting provider keys,
proxy settings, OAuth paths or the user's HOME. Memory/compute plugins, MCP,
the skill catalogue, remote compaction and checkpointing are disabled in the
temporary config. **Checkpointing is `never`, so even shadow-git commits are
disabled.** npm is put in offline mode and Node auto-install is disabled.

This is a test fixture, not an OS sandbox for arbitrary scripts. Scripts are
trusted test code. The shipped scenarios call only `read_file`, `ask_user`, and an
approval-gated shell operation against temporary paths; they do not contact
model endpoints or run compute jobs.

## What is asserted

- Boot renders the model and actual workspace and subscribes through the real
  gateway. Markdown arrives as multiple token events and is rendered without
  literal bold/code delimiters.
- The file-read scenario renders the tool panel/result. Its second model step
  verifies that AgentLoop returned the real fixture file's content.
- Thinking produces the collapsed label. `/status` runs its real RPC/CLI path;
  `/help` renders the local registry.
- A file gate holds a turn while the next prompt is visibly queued. Releasing
  it must produce the two completed turns in sequence.
- Esc and Ctrl+C cancel a gated provider generator. The Ctrl+C scenario uses
  three presses and asserts what each one does: cancel the turn, then offer to
  exit rather than exiting, then take the offer. The middle press is the bug
  the ladder exists for, an idle empty prompt one press after a cancel. Every
  other scenario ends through the same two-press quit.
- Resume starts a second real process against the same temporary session
  store. It must reopen the same ID and display the saved answer **without
  another provider call**.
- A scripted nonretryable error must be visible, then a new query succeeds.
- Normal exits must be zero and restore canonical/echo/signal terminal flags.
- In the default alternate-screen mode, exiting leaves that screen and replays
  the conversation into the shell's scrollback.
- `design_tasks` writes four task directories under the fixture's own task root
  through the plugin's `TaskSnapshot` contract, so the fixture cannot drift from
  the file a detached worker writes. No worker runs and no compute is reserved.
  The bar starts hidden; `/tasks show` and `/tasks hide` open and close it,
  `/tasks` lists every task and `/task <id>` and `/task logs <id>` read one.
  The bar counts all four active tasks, draws the newest three by snapshot
  mtime in that order and folds the rest into a `+N more`, then loses the fold
  and finally empties itself as the snapshots turn `completed`, announcing each
  ending once. The fixture pins each snapshot's mtime, which is what the monitor
  records as the task's last observation and sorts its rows on.

### Approval and clarification

The built-in direct `exec` tool reaches ApprovalBroker through the normal shell
policy: the script asks for `rm -- <disposable fixture file>`. Both approval
scenarios assert the current screen shows **Approval required**, **Allow once**,
**Deny**, and the digit-key hint. They compare the displayed seconds against the
broker's `expires_at`, then wait for a visible countdown decrease before answering.

`approval` presses `2` to deny, checks the exact `approval.respond` request and
successful RPC result, matches `approval.closed` by ID/reason, and verifies the
transcript reports denial and ends the action without another model step. The
fixture file must survive. It repeats in the same session to verify prompt and
broker cleanup. `approval_allow` presses `1`, checks the same round-trip, and
requires the file to disappear, an exit-code-zero tool result to reach the scripted
provider, and the model's final completion sentinel. Only the fixture file can be
deleted; no owner or repository files are involved.

`clarify` scripts the real `ask_user` tool. QuestionBroker emits `clarify.request`,
the inline prompt displays its choices and **Other**, and Down/Enter selects the
second choice. The scenario checks the exact `clarify.respond` answer, broker
acceptance, the rendered tool answer, its return to the provider, final completion,
and prompt dismissal. Answering must not send an extra `turn.send`. Clarification
has no countdown because this broker notification supplies no `expires_at`.

Auto-resume cannot be enabled through the current four-key `config.get`; this
harness tests the explicit `OPENDDE_HARNESS_TUI_RESUME` contract.

## Add a scenario

Add a rule to `script.json` and a function to `scenarios.py`, then register it
in `SCENARIOS`. Pytest and the CLI discover the same map automatically:

```python
def example(fixture):
    with fixture.launch() as ui:
        ui.ready()
        mark = len(ui.records())
        ui.submit("e2e example")
        ui.expect("E2E_EXAMPLE_DONE")
        ui.idle(mark)
        ui.quit()
```

A rule matches the most recent user text. Its first response is step zero;
each assistant tool-call round after that user message advances one step.
The step is derived from real history, so a new user turn (including a retry
after cancellation) starts at zero. Exhausting a script is an error.
Supported fields:

- `chunks`: ordered `{text?, thinking?, delay_ms?}` deltas.
- `tool_calls`: `{name, arguments}`; AgentLoop executes the actual registered
  tool. `expect_tool_result` on the next step verifies its input history.
- `gate`: absolute fixture-file path to wait for; `after_gate`: text chunks
  emitted when released. Waits are asynchronous/cancellable and bounded.
- `error`: a deterministic nonretryable provider error.

`{{fixture_file}}`, `{{hold_gate}}`, `{{slow_gate}}`, and `{{approval_command}}`
are substituted by Fixture. Add another substitution there when a scenario
needs its own file. Do not put real secrets or host paths in scripts.

Use `ui.expect`, `ui.expect_record`, `ui.event`, and `ui.wait` to schedule keys
from evidence. Do not add fixed sleeps to scenario functions. Provider delays
simulate streaming, and file gates control the important ordering boundaries.
Use response sentinels different from the prompt: otherwise an input echo can
make a broken test pass. `ui.keys` sends raw bytes (Esc is `b"\x1b"`, Ctrl+C is
`b"\x03"`); `ui.submit` pastes the body and sends a real Enter key.

## Artifacts and failures

Each scenario gets its own subdirectory containing config/workspace/session
files, expanded script, the RPC/provider journal, and for each launch:

- `terminal-N.bin`: untouched terminal bytes.
- `terminal-N.txt`: ANSI-stripped accumulated transcript.
- `terminal-N.screen.txt`: final pyte screen (used for idle-border checks).
- `terminal-N.json`: actual command, timed key schedule and exit code.
- `result.json`: status, timing, source/harness provenance and failure reason.

The accumulated transcript includes intermediate redraws. Use `ui.screen_text`
for assertions about what is currently visible, especially negative assertions;
use journal events to distinguish rendered output from a typed-input echo.

The CLI also writes `summary.json`. The gateway's own log is under
`home/.opendde_harness/logs/tui.log`. A failed scenario terminates/reaps only
its own PTY process group, with a bounded TERM/KILL cleanup; artifacts remain.
The driver does not reset shared processes or rewrite product code to make a
scenario pass. Default timeout is 20 seconds; `--timeout` changes the per-wait
deadline, not a sleep schedule.
