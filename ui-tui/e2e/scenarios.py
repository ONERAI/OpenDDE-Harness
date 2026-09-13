"""Observation-driven PTY scenarios; shared by run.py and pytest."""

import math
import re
import time

# What the status row offers after one Ctrl+C at an idle empty prompt. Ctrl+C
# is an interrupt, not an exit key, so quitting with it takes two presses.
# run.py reads this from here, so the string lives in one place.
QUIT_HINT = "press ctrl+c again to exit"


def boot(fixture):
    with fixture.launch() as ui:
        ui.ready()
        ui.expect(str(fixture.workspace))
        ui.quit()


def markdown(fixture):
    with fixture.launch() as ui:
        ui.ready()
        mark = len(ui.records())
        ui.submit("e2e markdown")
        ui.expect("A deterministic reply with code.")
        ui.expect("first item")
        ui.expect("E2E_MARKDOWN_DONE")
        ui.idle(mark)
        deltas = [
            r
            for r in ui.records()[mark:]
            if r["kind"] == "rpc_sent" and r["frame"].get("params", {}).get("event", {}).get("type") == "token.delta"
        ]
        if len(deltas) < 2:
            raise AssertionError("Expected actual incremental gateway token events")
        ui.quit()


def tool(fixture):
    with fixture.launch() as ui:
        ui.ready()
        mark = len(ui.records())
        ui.submit("e2e tool")
        ui.expect("read_file")
        ui.expect("E2E_REAL_TOOL_PAYLOAD")
        ui.expect("E2E_TOOL_DONE")
        ui.idle(mark)
        ui.expect_record(lambda r: r["kind"] == "tool_result_verified", "read_file result returned to provider")
        ui.quit()


def thinking(fixture):
    with fixture.launch() as ui:
        ui.ready()
        mark = len(ui.records())
        ui.submit("e2e thinking")
        ui.expect("Thinking")
        ui.expect("E2E_THINKING_DONE")
        ui.idle(mark)
        ui.quit()


def status(fixture):
    with fixture.launch() as ui:
        ui.ready()
        ui.submit("/status")
        ui.expect_record(lambda r: r["kind"] == "rpc_result" and r.get("method") == "session.status", "/status RPC")
        ui.expect("Model: scripted/e2e")
        ui.expect("Tokens:")
        ui.quit()


def help_command(fixture):
    with fixture.launch() as ui:
        ui.ready()
        ui.submit("/help")
        ui.expect("/thinking")
        ui.expect("/resume")
        ui.quit()


def instruction_write(fixture):
    """The agent writing AGENTS.md has to get past the user to do it."""
    with fixture.launch() as ui:
        ui.ready()
        mark = len(ui.records())
        ui.submit("e2e instructions")

        # Writing into the workspace is ordinarily allowed and unremarkable.
        # This file is not ordinary: the next turn would read it as the user's
        # own standing instructions, so it is asked about like a destructive
        # command.
        ui.expect_record(
            lambda r: r["kind"] == "rpc_sent" and r["frame"].get("method") == "approval.request",
            "approval for writing AGENTS.md",
            after=mark,
        )
        ui.wait(
            lambda: "standing instructions" in ui.screen_text and "Deny" in ui.screen_text,
            "a prompt that says what the file becomes",
        )

        target = fixture.workspace / "AGENTS.md"

        if target.exists():
            raise AssertionError("AGENTS.md was written before the user answered")

        ui.keys(b"2")
        ui.expect("denied")

        if target.exists():
            raise AssertionError("A denied write still landed")

        # And nothing was adopted: the session follows no instructions at all.
        ui.submit("/memory")
        ui.expect("No AGENTS.md or ODH.md")
        ui.quit()


def project_instructions(fixture):
    """`/memory`: nothing found, then two files written while the TUI is up."""
    with fixture.launch() as ui:
        ui.ready()
        ui.submit("/memory")
        # The workspace has none yet, and an empty table would read as a
        # failure of the search rather than as an absence of files.
        ui.expect("No AGENTS.md or ODH.md")
        ui.expect("~/.opendde_harness/AGENTS.md")

        # Written now, with the session already open: the files are read again
        # before each turn, so there is nothing to restart.
        (fixture.home / ".opendde_harness" / "AGENTS.md").write_text("Answer in one sentence.\n")
        (fixture.workspace / "AGENTS.md").write_text("This workspace is a fixture. Never write to it.\n")

        ui.submit("/memory")
        ui.expect("~/.opendde_harness/AGENTS.md")
        ui.expect("AGENTS.md")
        ui.expect("last word")

        ui.submit("/memory off AGENTS.md")
        ui.expect("AGENTS.md is off for this session")

        ui.submit("/memory on AGENTS.md")
        ui.expect("AGENTS.md is on for this session")

        ui.submit("/memory off nothing.md")
        ui.expect("no instruction file here is called nothing.md")
        ui.quit()


def queue(fixture):
    with fixture.launch() as ui:
        ui.ready()
        ui.submit("e2e hold")
        ui.expect("E2E_HOLD_STARTED")
        ui.submit("e2e queued")
        ui.expect("Queued: e2e queued")
        if any(r["kind"] == "provider_start" and r.get("rule") == "e2e queued" for r in ui.records()):
            raise AssertionError("Queued request reached the model while the first was gated")
        (fixture.root / "release-hold").touch()
        ui.expect("E2E_HOLD_DONE")
        ui.expect("E2E_QUEUED_DONE")
        ui.wait(
            lambda: (
                sum(
                    r["kind"] == "rpc_sent"
                    and r["frame"].get("params", {}).get("event", {}).get("type") == "message.complete"
                    for r in ui.records()
                )
                >= 2
            ),
            "both completed turns",
        )
        ui.wait(lambda: not ui.working, "queue drained to idle")
        ui.quit()


def cancel(fixture, key=b"\x1b"):
    with fixture.launch() as ui:
        ui.ready()
        mark = len(ui.records())
        ui.submit("e2e slow")
        ui.expect("E2E_SLOW_STARTED")
        ui.keys(key)
        ui.cancelled(mark)
        ui.expect_record(
            lambda r: r["kind"] == "provider_end" and r.get("rule") == "e2e slow" and r.get("outcome") == "cancelled",
            "scripted provider generator cancelled",
        )
        if any(r.get("rule") == "e2e slow" and r.get("outcome") == "complete" for r in ui.records()):
            raise AssertionError("Slow provider finished instead of being cancelled")
        ui.quit()


def ctrl_c(fixture):
    with fixture.launch() as ui:
        ui.ready()
        mark = len(ui.records())
        ui.submit("e2e slow")
        ui.expect("E2E_SLOW_STARTED")

        # One: cancels the turn.
        ui.keys(b"\x03")
        ui.cancelled(mark)
        ui.expect_record(
            lambda r: r["kind"] == "provider_end" and r.get("rule") == "e2e slow" and r.get("outcome") == "cancelled",
            "scripted provider generator cancelled",
        )
        if any(r.get("rule") == "e2e slow" and r.get("outcome") == "complete" for r in ui.records()):
            raise AssertionError("Slow provider finished instead of being cancelled")

        # Two: the turn has ended, so this arrives at an idle empty prompt.
        # That is the shape that used to exit the app outright, one press after
        # an ordinary cancel. It has to offer instead.
        ui.keys(b"\x03")
        ui.wait(lambda: QUIT_HINT in ui.screen_text, "the offer to exit")
        if ui.exit_code is not None:
            raise AssertionError("A Ctrl+C at an idle prompt exited instead of offering to")

        # Three: takes the offer.
        ui.quit()


def resume(fixture):
    with fixture.launch() as ui:
        ui.ready()
        mark = len(ui.records())
        ui.submit("e2e markdown")
        ui.expect("E2E_MARKDOWN_DONE")
        ui.idle(mark)
        opened = ui.expect_record(
            lambda r: r["kind"] == "rpc_result" and r.get("method") == "session.create", "new session id"
        )
        session_id = opened["frame"]["result"]["session_id"]
        ui.quit()
    with fixture.launch(resume=session_id) as ui:
        ui.ready()
        ui.expect("E2E_MARKDOWN_DONE")
        record = ui.expect_record(
            lambda r: r["kind"] == "rpc_result" and r.get("method") == "session.resume", "resumed messages"
        )
        if record["frame"]["result"]["session_id"] != session_id:
            raise AssertionError("Resume silently minted a different session")
        if any(r["kind"] == "provider_start" for r in ui.records()):
            raise AssertionError("Resume regenerated the answer instead of replaying saved messages")
        ui.quit()


def approval(fixture, choice="deny"):
    rule = "e2e approval" if choice == "deny" else "e2e allow approval"
    with fixture.launch() as ui:
        ui.ready()
        # Repeating denial checks that closing one prompt restores the editor
        # and does not leave the broker or scripted turn at an old step.
        for _ in range(2 if choice == "deny" else 1):
            mark, text_mark = len(ui.records()), len(ui.text)
            ui.submit(rule)
            requested = ui.expect_record(
                lambda r: r["kind"] == "rpc_sent" and r["frame"].get("method") == "approval.request",
                "shell approval",
                after=mark,
            )["frame"]["params"]
            ui.wait(
                lambda: all(
                    text in ui.screen_text for text in ("Approval required", "Allow once", "Deny", "1/2 choose")
                ),
                "inline approval choices and key hints",
            )

            def seconds():
                match = re.search(r"Approval required[^\n]*\((\d+)s\)", ui.screen_text)
                return int(match[1]) if match else None

            ui.wait(lambda: seconds() is not None, "approval countdown")
            remaining = seconds()
            expected = math.ceil(requested["expires_at"] - time.time())
            if not 1 <= remaining <= math.ceil(requested["expires_at"] - requested["created_at"]) + 1:
                raise AssertionError(f"Invalid approval countdown: {remaining}")
            if abs(remaining - expected) > 1:
                raise AssertionError(f"Countdown {remaining} disagrees with expires_at ({expected}s left)")
            ui.wait(lambda: seconds() is not None and 0 < seconds() < remaining, "countdown visibly decreases")
            if not fixture.approval_target.exists():
                raise AssertionError("The shell command ran before approval")

            ui.keys(b"2" if choice == "deny" else b"1")
            response = ui.expect_record(
                lambda r: r["kind"] == "rpc_request" and r["frame"].get("method") == "approval.respond",
                "approval choice received by gateway",
                after=mark,
            )["frame"]
            params = response["params"]
            if params != {
                "approval_id": requested["approval_id"],
                "session_id": requested["conversation_id"],
                "choice": choice,
            }:
                raise AssertionError(f"Wrong approval response: {params}")
            ui.expect_record(
                lambda r: (
                    r["kind"] == "rpc_result"
                    and r.get("method") == "approval.respond"
                    and r["frame"].get("id") == response["id"]
                    and r["frame"].get("result") == {"ok": True}
                ),
                "approval accepted by broker",
                after=mark,
            )
            ui.expect_record(
                lambda r: (
                    r["kind"] == "rpc_sent"
                    and r["frame"].get("method") == "approval.closed"
                    and r["frame"].get("params", {}).get("approval_id") == requested["approval_id"]
                    and r["frame"]["params"].get("reason") == choice
                ),
                "approval broker cleanup",
                after=mark,
            )
            completed = ui.event("tool.complete", after=mark)["frame"]["params"]["event"]["payload"]
            if completed["tool_call_id"] != requested["tool_call_id"]:
                raise AssertionError("Approval completed a different tool call")
            if choice == "deny":
                ui.expect("User denied this command", after=text_mark)
                ui.expect("no alternative method will be attempted", after=text_mark)
                if not fixture.approval_target.exists():
                    raise AssertionError("Denied command deleted the fixture")
                if any(r.get("rule") == rule and r.get("step") == 1 for r in ui.records()[mark:]):
                    raise AssertionError("Denied command continued to another model step")
            else:
                ui.expect("E2E_APPROVAL_ALLOW_DONE", after=text_mark)
                ui.expect_record(
                    lambda r: r["kind"] == "tool_result_verified" and r.get("rule") == rule,
                    "successful exec result returned to provider",
                    after=mark,
                )
                if fixture.approval_target.exists():
                    raise AssertionError("Allowed shell command did not delete the fixture")
            ui.idle(mark)
            ui.wait(lambda: "Approval required" not in ui.screen_text, "approval prompt removed from editor")
            replies = [
                r
                for r in ui.records()[mark:]
                if r["kind"] == "rpc_request" and r["frame"].get("method") == "approval.respond"
            ]
            if len(replies) != 1:
                raise AssertionError("Approval was answered more than once")
        ui.quit()


def approval_allow(fixture):
    approval(fixture, choice="allow")


def clarify(fixture):
    with fixture.launch() as ui:
        ui.ready()
        mark, text_mark = len(ui.records()), len(ui.text)
        ui.submit("e2e clarify")
        requested = ui.expect_record(
            lambda r: r["kind"] == "rpc_sent" and r["frame"].get("method") == "clarify.request",
            "real ask_user question",
            after=mark,
        )["frame"]["params"]
        ui.wait(
            lambda: all(
                text in ui.screen_text
                for text in (
                    "The agent needs an answer",
                    "E2E_CLARIFY_QUESTION",
                    "E2E_FIRST_CHOICE",
                    "E2E_SECOND_CHOICE",
                    "Other",
                    "enter answer",
                )
            ),
            "inline clarification and choices",
        )
        ui.keys(b"\x1b[B\r")
        response = ui.expect_record(
            lambda r: r["kind"] == "rpc_request" and r["frame"].get("method") == "clarify.respond",
            "clarification answer received by gateway",
            after=mark,
        )["frame"]
        if response["params"] != {"request_id": requested["request_id"], "answer": "E2E_SECOND_CHOICE"}:
            raise AssertionError(f"Wrong clarification response: {response['params']}")
        ui.expect_record(
            lambda r: (
                r["kind"] == "rpc_result"
                and r.get("method") == "clarify.respond"
                and r["frame"].get("id") == response["id"]
                and r["frame"].get("result") == {"ok": True}
            ),
            "clarification accepted by broker",
            after=mark,
        )
        ui.expect("answered: E2E_SECOND_CHOICE", after=text_mark)
        ui.expect("E2E_CLARIFY_DONE", after=text_mark)
        ui.expect_record(
            lambda r: r["kind"] == "tool_result_verified" and r.get("rule") == "e2e clarify",
            "ask_user answer returned to provider",
            after=mark,
        )
        ui.idle(mark)
        ui.wait(lambda: "The agent needs an answer" not in ui.screen_text, "clarification prompt removed")
        sends = [
            r for r in ui.records()[mark:] if r["kind"] == "rpc_request" and r["frame"].get("method") == "turn.send"
        ]
        if len(sends) != 1:
            raise AssertionError("Clarification answer leaked into a new chat turn")
        ui.quit()


def error(fixture):
    with fixture.launch() as ui:
        ui.ready()
        mark = len(ui.records())
        ui.submit("e2e failure")
        ui.expect("E2E_SCRIPTED_FAILURE")
        ui.idle(mark)
        mark = len(ui.records())
        ui.submit("e2e markdown")
        ui.expect("E2E_MARKDOWN_DONE")
        ui.idle(mark)
        ui.quit()


def fullscreen(fixture):
    """The default mode: the alternate screen, with the conversation handed to
    the shell's own buffer on the way out."""
    with fixture.launch(fullscreen=True) as ui:
        ui.ready()
        if not ui.alt_screen_entries:
            raise AssertionError("The default mode never entered the alternate screen")
        mark = len(ui.records())
        ui.submit("e2e markdown")
        ui.expect("E2E_MARKDOWN_DONE")
        ui.idle(mark)
        if "E2E_MARKDOWN_DONE" not in ui.screen_text:
            raise AssertionError(f"The reply is not on the alternate screen:\n{ui.screen_text}")
        entries = ui.alt_screen_entries
        ui.quit()
        replayed = ui.main_buffer_after_alt_exit()
        if "E2E_MARKDOWN_DONE" not in replayed:
            raise AssertionError(f"The conversation was not left in the shell's buffer:\n{replayed[-3000:]}")
        if "e2e markdown" not in replayed:
            raise AssertionError(f"The prompt was not left in the shell's buffer:\n{replayed[-3000:]}")
        if ui.alt_screen_entries != entries:
            raise AssertionError("Exiting re-entered the alternate screen")


# Four detached design tasks, newest snapshot first. The bar draws three of
# them and folds the rest into a "+N more", so four is the smallest fixture
# that shows both halves of that rule. The order is the monitor's: rows are
# sorted by the snapshot's own mtime, which the fixture pins, so which three
# are drawn is a fact this scenario can hold it to.
DESIGN_TASKS = [
    ("e2eaaaa1", "E2E_TARGET_ALPHA"),
    ("e2ebbbb2", "E2E_TARGET_BRAVO"),
    ("e2ecccc3", "E2E_TARGET_CHARLIE"),
    ("e2edddd4", "E2E_TARGET_DELTA"),
]


def design_tasks(fixture):
    base = time.time()
    for index, (task_id, target) in enumerate(DESIGN_TASKS):
        fixture.design_task(
            task_id,
            target=target,
            cycle=index + 1,
            mtime=base - index,
            log=f"E2E_WORKER_LOG {task_id} first line\nE2E_WORKER_LOG {task_id} last line\n",
        )

    with fixture.launch() as ui:
        ui.ready()

        # Nothing switches the bar on. Asking what is running starts watching,
        # and the bar draws itself because there is something to draw.
        ui.submit("/tasks")
        ui.expect("Protein design tasks")
        ui.wait(lambda: "Design tasks (4)" in ui.screen_text, "the bar counting four active tasks")
        ui.wait(lambda: "+1 more" in ui.screen_text, "the fourth task folded into a +N more")

        # Newest three by snapshot mtime, in that order, and the oldest folded
        # away. Nothing but the bar has printed a target yet, so the screen is
        # a fair account of the rows it drew.
        for _, target in DESIGN_TASKS[:3]:
            ui.wait(lambda target=target: target in ui.screen_text, f"a bar row for {target}")
        rows = [line.strip() for line in ui.screen_text.splitlines() if " · Design Cycle" in line]
        if [row.split(" · ")[0] for row in rows] != [target for _, target in DESIGN_TASKS[:3]]:
            raise AssertionError(f"The bar drew {rows} instead of the three newest snapshots in order")

        # The list above named every task, drawn or folded, including the one
        # the bar folded into its "+N more".
        for _, target in DESIGN_TASKS:
            ui.expect(target)

        # `/tasks <id>` is one task in full, out of its own snapshot.
        ui.submit(f"/tasks {DESIGN_TASKS[0][0]}")
        ui.expect(f"id: {DESIGN_TASKS[0][0]}")
        ui.expect("state: running (snapshot)")
        ui.expect("progress: 1/8")
        ui.expect("phase: Design Cycle")
        ui.expect("best objective: 0.8125")
        ui.expect("compute: e2e-worker")

        # `/task:logs` reads the worker log here rather than shelling out.
        ui.submit(f"/task:logs {DESIGN_TASKS[0][0]}")
        ui.expect(f"E2E_WORKER_LOG {DESIGN_TASKS[0][0]} last line")

        # The folded task ends. The bar loses its count and its "+N more"
        # without being told to, and says so once in the transcript.
        fixture.design_task(DESIGN_TASKS[3][0], target=DESIGN_TASKS[3][1], status="completed", cycle=8, mtime=base + 10)
        ui.wait(lambda: "Design tasks (3)" in ui.screen_text, "the bar after one task finished")
        ui.wait(lambda: "+1 more" not in ui.screen_text, "the +N more gone with three tasks left")
        ui.expect(f"protein design {DESIGN_TASKS[3][1]} ({DESIGN_TASKS[3][0]}) completed")

        # The rest end, and the bar empties itself rather than waiting to be
        # hidden. The finished tasks stay in `/tasks`.
        for index, (task_id, target) in enumerate(DESIGN_TASKS[:3]):
            fixture.design_task(task_id, target=target, status="completed", cycle=8, mtime=base + 20 + index)
        ui.wait(lambda: "Design tasks (" not in ui.screen_text, "an empty bar once every task ended")
        ui.submit("/tasks")
        for _, target in DESIGN_TASKS:
            # A row the list has never printed before: the status word and the
            # target in the same row, which only the finished list produces.
            ui.expect(f"completed {target}")
        ui.quit()


SCENARIOS = {
    "boot": boot,
    "fullscreen": fullscreen,
    "markdown": markdown,
    "tool": tool,
    "thinking": thinking,
    "status": status,
    "help": help_command,
    "instruction_write": instruction_write,
    "project_instructions": project_instructions,
    "queue": queue,
    "design_tasks": design_tasks,
    "escape_cancel": cancel,
    "ctrl_c": ctrl_c,
    "resume": resume,
    "approval": approval,
    "approval_allow": approval_allow,
    "clarify": clarify,
    "error": error,
}
