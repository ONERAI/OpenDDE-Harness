"""The host's own memory writer: the default owner of ``user.md`` / ``episodes.md``.

The self-evolution loop the project shipped before the plugin contract existed,
restored on the plumbing that replaced it. Two writes, in this order, and both
off the turn's critical path:

* **annotate** -- every completed turn the extraction outbox hands over is
  summarized into one or more tagged episode lines and appended to
  ``episodes.md``. The tag vocabulary is the trigger for the second write.
* **refresh** -- when a tag has gained :data:`REFRESH_HOT_TAG_THRESHOLD`
  episodes since the last time its section was rewritten, the matching H2 of
  ``user.md`` is rewritten from that tag's recent episodes and nothing else.
  Every other section is left byte for byte as it was, and the write is a
  compare-and-set against the profile the rewrite was computed from.

The split is the point. Annotation is cheap and runs always; the profile rewrite
is expensive and runs on evidence. Both prompts, the tool schemas and the tag
vocabulary are what the released consolidator used -- this is the same behaviour,
reached through :class:`~opendde_harness.memory_engine.backend.MemoryBackend`
rather than through a consolidator the loop drove itself.

Why it is a backend and not a second pipeline: a workspace has one owner of
automatic durable extraction. Configure ``memory.backend`` and a plugin is that
owner; leave it unset and this is, which is what a default install runs with.
Nothing here is reachable when a plugin backend is wired.

The model is the session's. Outside a turn it is the binding the host booted on;
inside one -- and inside the drain task a turn spawned, which copies the turn's
context -- it is whatever that turn is running on, so an annotation follows a
``/model`` switch the way the rest of the turn does.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from opendde_harness.memory_engine.backend import Memory
from opendde_harness.memory_engine.consolidate.consolidator import (
    FORESIGHT_HEADING,
    drop_bullets_without_src,
)
from opendde_harness.providers import messages as msg
from opendde_harness.providers.binding import ModelBinding, resolve

if TYPE_CHECKING:
    from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore

#: How many new episodes a tag needs before its profile section is rewritten.
#: Below it episodes still accumulate and ``user.md`` is left alone -- the
#: threshold is what keeps one throwaway mention from rewriting the profile.
REFRESH_HOT_TAG_THRESHOLD: int = 5

#: How many of a tag's most recent episodes the section rewrite reads.
REFRESH_MAX_EPISODES: int = 50

#: Bullets past which a rewritten section is reported as drifting. Not truncated
#: -- dropping content to satisfy a schema is worse than saying the schema was
#: missed -- but said out loud, because a profile this long is a diary.
SECTION_BULLET_DRIFT: int = 15


def _build_annotate_tool(*, enable_foresight: bool) -> list[dict]:
    """Construct the ``annotate_conversation`` tool.

    The ``foresight_hint`` slot is included only when ``enable_foresight``
    is True. With the flag off, the LLM isn't asked for predictions at
    all — saves tokens and simplifies the prompt.
    """
    properties: dict[str, dict] = {
        "episode_summary": {
            "type": "array",
            "description": (
                "One entry per distinct event. Each entry is a SINGLE LINE "
                "(no newlines), formatted exactly as:\n"
                "  '[YYYY-MM-DD HH:MM] <summary, <=100 chars> #tag1 #tag2'\n\n"
                "SUMMARY — must include concrete identifiers (file paths, "
                "function names, PR/issue numbers, percentages, time/size "
                "values). Generic descriptions waste a slot.\n"
                "  GOOD: 'PR #1287 merged: require_auth(scope) replaces 6 "
                "sites in api/views/+middleware/'\n"
                "  BAD:  'User worked on auth refactor'\n\n"
                "TAGS — 1-4 tags per entry, kebab-case. Two CLASSES:\n"
                "  (A) CONTENT tags — name WHAT the episode is about. "
                "Every episode MUST carry at least one content tag. "
                "Use existing slugs from the 'tags you've recently used' "
                "list (see prompt) before inventing new ones — DO NOT "
                "split one project across multiple slugs like "
                "#project-clawtrack-release / -docs / -cli; pick ONE "
                "stable slug per project. New project tags follow "
                "'#project-<work-slug>' where slug names the WORK, not "
                "the codebase. Other content tags: {#perf, #bug, "
                "#decision, #blocker, #deferred, #pivot, #pr, #review, "
                "#rfc, #design, #infra, #sql, #ml}.\n"
                "  (B) PROCESS tags — {#question, #habit, #answer} "
                "describe HOW the user is interacting, not WHAT about. "
                "They are SUFFIXES only — NEVER the primary tag. "
                "An episode tagged ONLY '#question' or ONLY '#habit' is "
                "INVALID and will be rejected. Always pair with at "
                "least one content tag.\n"
                "  AVOID '#task' entirely — it's meaningless filler.\n\n"
                "Order entries by conversation timestamp. Empty array only "
                "when the chunk produced no substantive event."
            ),
            "items": {"type": "string"},
        },
    }
    required: list[str] = ["episode_summary"]
    description = (
        "Annotate this conversation chunk for episodic memory. Produces "
        "tagged episode lines. Does NOT update the user profile — that "
        "happens separately via refresh_profile_section when tag "
        "frequency warrants a focused rewrite."
    )
    if enable_foresight:
        properties["foresight_hint"] = {
            "type": "array",
            "description": (
                "Predictions / behavioral patterns inferred from this "
                "conversation. Fill when ANY of these signals present:\n"
                "(a) User explicitly defers a task ('I'll come back to "
                "X tomorrow', 'next sprint').\n"
                "(b) A recurring pattern visible across 2+ episodes "
                "(e.g. Saturday runs across multiple weeks → predict "
                "next Saturday run; Sunday-night planning → predict "
                "next Sunday planning). Look back at the 'tags you've "
                "recently used' list — if it shows recurring habits, "
                "emit them as foresight.\n"
                "(c) User commits to a specific future action with "
                "time anchor ('I'll write RFC tomorrow', "
                "'release Monday 9am').\n"
                "(d) An upcoming dated event mentioned in conversation "
                "('birthday 5/25', 'demo next Friday', 'deadline EOM').\n"
                "Empty array ONLY if none of (a)-(d) signals present. "
                "Default lean: emit foresight when reasonable — a "
                "low-confidence prediction is more useful than no "
                "prediction. Aim for 1-3 entries per substantive "
                "annotate call."
            ),
            "items": {
                "type": "object",
                "required": [
                    "prediction",
                    "window",
                    "confidence",
                    "src_ts",
                ],
                "properties": {
                    "prediction": {"type": "string"},
                    "window": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "src_ts": {"type": "string"},
                },
            },
        }
        required.append("foresight_hint")
        description = (
            "Annotate this conversation chunk for episodic memory. "
            "Produces tagged episode lines and foresight predictions. "
            "Does NOT update the user profile — that happens separately "
            "via refresh_profile_section when tag frequency warrants a "
            "focused rewrite."
        )
    return [
        {
            "type": "function",
            "function": {
                "name": "annotate_conversation",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    ]


_REFRESH_SECTION_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "refresh_profile_section",
            "description": (
                "Rewrite ONE H2 section of user.md given recent episodes "
                "tagged with a specific topic. Other H2 sections are left "
                "untouched by the splicer — do not include their content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_heading": {
                        "type": "string",
                        "description": (
                            "Exact H2 heading line to replace, e.g. "
                            "'## Projects' or '## Habits'. Must include "
                            "the leading '## '. If the topic naturally fits "
                            "inside an existing H2 (e.g. tag #project-b -> "
                            "'## Projects'), use that existing heading and "
                            "structure project-specific content under H3 in "
                            "the body. Only create a new H2 if no existing "
                            "section fits."
                        ),
                    },
                    "section_body": {
                        "type": "string",
                        "description": (
                            "New markdown body for this section, NOT "
                            "including the heading line itself. Every bullet "
                            "MUST end with '[src: episodes.md @ "
                            "YYYY-MM-DD HH:MM]'. H3/H4 sub-headings are "
                            "allowed within the body."
                        ),
                    },
                },
                "required": ["section_heading", "section_body"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize a tool-call payload value to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _tool_args(args: Any) -> dict[str, Any] | None:
    """The tool call's arguments as a dict, or None when they are not one.

    A model layer hands these back parsed; a stringified payload and a
    single-element list are both shapes real providers have produced, so both are
    accepted rather than dropped as a failed annotation.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


def _stamp(message: dict[str, Any]) -> str:
    """A message's time as the annotation prompt reads it: ``YYYY-MM-DDTHH:MM``.

    pi stamps a message in epoch milliseconds. A message with no stamp -- one
    the host synthesised -- reads ``?`` rather than being given the wall clock,
    which would date it to whenever the drain got to it.
    """
    raw = message.get("timestamp")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return "?"
    try:
        return datetime.fromtimestamp(raw / 1000).strftime("%Y-%m-%dT%H:%M")
    except (OSError, OverflowError, ValueError):
        return "?"


def _format_messages(messages: list[dict[str, Any]]) -> str:
    """Render a turn's messages as the annotation prompt's transcript.

    One line per message: its stamp, its role, the tools it called, and what it
    said. Thinking is left out (it is not what happened, only how it was
    decided), images are left out, and the system prefix is left out -- it is the
    same on every turn and would be most of what the annotator reads.
    """
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "")
        if not role or role in msg.SYSTEM_ROLES:
            continue
        text = msg.text_of(message).strip()
        called = [str(block.get("name") or "") for block in msg.tool_calls_of(message)]
        called = [name for name in called if name]
        tools = f" [tools: {', '.join(called)}]" if called else ""
        if not text and not tools:
            continue
        lines.append(f"[{_stamp(message)}] {role.upper()}{tools}: {text}")
    return "\n".join(lines)


class HostMarkdownBackend:
    """The default :class:`~opendde_harness.memory_engine.backend.MemoryBackend`: the host's two markdown files.

    ``store`` annotates and refreshes; ``recall`` returns the profile block the
    system prompt carries. ``feedback`` is a no-op -- a markdown profile has no
    skill confidence to update -- and so are ``start`` / ``stop``: the files are
    opened per write, under the store's lock, so there is no connection to hold.
    """

    #: This backend owns ``user.md`` on both sides: it writes the file and its
    #: ``recall`` is where the profile reaches the prompt. The memory segment
    #: reads this to know not to also dump the file itself, which would put the
    #: profile in the prompt twice.
    owns_profile = True

    #: There is no agent-track store behind a markdown profile, so an
    #: ``agent_id`` recall can only ever answer empty. The skill router reads this
    #: and does not wire a source it would query every turn for nothing.
    agent_track = False

    def __init__(
        self,
        store: "MemoryStore",
        *,
        binding: ModelBinding,
        enable_foresight: bool = False,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        #: What to annotate with outside a turn. Inside one, the turn's binding
        #: answers instead -- see :func:`~opendde_harness.providers.binding.resolve`.
        self._binding = binding
        self._enable_foresight = enable_foresight
        self._now_fn = now_fn or datetime.now

    # ── MemoryBackend: the read side ───────────────────────────────────

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        top_k: int,
    ) -> list[Memory]:
        """The profile's relevant sections, as one block the prompt carries verbatim.

        The agent track is empty by design: a markdown profile holds what is true
        about the user, and has no case library or skill store to answer an
        agent-track query with. Neither id set, or both, is a caller bug and
        answers empty as the Protocol says.

        ``top_k`` sizes an index's hit list and does not apply here -- the
        profile is one document, and how much of it is relevant is decided by the
        store's own section budget, which is what put two sections plus ``##
        Notes`` in the prompt before any backend existed.
        """
        if (user_id is None) == (agent_id is None):
            return []
        if agent_id is not None:
            return []
        block = self._store.get_memory_context(current_message=query)
        if not block:
            return []
        return [
            Memory(
                text=block,
                score=1.0,
                metadata={"owner_type": "user", "source": str(self._store.memory_file)},
            )
        ]

    # ── MemoryBackend: the write side ──────────────────────────────────

    async def store(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Annotate this slice into ``episodes.md``, then refresh any hot section.

        Returns whether the episodes landed. False is a slice the outbox will
        offer again, so the annotation is the only thing that decides it: a
        section rewrite that fails leaves its tag offset where it was and is
        retried by the next turn's hot-tag scan, and reporting the whole slice
        unstored for it would annotate the same conversation twice.

        ``metadata`` is not read. A turn carries none, and there is no index here
        to partition by app or project.
        """
        if not messages:
            return True
        bound = resolve(None, self._binding)
        if not await self._annotate(messages, bound):
            return False
        await self._refresh_hot_tags(bound)
        return True

    async def feedback(self, signals: dict[str, Any]) -> None:
        """Nothing to consume: the profile has no per-skill confidence to move."""
        return None

    async def start(self) -> None:
        """Nothing to open. The files are written per call under the store's lock."""
        return None

    async def stop(self) -> None:
        """Nothing to close."""
        return None

    # ── Annotation ─────────────────────────────────────────────────────

    async def _annotate(self, messages: list[dict[str, Any]], bound: ModelBinding) -> bool:
        """Summarize the slice into tagged episode lines and append them.

        The prompt, the tool and the tag vocabulary are the released
        consolidator's. Foresight is asked for only when it is enabled, so with
        the flag off the model is not asked for predictions at all and the schema
        stays one slot wide.
        """
        transcript = _format_messages(messages)
        if not transcript.strip():
            return True
        enable_foresight = self._enable_foresight
        now_str = self._now_fn().strftime("%Y-%m-%d %H:%M (%A)")
        if enable_foresight:
            slot_lines = (
                "- episode_summary: ARRAY of single-line entries "
                '"[YYYY-MM-DD HH:MM] <summary, <=100 chars> #tag1 #tag2".\n'
                "- foresight_hint: ARRAY of predictions; [] when no deferred / "
                "recurring signal."
            )
            example_tail = (
                "\nforesight_hint:\n"
                '  - {{"prediction": "User will revisit WebSocket leak '
                'fix after load test next week", "window": "5-7 days", '
                '"confidence": "medium", "src_ts": '
                '"2024-11-08 14:20"}}\n'
                '  - {{"prediction": "User runs every Saturday morning '
                '(recurring habit, 3+ observations)", "window": '
                '"recurring weekly", "confidence": "high", '
                '"src_ts": "2024-11-09 10:00"}}\n'
                '  - {{"prediction": "Q4 retrospective scheduled for '
                'next Friday", "window": "5 days", "confidence": '
                '"high", "src_ts": "2024-11-09 14:00"}}\n'
                "(Again: FORMAT examples from an unrelated domain. "
                "Produce predictions only for the conversation above.)\n"
            )
            sys_line = (
                "You are a conversation annotator. Call "
                "annotate_conversation exactly once with both slots filled "
                "(foresight_hint may be [])."
            )
        else:
            slot_lines = (
                "- episode_summary: ARRAY of single-line entries "
                '"[YYYY-MM-DD HH:MM] <summary, <=100 chars> #tag1 #tag2".'
            )
            example_tail = ""
            sys_line = (
                "You are a conversation annotator. Call annotate_conversation exactly once with episode_summary filled."
            )
        # Feed the LLM its recently-used project slugs so it reuses
        # them instead of inventing new variants every call (e.g.
        # #project-clawtrack-release vs ...-cli vs ...-coverage).
        recent_tags = await asyncio.to_thread(self._store.recent_project_tags, days=14, limit=12)
        if recent_tags:
            tag_history_lines = "\n".join(f"  - #{tag} ({n}x in last 14 days)" for tag, n in recent_tags)
            tag_history_block = (
                "\n## Project tags you've recently used — REUSE these "
                "slugs when describing the same project; do NOT invent "
                "new variants:\n" + tag_history_lines + "\n"
            )
        else:
            tag_history_block = ""

        prompt = f"""Annotate this conversation chunk. Call annotate_conversation with:

{slot_lines}

## Critical rules

1. **Each episode summary must include specific identifiers** — file
   names, function names, PR numbers, percentages, durations. Avoid
   vague verbs like "worked on" / "discussed" / "planned"; describe the
   concrete artifact, decision, or finding.
2. **Reuse project slugs across calls**. If a project slug already
   exists in the "tags you've recently used" list below, use it
   verbatim. Splitting one project into multiple slugs
   (#project-clawtrack-release / -cli / -docs) destroys the tag-based
   refresh trigger — pick ONE stable slug per project.
3. **Tag the WORK, not the codebase**: `#project-<work-slug>` where the
   slug names the topic. Use `#project-auth-refactor` not
   `#project-backend-api`.
4. **Process tags can't stand alone**. `#question`, `#habit`, `#answer`
   describe HOW the user is talking, not WHAT about. Every episode
   needs at least one CONTENT tag (a `#project-*` or one of {{#perf,
   #bug, #decision, #blocker, #deferred, #pivot, #pr, #review, #rfc,
   #design, #infra, #sql, #ml}}) IN ADDITION to any process tag.
5. **Avoid the generic #task tag**.
{tag_history_block}
## Current Time
{now_str}

## Conversation to Annotate
{transcript}

## Output shape example
The examples below are from an UNRELATED domain (websocket / DB / feature-flag work). They demonstrate the FORMAT only. DO NOT copy any of their text or topics — generate entries that describe the actual conversation above.

episode_summary:
  - "[2024-11-08 14:20] Identified memory leak in WebSocketManager.broadcast(); ~200MB growth/hour under load #project-ws-stability #perf #bug"
  - "[2024-11-09 10:00] Migrated user_sessions from MyISAM to InnoDB (~12M rows, 4h offline window) #project-db-migration #infra #decision"
  - "[2024-11-11 16:30] Feature flag 'dark-mode-v2' ramped 10%->50% after 24h of steady metrics #project-feature-flag-rollout #pr #decision"
{example_tail}"""
        try:
            response = await bound.provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": sys_line},
                    {"role": "user", "content": prompt},
                ],
                tools=_build_annotate_tool(enable_foresight=enable_foresight),
                model=bound.model,
                tool_choice="required",
            )
        except Exception:
            logger.exception("annotate: the model call raised; this turn is not in episodes.md yet")
            return False
        if response.finish_reason == "error":
            # A call that did not happen. Worth offering again: the next attempt
            # runs against a service that may have come back.
            logger.warning("annotate: the model call failed -- {}", (response.content or "")[:200])
            return False
        # An answer that is not an annotation is acknowledged, not retried. The
        # model service has no wire spelling for ``tool_choice: required``, so the
        # call is asked for by the prompt and a model can answer in prose
        # instead -- and it will answer the same way to the same prompt. Three
        # attempts would buy three identical non-answers and then tell the user
        # their turn was lost to a service outage that never happened.
        if not response.has_tool_calls:
            logger.warning("annotate: the model answered without calling annotate_conversation; turn not indexed")
            return True
        args = _tool_args(response.tool_calls[0].arguments)
        if args is None:
            logger.warning("annotate: unusable tool arguments; turn not indexed")
            return True

        episodes = args.get("episode_summary") or []
        if isinstance(episodes, str):
            episodes = [episodes]
        lines = [_ensure_text(entry).strip() for entry in episodes]
        lines = [line for line in lines if line]
        try:
            written = await asyncio.to_thread(self._store.append_episodes, lines)
        except OSError:
            logger.exception("annotate: episodes.md could not be appended to")
            return False
        dropped = len(lines) - written
        logger.info(
            "annotate: {} message(s) -> {} episode(s){}",
            len(messages),
            written,
            f", {dropped} dropped as process-only" if dropped else "",
        )

        if enable_foresight:
            await self._append_foresight(args.get("foresight_hint") or [])
        return True

    async def _append_foresight(self, foresights: list[Any]) -> None:
        """Persist predictions to ``## Foresight``, off the event loop.

        Best-effort and reported, not returned: the episodes are already on disk,
        and a failed prediction is not a reason to annotate the conversation
        again.
        """
        entries = [entry for entry in foresights if isinstance(entry, dict)]
        if not entries:
            return
        try:
            written = await asyncio.to_thread(self._store.append_foresight, entries)
        except OSError:
            logger.exception("foresight: user.md could not be written")
            return
        logger.info(
            "foresight: {} prediction(s) emitted -> {} written, {} already known",
            len(entries),
            written,
            len(entries) - written,
        )

    # ── The profile, one hot section at a time ─────────────────────────

    async def _refresh_hot_tags(self, bound: ModelBinding) -> int:
        """Rewrite the section behind every tag that has heated up. Hottest first.

        Serial on purpose: two sections rewritten at once would each compute
        against a profile the other is about to change, and one of them would
        lose its compare-and-set. A tag whose rewrite did not land keeps its
        offset, so the next turn tries it again.
        """
        hot = await asyncio.to_thread(self._store.hot_tags, REFRESH_HOT_TAG_THRESHOLD)
        refreshed = 0
        for tag, current_count, _previous in hot:
            if not await self._refresh_section(tag, bound):
                logger.warning("profile refresh: #{} was not rewritten; its offset stays where it was", tag)
                continue
            await asyncio.to_thread(self._store.commit_refresh_offset, tag, current_count)
            refreshed += 1
        return refreshed

    async def _refresh_section(self, tag: str, bound: ModelBinding) -> bool:
        """Rewrite the one H2 section ``tag``'s recent episodes belong to.

        The model picks the target heading (or invents one when none fits) and
        returns that section's whole new body. The splice replaces that section
        and nothing else. A bullet that cites no episode is dropped before the
        write -- the prompt requires the citation, and an uncited bullet is the
        shape a speculation takes.
        """
        relevant = await asyncio.to_thread(self._store.episodes_for_tag, tag, REFRESH_MAX_EPISODES)
        if not relevant:
            logger.debug("profile refresh: #{} has no episodes", tag)
            return True
        current_profile = await asyncio.to_thread(self._store.read_long_term)
        now_str = self._now_fn().strftime("%Y-%m-%d %H:%M (%A)")
        episodes_block = "\n".join(relevant)
        prompt = f"""Update ONE H2 section of user.md based on the recent
#{tag} episodes below. user.md is a PROFILE SNAPSHOT (current state per
topic), NOT an event log — episodes.md already keeps the event log.

<principles>
1. Explicit Evidence Required — only place a fact in user.md if you can
   cite an episode timestamp. No speculation, no inference from titles.
2. Quality Over Quantity — 5 accurate bullets > 15 noisy ones.
   An empty section is OK.
3. Inertia — existing bullets are correct unless a NEW episode
   contradicts them. UPDATE bullets in place rather than rewrite.
4. Reject Events — one-off events, emotional states ("anxiety",
   "frustration"), and transient process work ("in middle of debugging X")
   do NOT belong in user.md — they're already in episodes.md.
5. Profile Snapshot, Not Diary — every bullet answers "what is true
   about this user right now?", not "what happened on day X?".
6. Abstraction, Not Enumeration — a profile bullet captures a PATTERN,
   not a list of instances. When N episodes share a theme, write ONE
   bullet describing the abstraction; do NOT comma-list the instances
   inside the bullet.
   ✓ "Spends commute/break time on child-related research"
   ✗ "Researches English materials, breakfast recipes, sunscreen,
      dental care, vaccines, parent-child games, homework, time
      management"
   The 9-item enumeration above defeats the snapshot — each instance
   already lives in episodes.md; user.md only needs the theme.
</principles>

<section_schemas>
Each H2 section follows a semi-structured convention. Use **Field**:
prefix for required slots; bullets without prefix are ad-hoc additions.

## Identity (≤ 5 bullets — stable role/personal facts)
  - **Name**: ...
  - **Role**: ...
  - **Stack**: ...
  - **Location**: ...
  - **Key relations**: <name + role, e.g. "周晓棠 (girlfriend)">

## Preferences (≤ 5 bullets — working style / tools / quiet hours)
  - **Communication**: terse | verbose | mixed; emoji-friendly Y/N
  - **Tools**: <comma-separated preferences>
  - **Quiet hours**: <when not to interrupt>
  - ad-hoc preference bullets allowed

## Projects → ### <project-name> (each H3 has 4-6 bullets)
  - **Type**: work | side project | learning | personal
  - **Status**: <one-line current state>
  - **Recent work**: <2-3 descriptive items, NOT per-day events>
  - **Next**: <upcoming actions>
  - Optional: **Stack**, **Stakeholders**, **Deadline**

## Habits (≤ 6 bullets — recurring patterns, ≥ 2 observations to qualify)
  - **<pattern>** (confirmed by N obs; freq: weekly | daily | sporadic)
    Example: "Saturday morning run (confirmed by 4+ obs; freq: weekly)"

## Notes (≤ 8 bullets — important specific facts)
  - <birthday / deadline / preferred X / similar facts>

## Foresight — AUTO-MANAGED by a different path. DO NOT TARGET; do NOT
rewrite its contents.
</section_schemas>

<triage_each_episode>
Before deciding what to write, classify each new episode:

KEEP (write into user.md) if it represents:
  - identity / role / relationship fact → ## Identity
  - working preference confirmed → ## Preferences
  - project state change (status, deliverable, decision) → ## Projects
  - recurring pattern with ≥ 2 observations → ## Habits
  - dated commitment / deadline / specific fact → ## Notes

REJECT (stays in episodes.md only, do NOT add to user.md) if it's:
  - one-off event ("ran today", "had lunch", "PR merged" — the PR-merged
    detail goes in episodes.md; the project's Status field captures the
    end state, not the per-event)
  - emotional state ("anxious", "frustrated", "excited")
  - transient process ("in middle of debugging X", "testing Y")
  - in-progress detail that resolves soon
  - already covered by an existing bullet without new info
  - just a question the user asked
</triage_each_episode>

<update_protocol>
For each episode that PASSES triage, follow this order STRICTLY:

1. UPDATE first — find an existing bullet on the same subject; refine it
   in place to reflect the latest evidence.
   Example: existing "**Status**: pre-release testing"
            + new "PR #1287 merged: v1.0 released"
            → "**Status**: v1.0 released (5/15), gathering feedback"

2. CONSOLIDATE second — merge related bullets in the same section.
   Example: "**Recent work**: CLI bug" + new "doc generation broken"
            → "**Recent work**: CLI bug + doc generation broken"

   ANTI-PATTERN — DO NOT enumerate. When N episodes share a THEME but
   each adds a different specific instance, do NOT comma-list every
   instance inside the bullet.
   Existing "Researches child topics" + new episode "researched vaccines":
     ✗ bad:  "Researches child topics including English, breakfast,
              vaccines, dental care, sunscreen, ..."
     ✓ good: leave the bullet UNCHANGED — the theme is already captured;
             the specific vaccine instance lives in episodes.md.
   Apply this whenever you find yourself reaching for "including", "such
   as", "e.g.", or a comma-list of nouns inside one bullet.

3. REMOVE third — drop bullets obsoleted by new evidence.
   Example: "**Status**: pre-release anxiety" → DROP once "released" lands.

4. APPEND last — only if truly new topic AND under the section cap.

After processing, respect section caps (see <section_schemas>).
If you'd exceed a cap, CONSOLIDATE harder.
</update_protocol>

## Current Time
{now_str}

## Current user.md (UPDATE/CONSOLIDATE/REJECT — don't just append)
{current_profile or "(empty)"}

## Recent episodes tagged #{tag} ({len(relevant)} entries — fold into
the matching section after triage)
{episodes_block}

## Output
section_heading: the H2 line you're updating (verbatim; for project
   work use `## Projects` — the H3 sub-section goes inside section_body).
section_body: full new content for that H2, every bullet ending with
   `[src: episodes.md @ <ts>]`. Use the LATEST relevant ts when merging.
"""
        try:
            response = await bound.provider.chat_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You maintain a structured user profile in user.md, "
                            "NOT an event log. Follow the <principles>, "
                            "<section_schemas>, <triage_each_episode>, and "
                            "<update_protocol> blocks in the user message. "
                            "Prefer UPDATE over APPEND; respect per-section "
                            "size caps; reject events / emotions / transient "
                            "process work that already lives in episodes.md."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=_REFRESH_SECTION_TOOL,
                model=bound.model,
                tool_choice="required",
            )
        except Exception:
            logger.exception("profile refresh: the model call for #{} raised", tag)
            return False
        if response.finish_reason == "error":
            logger.warning("profile refresh: the model call for #{} failed", tag)
            return False
        if not response.has_tool_calls:
            logger.warning("profile refresh: the model did not call refresh_profile_section for #{}", tag)
            return False
        args = _tool_args(response.tool_calls[0].arguments)
        if args is None or "section_heading" not in args or "section_body" not in args:
            logger.warning("profile refresh: unexpected tool arguments for #{}", tag)
            return False
        heading = _ensure_text(args["section_heading"]).strip()
        body = _ensure_text(args["section_body"])
        if not heading.startswith("## "):
            logger.warning("profile refresh: #{} named {!r}, which is not an H2 heading", tag, heading)
            return False

        # ``## Foresight`` is written by one path only, and its bullets carry the
        # paren-style source the bracket pattern does not match -- filtering it
        # here would empty the section.
        if heading != FORESIGHT_HEADING:
            body, uncited = drop_bullets_without_src(body)
            if uncited:
                logger.warning("profile refresh: dropped {} uncited bullet(s) from {!r}", uncited, heading)

        bullets = sum(1 for line in body.splitlines() if line.lstrip().startswith("-"))
        if bullets > SECTION_BULLET_DRIFT:
            logger.warning(
                "profile refresh: {!r} came back with {} bullets; the profile is drifting into a diary",
                heading,
                bullets,
            )

        landed = await asyncio.to_thread(self._store.splice_section, heading, body, current_profile)
        if landed:
            logger.info(
                "profile refresh: {!r} rewritten from {} #{} episode(s) -> {} bullets",
                heading,
                len(relevant),
                tag,
                bullets,
            )
        return landed


__all__ = ["REFRESH_HOT_TAG_THRESHOLD", "HostMarkdownBackend"]
