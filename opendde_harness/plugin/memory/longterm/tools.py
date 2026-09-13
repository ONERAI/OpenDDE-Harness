"""``understand_media`` — the memory library's multimodal understanding, as an agent tool.

Contributed via the manifest's ``[[plugin.contributes.tools]]`` slot and
registered into the agent's tool set at boot. The LLM calls it on demand to
read the contents of an attachment it can't natively consume — a PDF, an
audio clip, an Office doc, a scanned image — instead of OpenDDE Harness parsing
every attachment up front. The attachment paths are surfaced to the model in
the user message (see ``render.build_user_content``); the model passes them
back here.

The tool is deliberately thin: all parsing lives in
:func:`opendde_harness.plugin.memory.longterm.multimodal.understand_files`, which reuses the exact
parser the memory library runs during memory ingest.
"""

from __future__ import annotations

import logging
from typing import Any

from opendde_harness.agent.tools.base import Tool
from opendde_harness.plugin.memory.longterm._library import MULTIMODAL_EXTRA, multimodal_parser_installed
from opendde_harness.plugin.memory.longterm.multimodal import MultimodalUnavailableError, understand_files

logger = logging.getLogger("opendde_harness.plugin.memory.longterm")


class UnderstandMediaTool(Tool):
    """Read/understand attached files via the memory library's multimodal parser."""

    @property
    def name(self) -> str:
        return "understand_media"

    @property
    def description(self) -> str:
        return (
            "Extract the contents of attachments you cannot read directly, as "
            "text: PDFs, audio (transcription), Office documents "
            "(docx/xlsx/pptx), and http(s) URLs (fetched and parsed by "
            "content type). Pass the file path(s) shown in the "
            "'[Attachment: ...]' or '[Image: ...]' notes of the user message, "
            "and/or http(s) URLs. For an image prefer the read tool, which hands you the picture "
            "itself; reach for this tool on an image only when you cannot see "
            "images or when a scan needs OCR — it returns another model's "
            "transcription, not the original. Video is not supported."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "File path(s) to understand, exactly as shown in the "
                        "'[Attachment: <name> (path: <path>)]' or "
                        "'[Image: <name> (path: <path>)]' notes, and/or "
                        "http(s) URL(s) to fetch and read."
                    ),
                },
            },
            "required": ["paths"],
        }

    async def execute(self, paths: Any = None, **_: Any) -> str:
        if isinstance(paths, str):
            paths = [paths]
        if not paths or not isinstance(paths, list):
            return "Error: 'paths' is required — a list of attachment file paths."

        try:
            results = await understand_files([str(p) for p in paths])
        except MultimodalUnavailableError as e:
            return f"Error: multimodal understanding is unavailable. {e}"

        parts: list[str] = []
        for r in results:
            text = r.get("text")
            if text:
                parts.append(f"## {r['name']}\n{text}")
            else:
                parts.append(f"## {r['name']}\n[could not understand: {r.get('error')}]")
        return "\n\n".join(parts) if parts else "No files were provided."


def _multimodal_available() -> bool:
    """True if the optional multimodal parser extra is usable.

    Isolated as a seam so the registration gate is unit-testable without
    the heavy extra installed. Uses the library's own availability check
    (raises when the parser extra is absent).
    """
    try:
        multimodal_parser_installed()
    except Exception as e:
        logger.info(
            "understand_media not registered: multimodal parser extra "
            "unavailable (%s). Install with `pip install '%s'`.",
            e,
            MULTIMODAL_EXTRA,
        )
        return False
    return True


def make_understand_media_tool(ctx: Any) -> Tool | None:
    """Plugin tool-factory entry point (manifest ``contributes.tools``).

    Stateless -- the parser resolves its model at call time (the conversation's
    default, through the model service), so ``ctx`` is accepted for signature
    symmetry but not used.

    Returns ``None`` when the optional multimodal parser extra isn't
    installed, so the host declines to register a tool that could never
    succeed. The LLM then never sees ``understand_media`` in environments
    without the extra, rather than discovering it's unavailable only after
    spending a tool call on it. Whether the conversation model can read the
    attachment is a call-time answer, reported by the tool.
    """
    del ctx
    # Point the library at the memory home before any library import resolves
    # settings (the parser reads them at call time), and at this plugin's own
    # model client: the parser runs in this process, and the model it parses
    # with is the conversation's, through the model service -- the same client
    # the memory server is started with.
    from opendde_harness.plugin.memory.longterm._library import patch_llm_client
    from opendde_harness.plugin.memory.longterm._service_llm import ModelServiceLLMClient
    from opendde_harness.plugin.memory.longterm.settings import configure_memory_env, ensure_memory_home, memory_root

    root = memory_root()
    configure_memory_env(root)
    ensure_memory_home(root)
    patch_llm_client(ModelServiceLLMClient)
    if not _multimodal_available():
        return None
    return UnderstandMediaTool()


__all__ = ["UnderstandMediaTool", "make_understand_media_tool"]
