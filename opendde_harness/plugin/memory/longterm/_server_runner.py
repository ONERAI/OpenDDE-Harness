"""Start the long-term memory server with this project's model layer as its LLM.

The library's server is started in this process after its LLM seam is pointed
at :class:`~opendde_harness.plugin.memory.longterm._service_llm.ModelServiceLLMClient`,
so every model call the server makes -- extraction, consolidation, the
retrieval decider, multimodal parsing -- goes to the conversation's default
model through the model service. Started by ``_server.py`` as
``python -m opendde_harness.plugin.memory.longterm._server_runner <server
command line> --root <root> --config <config.json>``; the config path is what
the client resolves the default model from, and it is passed rather than
inherited because a ``--config`` given to the parent lives in its memory, not
in its environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

from opendde_harness.plugin.memory.longterm._library import patch_llm_client, start_server


def _option(args: list[str], name: str) -> Path:
    try:
        return Path(args[args.index(name) + 1]).expanduser().resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"{name} PATH is required") from exc


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    root = _option(args, "--root")
    config_path = _option(args, "--config")

    from opendde_harness.config.loader import set_config_path

    from ._service_llm import ModelServiceLLMClient

    set_config_path(config_path)
    patch_llm_client(ModelServiceLLMClient)
    start_server(root)


if __name__ == "__main__":
    main()
