"""LLM provider abstraction module."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opendde_harness.providers.base import LLMProvider, LLMResponse

__all__ = ["LLMProvider", "LLMResponse"]

# Lazy re-exports (PEP 562). The reason they are lazy has changed: the import
# that dominated CLI cold start was litellm, and it is gone. What remains is an
# ordinary courtesy -- importing one provider submodule should not pull the
# others in with it.
_LAZY_EXPORTS = {
    "LLMProvider": "opendde_harness.providers.base",
    "LLMResponse": "opendde_harness.providers.base",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
