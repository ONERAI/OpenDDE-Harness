"""
OpenDDE Harness — antibody design harness: an agent runtime, a protein-design
plugin, and the compute service they drive.

Three feature pillars:
    1. Context Management   — context_engine/          (deterministic selection)
    2. Token Efficiency     — token_wise/
    3. Skill Self-Evolution — memory_engine/skill_forge/

The base agent runtime (agent/, cli/, config/, providers/, session/,
templates/, utils/) originated from the MIT-licensed nanobot project by
HKUDS. Feature pillars listed above are new to OpenDDE Harness.
See LICENSE for attribution.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("opendde-harness")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
__logo__ = "ϒ"
