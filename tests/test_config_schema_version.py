"""The shape of ``config.json``, and the number that tracks it.

``ddeharness onboard`` reuses a configuration whose shape this build
understands and moves aside one it does not, so the number has to move when
the shape does. Nobody remembers to bump a constant, so this fails when the
shape moves and the constant has not, and says what to do about it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from opendde_harness.config.schema import CONFIG_SCHEMA_VERSION, Config

#: The shape at :data:`CONFIG_SCHEMA_VERSION`. Both move together or neither
#: does.
SHAPE = "e484b583c676eef4"


def shape_of(model: type[BaseModel], seen: set[str] | None = None) -> Any:
    """Field names and the types under them, as something comparable.

    Names and nesting are what a written file carries, so they are what is
    hashed; a default that changes, a docstring, an order are not the file's
    shape and do not move the number.
    """
    seen = seen if seen is not None else set()
    if model.__name__ in seen:
        return model.__name__
    seen = seen | {model.__name__}
    fields = {}
    for name, field in model.model_fields.items():
        annotation = field.annotation
        nested = [
            argument
            for argument in (getattr(annotation, "__args__", ()) or (annotation,))
            if isinstance(argument, type) and issubclass(argument, BaseModel)
        ]
        fields[field.alias or name] = [shape_of(argument, seen) for argument in nested] if nested else str(annotation)
    return fields


def test_the_shape_and_the_number_move_together():
    digest = hashlib.sha256(json.dumps(shape_of(Config), sort_keys=True, default=str).encode()).hexdigest()[:16]

    assert digest == SHAPE, (
        "config.json's shape changed. Raise CONFIG_SCHEMA_VERSION by one "
        f"(now {CONFIG_SCHEMA_VERSION}) so `ddeharness onboard` starts the next run clean, "
        f"and set SHAPE in this test to {digest!r}."
    )


def test_a_written_config_carries_the_number():
    """So a later build can tell whether the file is one it understands."""
    written = Config().model_dump(by_alias=True)

    assert written["schemaVersion"] == CONFIG_SCHEMA_VERSION
