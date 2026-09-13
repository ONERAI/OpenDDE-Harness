"""What a provider can be offered as serving, when the model service cannot say.

The service is the catalogue: it answers ``models`` with one row per model the
configured set serves, and the picker and the wizard both read that so the two
cannot disagree. :mod:`common_models` is the curated shortlist behind it, read
only where the question cannot be put to the service at all -- which is the
case this file is about, because the alternative there is an empty list.

What replaced ``providers/pin.py``: nothing. A model id's prefix names its
provider, so there is no rule left that has to decide whether a pinned provider
serves a bare id -- these are the shortlist's own tests, which that suite also
carried.
"""

from opendde_harness.providers import common_models, model_id
from tests._config import config as build_config
from tests._config import keyed


def test_every_shortlist_is_keyed_by_a_pi_id_and_qualified_with_it():
    """A shortlist entry is stored exactly as written, so it has to name its own
    provider: a bare id would be stored bare and route nowhere."""
    from opendde_harness.providers import pi_ids

    for provider, models in common_models.COMMON_MODELS.items():
        assert pi_ids.is_builtin(provider), provider
        for model in models:
            assert model_id.provider_of(model) == provider, model


def test_the_shortlist_answers_inside_a_running_loop_without_asking_the_service(monkeypatch):
    """``models_for_provider`` runs its own event loop, so a caller already on
    one -- the RPC server, the wizard's async paths -- would get an exception
    rather than a list. It falls back to the shortlist instead, and asks the
    service nothing."""
    asked: list[str] = []

    async def rows(_config):
        asked.append("models")
        return [{"id": "deepseek-chat", "name": "DeepSeek Chat", "provider": "deepseek"}]

    monkeypatch.setattr(common_models, "service_rows", rows)

    async def inside_a_loop():
        return common_models.models_for_provider(build_config(keyed("deepseek")), "deepseek")

    import asyncio

    offered = asyncio.run(inside_a_loop())

    assert asked == []
    # Bare, as the endpoint serves them: the shortlist stores qualified ids and
    # the service's own rows are bare, so the fallback answers in the same shape.
    assert [row["id"] for row in offered] == [
        model_id.bare(model) for model in common_models.common_models_for("deepseek")
    ]


def test_a_service_that_cannot_answer_falls_back_to_the_shortlist(monkeypatch):
    """No Node, no built bundle, a configuration it refused: a curated shortlist
    is better than an empty picker, and it is the only offline answer there is."""

    async def refuse(_config):
        raise RuntimeError("no model service here")

    monkeypatch.setattr(common_models, "service_rows", refuse)

    offered = common_models.models_for_provider(build_config(keyed("deepseek")), "deepseek")

    assert [row["id"] for row in offered] == [
        model_id.bare(model) for model in common_models.common_models_for("deepseek")
    ]


def test_the_services_rows_are_grouped_by_the_provider_that_serves_them():
    """One request answers for every provider at once, so the grouping is here.
    Matched on the pi id outright -- the config and the rows use the same ids."""
    rows = [
        {"id": "deepseek-chat", "provider": "deepseek"},
        {"id": "glm-4.6", "provider": "zai"},
    ]

    assert common_models.rows_for_provider(rows, "deepseek") == [rows[0]]
    assert common_models.rows_for_provider(rows, "nobody") == []
