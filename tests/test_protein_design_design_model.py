"""Which model a detached design task runs on, and where that choice is made."""

from __future__ import annotations

import json

import pytest

from opendde_harness.plugin.protein_design.core.contracts import WorkflowConfig
from opendde_harness.plugin.protein_design.servers.worker import design_model
from opendde_harness.plugin.protein_design.tools.control import StartTool
from opendde_harness.providers.binding import ModelBinding, use_binding


def _workflow(**overrides) -> WorkflowConfig:
    base = WorkflowConfig.model_construct(llm_model=None)
    return base.model_copy(update=overrides)


def test_the_yaml_model_wins_and_the_default_is_the_fallback():
    assert (
        design_model(_workflow(llm_model="openai-codex/gpt-5.6-luna"), "yunjintao/qwen3.8-flash")
        == "openai-codex/gpt-5.6-luna"
    )
    assert design_model(_workflow(), "yunjintao/qwen3.8-flash") == "yunjintao/qwen3.8-flash"


class _Runtime:
    def __init__(self, workflow):
        self.workflow = workflow

    def config_from_path(self, _path):
        return self.workflow


class _Snapshot:
    task_id = "t1"

    def model_dump(self, mode="json"):
        return {"task_id": self.task_id}


@pytest.mark.asyncio
async def test_a_task_started_from_a_conversation_designs_on_that_conversations_model():
    launched = {}

    async def launcher(workflow):
        launched["model"] = workflow.llm_model
        return _Snapshot()

    tool = StartTool(_Runtime(_workflow()), launcher=launcher)
    with use_binding(ModelBinding(provider=object(), model="openai-codex/gpt-5.6-luna"), window=None):
        result = await tool.execute(config_path="x.yaml", user_confirmed=True)
    assert launched["model"] == "openai-codex/gpt-5.6-luna"
    assert json.loads(result.model_text)["design_model"] == "openai-codex/gpt-5.6-luna"


@pytest.mark.asyncio
async def test_a_yaml_that_names_a_model_keeps_it_whatever_the_conversation_runs_on():
    launched = {}

    async def launcher(workflow):
        launched["model"] = workflow.llm_model
        return _Snapshot()

    tool = StartTool(_Runtime(_workflow(llm_model="deepseek/deepseek-v4-flash")), launcher=launcher)
    with use_binding(ModelBinding(provider=object(), model="openai-codex/gpt-5.6-luna"), window=None):
        await tool.execute(config_path="x.yaml", user_confirmed=True)
    assert launched["model"] == "deepseek/deepseek-v4-flash"
