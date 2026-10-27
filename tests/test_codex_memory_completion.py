import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from letta.agents.letta_agent_v3 import LettaAgentV3
from letta.helpers import ToolRulesSolver
from letta.llm_api.codex_cli_client import CodexCLIClient
from letta.schemas.letta_message_content import ReasoningContent, TextContent
from letta.schemas.letta_stop_reason import StopReasonType
from letta.schemas.openai.chat_completion_response import FunctionCall, ToolCall, UsageStatistics
from letta.schemas.tool_execution_result import ToolExecutionResult
from letta.schemas.tool_rule import ChildToolRule, ContinueToolRule, RequiredBeforeExitToolRule


async def handle_response(
    tool_names,
    *,
    text="The fix is complete and verified.",
    provider="codex_cli",
    statuses=None,
    rules=None,
    content=None,
    is_final_step=False,
    valid_tool_names=None,
):
    agent = object.__new__(LettaAgentV3)
    available_names = set(tool_names) | {"follow_up"}
    agent.agent_state = SimpleNamespace(
        id="agent-00000000-0000-4000-8000-000000000001",
        llm_config=SimpleNamespace(model="test-model", model_endpoint_type=provider),
        timezone="UTC",
        tools=[SimpleNamespace(name=name, return_char_limit=1000, enable_parallel_execution=True) for name in available_names],
    )
    agent.logger = Mock()
    agent.client_tools = []
    agent._execute_tool = AsyncMock(
        side_effect=[ToolExecutionResult(status=status) for status in (statuses or ["success"] * len(tool_names))]
    )
    solver = ToolRulesSolver(tool_rules=rules or [])
    messages, should_continue, stop_reason = await agent._handle_ai_response(
        valid_tool_names=list(available_names) if valid_tool_names is None else valid_tool_names,
        tool_rules_solver=solver,
        usage=UsageStatistics(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        content=[TextContent(text=text)] if content is None else content,
        tool_calls=[
            ToolCall(id=f"call_{index}", type="function", function=FunctionCall(name=name, arguments="{}"))
            for index, name in enumerate(tool_names)
        ],
        is_final_step=is_final_step,
        finish_reason="tool_calls",
    )
    return agent, messages, should_continue, stop_reason


@pytest.mark.parametrize(
    "tool_names",
    [
        ["archival_memory_insert"],
        ["core_memory_append"],
        ["core_memory_replace"],
        ["archival_memory_delete"],
        ["archival_memory_insert", "archival_memory_insert"],
        ["core_memory_append", "archival_memory_insert"],
    ],
)
def test_completed_memory_writes_end_turn_and_preserve_messages(tool_names):
    agent, messages, should_continue, stop_reason = asyncio.run(handle_response(tool_names))

    assert should_continue is False
    assert stop_reason.stop_reason == StopReasonType.end_turn.value
    assert agent._execute_tool.await_count == len(tool_names)
    assert len(messages) == 2
    assert messages[0].content[0].text == "The fix is complete and verified."
    assert len(messages[1].tool_returns) == len(tool_names)
    assert all(result.status == "success" for result in messages[1].tool_returns)


@pytest.mark.parametrize(
    "options",
    [
        {"tool_names": ["archival_memory_search"]},
        {"tool_names": ["archival_memory_insert", "archival_memory_search"]},
        {"tool_names": ["follow_up"]},
        {"tool_names": ["archival_memory_insert"], "text": "  "},
        {"tool_names": ["archival_memory_insert"], "content": []},
        {"tool_names": ["archival_memory_insert"], "content": [ReasoningContent(reasoning="Remember this", is_native=True)]},
        {"tool_names": ["archival_memory_insert"], "provider": "openai"},
        {"tool_names": ["archival_memory_insert"], "statuses": ["error"]},
        {"tool_names": ["archival_memory_insert", "core_memory_replace"], "statuses": ["success", "error"]},
        {"tool_names": ["archival_memory_insert"], "rules": [RequiredBeforeExitToolRule(tool_name="follow_up")]},
        {"tool_names": ["archival_memory_insert"], "rules": [ContinueToolRule(tool_name="archival_memory_insert")]},
        {
            "tool_names": ["archival_memory_insert"],
            "rules": [ChildToolRule(tool_name="archival_memory_insert", children=["follow_up"])],
        },
        {"tool_names": ["archival_memory_insert"], "valid_tool_names": []},
    ],
)
def test_incomplete_or_non_codex_work_keeps_continuation(options):
    _, _, should_continue, _ = asyncio.run(handle_response(**options))

    assert should_continue


def test_max_steps_stop_reason_is_preserved():
    _, _, should_continue, stop_reason = asyncio.run(handle_response(["archival_memory_insert"], is_final_step=True))

    assert not should_continue
    assert stop_reason.stop_reason == StopReasonType.max_steps.value


def test_final_answer_with_two_archival_tags_does_not_request_another_model_step():
    clean_text, tool_calls = CodexCLIClient._extract_memory_ops(
        "The fix is complete.\n[세션요약]반복 실행 수정[/세션요약]\n"
        '<letta_archival_insert tags="work">Fix and verification results.</letta_archival_insert>\n'
        '<letta_archival_insert tags="recovery">Failure and successful alternative.</letta_archival_insert>'
    )
    _, messages, should_continue, stop_reason = asyncio.run(handle_response([tool.function.name for tool in tool_calls], text=clean_text))

    assert should_continue is False
    assert stop_reason.stop_reason == StopReasonType.end_turn.value
    assert messages[0].content[0].text == clean_text
    assert len(messages[1].tool_returns) == 2
    assert all(json.loads(part.text)["status"] == "OK" for part in messages[1].content)
