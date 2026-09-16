from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any, Protocol

from videoshop.agents.adapters import ToolCallParseError, agent_step_from_tool_calls
from videoshop.agents.prompt import build_system_prompt, few_shot_messages, observation_to_user_message
from videoshop.agents.tool_specs import FINAL_ACTION_TOOL, video_shop_tool_specs
from videoshop.simulator.schemas import AgentStep, EnvState, FinalAction, Observation, ToolCallRequest, ToolCallResult
from videoshop.simulator.tool_runtime import ToolRuntime


class ToolCallingClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Sequence[Any]:
        """Return provider-native tool calls for the next assistant turn."""


class FunctionCallingAgent:
    """Provider-agnostic tool-calling loop for VideoShopEnv.

    The client only needs to return tool/function calls. The loop executes non-final
    tools through ToolRuntime, appends their results to messages, and stops when the
    model calls submit_final_action.
    """

    def __init__(
        self,
        client: ToolCallingClient | Callable[[list[dict[str, Any]], list[dict[str, Any]]], Sequence[Any]],
        *,
        max_tool_rounds: int = 4,
        include_few_shots: bool = True,
    ) -> None:
        self.client = client
        self.max_tool_rounds = max_tool_rounds
        self.include_few_shots = include_few_shots
        self.tools = video_shop_tool_specs()
        self.runtime = ToolRuntime()

    def act(self, observation: Observation, state: EnvState) -> AgentStep:
        messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt()}]
        if self.include_few_shots:
            messages.extend(few_shot_messages())
        messages.append(observation_to_user_message(observation))

        collected_requests: list[ToolCallRequest] = []

        repair_warnings: list[str] = []
        for round_index in range(self.max_tool_rounds):
            raw_calls = list(self._complete(messages))
            try:
                step = agent_step_from_tool_calls(raw_calls)
            except ToolCallParseError as exc:
                return _fallback_step(collected_requests, "tool_call_parse_error", str(exc))

            collected_requests.extend(step.tool_requests)

            if step.final_action is not None:
                metadata = {"agent_repair_warnings": repair_warnings} if repair_warnings else {}
                return AgentStep(tool_requests=collected_requests, final_action=step.final_action, metadata=metadata)

            if not step.tool_requests:
                repair_warnings.append("missing_final_action")
                if round_index < self.max_tool_rounds - 1:
                    messages.append(_final_action_repair_message())
                    continue
                break

            results = self.runtime.execute_many(state, step.tool_requests)
            messages.append({"role": "assistant", "tool_calls": _tool_calls_for_messages(raw_calls)})
            messages.extend(_tool_result_messages(results, raw_calls))

        return _fallback_step(collected_requests, "missing_final_action", "Model did not call submit_final_action.")

    def _complete(self, messages: list[dict[str, Any]]) -> Sequence[Any]:
        if hasattr(self.client, "complete"):
            return self.client.complete(messages, self.tools)  # type: ignore[union-attr]
        return self.client(messages, self.tools)  # type: ignore[misc]


def _fallback_step(tool_requests: list[ToolCallRequest], warning: str, detail: str) -> AgentStep:
    return AgentStep(
        tool_requests=tool_requests,
        final_action=FinalAction(
            action_type="delay_recommendation",
            product_id=None,
            reason=f"Fallback delay because {warning}: {detail}",
            reasoning_summary={
                "observation_facts": [],
                "evidence_used": [],
                "candidate_comparison": [],
                "rejected_options": ["model output could not be safely converted into an environment action"],
                "decision_rule": f"Use delay_recommendation fallback when {warning} occurs.",
                "confidence": 1.0,
            },
        ),
        metadata={"agent_warnings": [warning], "warning_detail": detail},
    )


def _final_action_repair_message() -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Your previous response did not call any tool or submit a final action. "
            "Now call submit_final_action exactly once. Use valid JSON arguments. "
            "If no safe recommendation exists, use action_type=delay_recommendation and product_id=null. "
            "Include a concise reasoning_summary object."
        ),
    }


def _tool_calls_for_messages(raw_calls: Sequence[Any]) -> list[Any]:
    return list(raw_calls)


def _tool_result_messages(results: list[ToolCallResult], raw_calls: Sequence[Any]) -> list[dict[str, str]]:
    messages = []
    call_ids = [_tool_call_id(call) for call in raw_calls]
    for index, result in enumerate(results):
        message = {
            "role": "tool",
            "name": result.tool,
            "content": json.dumps(asdict(result), ensure_ascii=False),
        }
        if index < len(call_ids) and call_ids[index]:
            message["tool_call_id"] = call_ids[index]
        messages.append(message)
    return messages


def _tool_call_id(raw_call: Any) -> str | None:
    if isinstance(raw_call, dict):
        return raw_call.get("id")
    if hasattr(raw_call, "id"):
        return getattr(raw_call, "id")
    if hasattr(raw_call, "model_dump"):
        return raw_call.model_dump().get("id")
    return None


class ScriptedToolCallingClient:
    """Tiny test/demo client that returns one scripted batch of tool calls per turn."""

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self.turns = list(turns)
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Sequence[Any]:
        del messages, tools
        if self.calls >= len(self.turns):
            return [{"name": FINAL_ACTION_TOOL, "arguments": {"action_type": "delay_recommendation", "product_id": None, "reason": "No scripted action remains."}}]
        turn = self.turns[self.calls]
        self.calls += 1
        return turn
