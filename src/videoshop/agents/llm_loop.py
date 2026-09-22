from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any, Protocol

from videoshop.agents.adapters import ToolCallParseError, agent_step_from_tool_calls, normalize_tool_call
from videoshop.agents.prompt import build_system_prompt, few_shot_messages, observation_to_user_message
from videoshop.agents.tool_specs import FINAL_ACTION_TOOL, video_shop_tool_specs
from videoshop.agents.validation import final_action_errors, request_state_errors
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
        system_prompt: str | None = None,
        few_shots: list[dict[str, str]] | None = None,
    ) -> None:
        self.client = client
        self.max_tool_rounds = max_tool_rounds
        self.include_few_shots = include_few_shots
        # The composite environment needs a different brief: the clip and anchor product
        # are already fixed, and coupon availability is only knowable through get_coupon.
        # `FeedInterventionEnv` / `build_intervention_agent` swap these for the
        # intervention versions; the defaults here stay the frozen v1 protocol.
        self.system_prompt = system_prompt or build_system_prompt()
        self.few_shots = few_shots
        self.tools = video_shop_tool_specs()
        self.runtime = ToolRuntime()

    def act(self, observation: Any, state: EnvState) -> AgentStep:
        """`observation` is a v1 `Observation` or a feed `InterventionObservation`.

        Both are dataclasses that `observation_to_user_message` serialises the same way.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if self.include_few_shots:
            messages.extend(self.few_shots if self.few_shots is not None else few_shot_messages())
        messages.append(observation_to_user_message(observation))

        collected_requests: list[ToolCallRequest] = []
        collected_results: list[ToolCallResult] = []
        tool_call_trace: list[dict[str, Any]] = []

        repair_warnings: list[str] = []
        for round_index in range(self.max_tool_rounds):
            raw_calls = list(self._complete(messages))
            trace_entry: dict[str, Any] = {
                "round": round_index + 1,
                "raw_calls": [_serialize_raw_call(call) for call in raw_calls],
            }
            provider_metadata = getattr(self.client, "last_response_metadata", None)
            if provider_metadata:
                trace_entry["provider_metadata"] = dict(provider_metadata)
            try:
                step = agent_step_from_tool_calls(raw_calls)
            except ToolCallParseError as exc:
                warning = "tool_call_parse_error"
                repair_warnings.append(warning)
                trace_entry.update({"status": "repair", "error": str(exc)})
                tool_call_trace.append(trace_entry)
                if round_index < self.max_tool_rounds - 1:
                    messages.append(_repair_message(warning, str(exc)))
                    continue
                return _fallback_step(
                    collected_requests,
                    warning,
                    str(exc),
                    repair_warnings=repair_warnings,
                    tool_call_trace=tool_call_trace,
                )

            request_errors = [
                error
                for request in step.tool_requests
                for error in request_state_errors(request, state)
            ]
            if request_errors:
                warning = "invalid_tool_arguments"
                detail = "; ".join(request_errors)
                repair_warnings.append(warning)
                trace_entry.update({"status": "repair", "error": detail})
                tool_call_trace.append(trace_entry)
                if round_index < self.max_tool_rounds - 1:
                    messages.append(_repair_message(warning, detail))
                    continue
                return _fallback_step(
                    collected_requests,
                    warning,
                    detail,
                    repair_warnings=repair_warnings,
                    tool_call_trace=tool_call_trace,
                )

            results = self.runtime.execute_many(state, step.tool_requests)
            collected_requests.extend(step.tool_requests)
            collected_results.extend(results)
            trace_entry["tool_results"] = [asdict(result) for result in results]

            non_final_raw_calls = _non_final_raw_calls(raw_calls)
            if non_final_raw_calls:
                messages.append({"role": "assistant", "tool_calls": _tool_calls_for_messages(non_final_raw_calls)})
                messages.extend(_tool_result_messages(results, non_final_raw_calls))

            if step.final_action is not None:
                errors = final_action_errors(step.final_action, state, collected_results)
                if errors:
                    warning = "invalid_final_action"
                    detail = "; ".join(errors)
                    repair_warnings.append(warning)
                    trace_entry.update({"status": "repair", "error": detail})
                    tool_call_trace.append(trace_entry)
                    if round_index < self.max_tool_rounds - 1:
                        messages.append(_repair_message(warning, detail))
                        continue
                    return _fallback_step(
                        collected_requests,
                        warning,
                        detail,
                        repair_warnings=repair_warnings,
                        tool_call_trace=tool_call_trace,
                    )

                trace_entry["status"] = "accepted"
                tool_call_trace.append(trace_entry)
                metadata: dict[str, Any] = {"tool_call_trace": tool_call_trace}
                if repair_warnings:
                    metadata["agent_repair_warnings"] = repair_warnings
                return AgentStep(tool_requests=collected_requests, final_action=step.final_action, metadata=metadata)

            if not step.tool_requests:
                repair_warnings.append("missing_final_action")
                trace_entry.update({"status": "repair", "error": "Model returned no tool calls."})
                tool_call_trace.append(trace_entry)
                if round_index < self.max_tool_rounds - 1:
                    messages.append(_final_action_repair_message())
                    continue
                break

            trace_entry["status"] = "tools_executed"
            tool_call_trace.append(trace_entry)

        return _fallback_step(
            collected_requests,
            "missing_final_action",
            "Model did not call submit_final_action.",
            repair_warnings=repair_warnings,
            tool_call_trace=tool_call_trace,
        )

    def _complete(self, messages: list[dict[str, Any]]) -> Sequence[Any]:
        if hasattr(self.client, "complete"):
            return self.client.complete(messages, self.tools)  # type: ignore[union-attr]
        return self.client(messages, self.tools)  # type: ignore[misc]


def _fallback_step(
    tool_requests: list[ToolCallRequest],
    warning: str,
    detail: str,
    *,
    repair_warnings: list[str] | None = None,
    tool_call_trace: list[dict[str, Any]] | None = None,
) -> AgentStep:
    metadata: dict[str, Any] = {
        "agent_warnings": [warning],
        "warning_detail": detail,
        "tool_call_trace": tool_call_trace or [],
    }
    if repair_warnings:
        metadata["agent_repair_warnings"] = repair_warnings
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
        metadata=metadata,
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


def _repair_message(warning: str, detail: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"Your previous response was invalid ({warning}): {detail}. "
            "Correct the invalid fields and continue using valid JSON tool calls. "
            "End by calling submit_final_action exactly once with a concise reasoning_summary object."
        ),
    }


def _tool_calls_for_messages(raw_calls: Sequence[Any]) -> list[Any]:
    return list(raw_calls)


def _non_final_raw_calls(raw_calls: Sequence[Any]) -> list[Any]:
    return [call for call in raw_calls if normalize_tool_call(call)[0] != FINAL_ACTION_TOOL]


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


def _serialize_raw_call(raw_call: Any) -> Any:
    if isinstance(raw_call, dict):
        return raw_call
    if hasattr(raw_call, "model_dump"):
        return raw_call.model_dump()
    if hasattr(raw_call, "dict"):
        return raw_call.dict()
    return repr(raw_call)


class ScriptedToolCallingClient:
    """Tiny test/demo client that returns one scripted batch of tool calls per turn."""

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self.turns = list(turns)
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Sequence[Any]:
        del messages, tools
        if self.calls >= len(self.turns):
            return [
                {
                    "name": FINAL_ACTION_TOOL,
                    "arguments": {
                        "action_type": "delay_recommendation",
                        "product_id": None,
                        "reason": "No scripted action remains.",
                        "reasoning_summary": {
                            "observation_facts": [],
                            "evidence_used": [],
                            "decision_rule": "Delay when no scripted action remains.",
                            "confidence": 1.0,
                        },
                    },
                }
            ]
        turn = self.turns[self.calls]
        self.calls += 1
        return turn
