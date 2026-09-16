from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from videoshop.agents.tool_specs import FINAL_ACTION_TOOL
from videoshop.simulator.schemas import AgentStep, FinalAction, ToolCallRequest


class ToolCallParseError(ValueError):
    pass


def agent_step_from_json(payload: str | dict[str, Any]) -> AgentStep:
    """Parse the plain AgentStep JSON format used for SFT/DPO data."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    tool_requests = [
        ToolCallRequest(tool=str(item["tool"]), args=dict(item.get("args", {})))
        for item in data.get("tool_requests", [])
    ]
    final_data = data.get("final_action")
    final_action = _final_action_from_args(final_data) if final_data else None
    return AgentStep(tool_requests=tool_requests, final_action=final_action)


def agent_step_from_tool_calls(tool_calls: Iterable[Any]) -> AgentStep:
    """Parse provider-native tool/function calls into VideoShopEnv's AgentStep."""
    requests: list[ToolCallRequest] = []
    final_action: FinalAction | None = None

    for raw_call in tool_calls:
        name, args = normalize_tool_call(raw_call)
        if name == FINAL_ACTION_TOOL:
            final_action = _final_action_from_args(args)
        else:
            requests.append(ToolCallRequest(tool=name, args=args))

    return AgentStep(tool_requests=requests, final_action=final_action)


def normalize_tool_call(raw_call: Any) -> tuple[str, dict[str, Any]]:
    """Normalize common OpenAI/Grok/Qwen-style tool call shapes.

    Supported examples:
    - {"name": "get_coupon", "arguments": {"product_id": "P001"}}
    - {"type": "function", "function": {"name": "get_coupon", "arguments": "{...}"}}
    - objects with .function.name and .function.arguments attributes.
    """
    call = _as_mapping(raw_call)
    function = call.get("function")
    if function is not None:
        fn = _as_mapping(function)
        name = fn.get("name")
        arguments = fn.get("arguments", {})
    else:
        name = call.get("name") or call.get("tool")
        arguments = call.get("arguments", call.get("args", {}))

    if not isinstance(name, str) or not name:
        raise ToolCallParseError(f"Tool call is missing a function name: {raw_call!r}")

    return name, _parse_arguments(arguments)


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = _repair_and_load_arguments(arguments)
        if not isinstance(parsed, dict):
            raise ToolCallParseError("Tool call arguments JSON must decode to an object.")
        return parsed
    raise ToolCallParseError(f"Unsupported tool call arguments type: {type(arguments).__name__}")


def _final_action_from_args(args: dict[str, Any] | None) -> FinalAction:
    if args is None:
        raise ToolCallParseError("submit_final_action requires arguments.")
    if "action_type" not in args:
        raise ToolCallParseError(f"submit_final_action is missing required action_type: {args!r}")
    return FinalAction(
        action_type=str(args["action_type"]),
        product_id=_normalize_nullable_product_id(args.get("product_id")),
        reason=str(args.get("reason", "")),
        reasoning_summary=_normalize_reasoning_summary(args.get("reasoning_summary")),
    )


def _normalize_nullable_product_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
        return None
    return str(value)


def _normalize_reasoning_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            summary = {"summary": value}
            if value.startswith(("{", "[")):
                summary["parse_error"] = "invalid_reasoning_summary_json"
            return summary
        if isinstance(parsed, dict):
            return parsed
        return {"summary": parsed}
    return {"summary": value}


def _repair_and_load_arguments(arguments: str) -> dict[str, Any]:
    repaired = arguments.strip()
    repaired = repaired.replace("'", '"')
    repaired = re.sub(r":\s*([A-Za-z][A-Za-z0-9_-]*)\s*([,}])", r': "\1"\2', repaired)
    repaired = re.sub(r",\s*}", "}", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(f"Malformed tool call arguments: {arguments!r}") from exc


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    attrs = {}
    for key in ("type", "name", "tool", "arguments", "args", "function"):
        if hasattr(value, key):
            attrs[key] = getattr(value, key)
    if attrs:
        return attrs
    raise ToolCallParseError(f"Unsupported tool call object: {value!r}")
