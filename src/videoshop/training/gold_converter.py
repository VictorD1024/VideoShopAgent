from __future__ import annotations

import json
from collections import Counter
from typing import Any

from videoshop.agents.prompt import build_system_prompt
from videoshop.agents.tool_specs import ACTION_TYPES, FINAL_ACTION_TOOL, video_shop_tool_specs


REQUIRED_REASONING_FIELDS = {"observation_facts", "evidence_used", "decision_rule"}
PRODUCT_ACTIONS = {
    "show_product_card",
    "show_coupon",
    "switch_to_substitute",
    "show_explanation",
}
REQUIRED_ACTION_TOOLS = {
    "show_coupon": "get_coupon",
    "switch_to_substitute": "find_substitute",
    "show_explanation": "explain_recommendation",
}


def select_gold_episodes(episodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    gold: list[dict[str, Any]] = []
    rejected: dict[str, list[str]] = {}
    for episode in episodes:
        reasons = gold_rejection_reasons(episode)
        if reasons:
            rejected[str(episode.get("episode_id", "unknown"))] = reasons
        else:
            gold.append(episode)
    return gold, rejected


def gold_rejection_reasons(episode: dict[str, Any]) -> list[str]:
    reasons: set[str] = set()
    if episode.get("outcome") == "failed":
        reasons.add("episode_failed")
    if (episode.get("metrics") or {}).get("constraint_violations", 0):
        reasons.add("constraint_violation")

    steps = episode.get("steps") or []
    if not steps:
        reasons.add("empty_episode")

    for step in steps:
        agent_step = step.get("agent_step") or {}
        metadata = agent_step.get("metadata") or {}
        final_action = agent_step.get("final_action") or {}
        reasoning = step.get("reasoning_summary")
        tool_results = step.get("tool_results") or []

        if step.get("violations"):
            reasons.add("step_violation")
        if metadata.get("agent_warnings"):
            reasons.add("agent_warning")
        if metadata.get("agent_repair_warnings"):
            reasons.add("agent_repair_warning")
        if any(not result.get("success", False) for result in tool_results):
            reasons.add("tool_failure")
        if not _valid_reasoning_summary(reasoning):
            reasons.add("invalid_reasoning_summary")

        action_type = final_action.get("action_type")
        product_id = final_action.get("product_id")
        if action_type not in ACTION_TYPES:
            reasons.add("invalid_action_type")
        elif action_type == "delay_recommendation" and product_id is not None:
            reasons.add("delay_has_product_id")
        elif action_type in PRODUCT_ACTIONS and not product_id:
            reasons.add("missing_product_id")

        candidate_ids = {
            product.get("product_id")
            for product in (step.get("state") or {}).get("candidate_products", [])
        }
        if product_id and product_id not in candidate_ids:
            reasons.add("unknown_product_id")

        required_tool = REQUIRED_ACTION_TOOLS.get(action_type)
        successful_tools = {
            result.get("tool") for result in tool_results if result.get("success", False)
        }
        if required_tool and required_tool not in successful_tools:
            reasons.add("missing_action_evidence")

    return sorted(reasons)


def build_openai_tool_sft_samples(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    tools = video_shop_tool_specs()
    for episode in episodes:
        for step in episode["steps"]:
            samples.append(
                {
                    "id": _sample_id(episode, step),
                    "task": "videoshop_agent_tool_calling_sft",
                    "messages": _openai_messages(step),
                    "tools": tools,
                    "metadata": _metadata(episode, step),
                }
            )
    return samples


def build_action_json_sft_samples(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for episode in episodes:
        for step in episode["steps"]:
            agent_step = step["agent_step"]
            target = {
                "tool_requests": agent_step.get("tool_requests", []),
                "final_action": _final_action_arguments(agent_step["final_action"]),
            }
            samples.append(
                {
                    "id": _sample_id(episode, step),
                    "task": "videoshop_agent_action_sft",
                    "messages": [
                        {"role": "system", "content": build_system_prompt()},
                        _observation_message(step),
                        {"role": "assistant", "content": _json_dumps(target)},
                    ],
                    "metadata": _metadata(episode, step),
                }
            )
    return samples


def build_conversion_report(
    episodes: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    rejected: dict[str, list[str]],
    sample_count: int,
) -> dict[str, Any]:
    reason_counts = Counter(reason for reasons in rejected.values() for reason in reasons)
    steps = [step for episode in gold for step in episode.get("steps", [])]
    action_counts = Counter(
        ((step.get("agent_step") or {}).get("final_action") or {}).get("action_type", "missing")
        for step in steps
    )
    tool_counts = Counter(
        request.get("tool", "missing")
        for step in steps
        for request in (step.get("agent_step") or {}).get("tool_requests", [])
    )
    scenario_counts = Counter(_scenario_type(episode.get("scenario_id", "")) for episode in gold)
    return {
        "input_episodes": len(episodes),
        "gold_episodes": len(gold),
        "rejected_episodes": len(rejected),
        "training_samples": sample_count,
        "openai_tool_sequence": "reconstructed_from_flattened_tool_requests",
        "action_type_counts": dict(sorted(action_counts.items())),
        "tool_call_counts": dict(sorted(tool_counts.items())),
        "scenario_type_counts": dict(sorted(scenario_counts.items())),
        "returned_or_refunded_steps": sum(
            bool((step.get("user_response") or {}).get("returned_or_refunded")) for step in steps
        ),
        "gold_episode_ids": [episode["episode_id"] for episode in gold],
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "rejected": rejected,
    }


def _scenario_type(scenario_id: str) -> str:
    value = scenario_id.removeprefix("synthetic_")
    prefix, separator, suffix = value.rpartition("_")
    return prefix if separator and suffix.isdigit() else value


def _openai_messages(step: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
        _observation_message(step),
    ]
    agent_step = step["agent_step"]
    requests = agent_step.get("tool_requests", [])
    results = step.get("tool_results", [])

    if requests:
        tool_calls = []
        for index, request in enumerate(requests):
            call_id = _tool_call_id(step, index)
            tool_calls.append(_tool_call(call_id, request["tool"], request.get("args", {})))
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})

        for index, result in enumerate(results):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _tool_call_id(step, index),
                    "name": result["tool"],
                    "content": _json_dumps(result),
                }
            )

    final_call_id = f"{_sample_id_from_step(step)}_final"
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call(
                    final_call_id,
                    FINAL_ACTION_TOOL,
                    _final_action_arguments(agent_step["final_action"]),
                ),
            ],
        }
    )
    return messages


def _valid_reasoning_summary(reasoning: Any) -> bool:
    if not isinstance(reasoning, dict) or not reasoning or reasoning.get("parse_error"):
        return False
    if not REQUIRED_REASONING_FIELDS.issubset(reasoning):
        return False
    if not isinstance(reasoning.get("observation_facts"), list):
        return False
    if not isinstance(reasoning.get("evidence_used"), list):
        return False
    return isinstance(reasoning.get("decision_rule"), str) and bool(reasoning["decision_rule"].strip())


def _observation_message(step: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "user",
        "content": json.dumps({"observation": step["observation"]}, ensure_ascii=False, indent=2),
    }


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": _json_dumps(arguments)},
    }


def _final_action_arguments(final_action: dict[str, Any]) -> dict[str, Any]:
    allowed = ("action_type", "product_id", "reason", "reasoning_summary")
    return {key: final_action[key] for key in allowed if key in final_action}


def _metadata(episode: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    response = step.get("user_response") or {}
    return {
        "episode_id": episode["episode_id"],
        "scenario_id": episode["scenario_id"],
        "t": step["t"],
        "reward": step.get("reward", 0.0),
        "total_reward": episode.get("total_reward", 0.0),
        "outcome": episode.get("outcome"),
        "clicked": bool(response.get("clicked")),
        "added_to_cart": bool(response.get("added_to_cart")),
        "purchased": bool(response.get("purchased")),
        "returned_or_refunded": bool(response.get("returned_or_refunded")),
        "tool_sequence_reconstructed": bool((step.get("agent_step") or {}).get("tool_requests")),
        "quality_tier": "gold",
    }


def _sample_id(episode: dict[str, Any], step: dict[str, Any]) -> str:
    return f"{episode['episode_id']}_T{int(step['t']):03d}"


def _sample_id_from_step(step: dict[str, Any]) -> str:
    scenario = str(step.get("scenario_id", "scenario"))
    return f"{scenario}_T{int(step['t']):03d}"


def _tool_call_id(step: dict[str, Any], index: int) -> str:
    return f"{_sample_id_from_step(step)}_tool_{index + 1:02d}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
