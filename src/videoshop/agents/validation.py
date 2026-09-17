from __future__ import annotations

from typing import Any

from videoshop.agents.tool_specs import ACTION_TYPES, FINAL_ACTION_TOOL
from videoshop.simulator.schemas import EnvState, FinalAction, ToolCallRequest, ToolCallResult


PRODUCT_TOOLS = {"get_coupon", "find_substitute", "explain_recommendation"}
KNOWN_TOOLS = {"retrieve_candidates", "rank_products", *PRODUCT_TOOLS}
PRODUCT_ACTIONS = {
    "show_product_card",
    "show_coupon",
    "switch_to_substitute",
    "show_explanation",
}
REQUIRED_REASONING_FIELDS = {"observation_facts", "evidence_used", "decision_rule"}


def tool_argument_errors(name: str, args: dict[str, Any]) -> list[str]:
    if name == FINAL_ACTION_TOOL:
        return _final_action_argument_errors(args)
    if name not in KNOWN_TOOLS:
        return [f"unknown tool: {name}"]

    if name == "retrieve_candidates":
        errors = _extra_key_errors(args, {"limit"})
        limit = args.get("limit", 5)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            errors.append("retrieve_candidates.limit must be an integer from 1 to 20")
        return errors

    if name == "rank_products":
        errors = _extra_key_errors(args, {"candidate_product_ids"})
        product_ids = args.get("candidate_product_ids")
        if product_ids is not None and (
            not isinstance(product_ids, list)
            or any(not isinstance(product_id, str) or not product_id for product_id in product_ids)
        ):
            errors.append("rank_products.candidate_product_ids must be an array of non-empty strings")
        return errors

    errors = _extra_key_errors(args, {"product_id"})
    product_id = args.get("product_id")
    if not isinstance(product_id, str) or not product_id.strip():
        errors.append(f"{name}.product_id must be a non-empty string")
    return errors


def request_state_errors(request: ToolCallRequest, state: EnvState) -> list[str]:
    errors = tool_argument_errors(request.tool, request.args)
    candidate_ids = {product.product_id for product in state.candidate_products}
    if request.tool == "rank_products":
        unknown = [
            product_id
            for product_id in request.args.get("candidate_product_ids") or []
            if product_id not in candidate_ids
        ]
        if unknown:
            errors.append(f"rank_products contains unknown product_ids: {unknown}")
    if request.tool in PRODUCT_TOOLS:
        product_id = request.args.get("product_id")
        if isinstance(product_id, str) and product_id not in candidate_ids:
            errors.append(f"{request.tool} contains unknown product_id: {product_id}")
    return errors


def final_action_errors(
    action: FinalAction,
    state: EnvState,
    tool_results: list[ToolCallResult],
    *,
    require_reasoning_summary: bool = True,
) -> list[str]:
    errors: list[str] = []
    candidate_ids = {product.product_id for product in state.candidate_products}

    if action.action_type not in ACTION_TYPES:
        errors.append(f"unknown action_type: {action.action_type}")
    if not action.reason.strip():
        errors.append("final action reason must be non-empty")
    if action.action_type == "delay_recommendation" and action.product_id is not None:
        errors.append("delay_recommendation requires product_id=null")
    if action.action_type in PRODUCT_ACTIONS and not action.product_id:
        errors.append(f"{action.action_type} requires a product_id")
    if action.product_id and action.product_id not in candidate_ids:
        errors.append(f"final action contains unknown product_id: {action.product_id}")
    if require_reasoning_summary:
        errors.extend(reasoning_summary_errors(action.reasoning_summary))

    successful = [result for result in tool_results if result.success]
    if action.action_type == "show_coupon" and not any(
        result.tool == "get_coupon"
        and result.input.get("product_id") == action.product_id
        and bool(result.output.get("available"))
        for result in successful
    ):
        errors.append("show_coupon requires an available get_coupon result for the same product")
    if action.action_type == "show_explanation" and not any(
        result.tool == "explain_recommendation" and result.input.get("product_id") == action.product_id
        for result in successful
    ):
        errors.append("show_explanation requires explain_recommendation evidence for the same product")
    if action.action_type == "switch_to_substitute" and not any(
        result.tool == "find_substitute" and result.output.get("product_id") == action.product_id
        for result in successful
    ):
        errors.append("switch_to_substitute requires a matching find_substitute result")
    return errors


def reasoning_summary_errors(summary: Any) -> list[str]:
    if not isinstance(summary, dict) or not summary:
        return ["reasoning_summary must be a non-empty object"]
    if summary.get("parse_error"):
        return ["reasoning_summary contains invalid JSON"]

    errors: list[str] = []
    missing = sorted(REQUIRED_REASONING_FIELDS - set(summary))
    if missing:
        errors.append(f"reasoning_summary is missing required fields: {missing}")
    for field in ("observation_facts", "evidence_used", "candidate_comparison", "rejected_options"):
        if field in summary and (
            not isinstance(summary[field], list)
            or any(not isinstance(item, str) for item in summary[field])
        ):
            errors.append(f"reasoning_summary.{field} must be an array of strings")
    if "decision_rule" in summary and (
        not isinstance(summary["decision_rule"], str) or not summary["decision_rule"].strip()
    ):
        errors.append("reasoning_summary.decision_rule must be a non-empty string")
    confidence = summary.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        errors.append("reasoning_summary.confidence must be between 0 and 1")
    return errors


def _final_action_argument_errors(args: dict[str, Any]) -> list[str]:
    allowed = {"action_type", "product_id", "reason", "reasoning_summary"}
    errors = _extra_key_errors(args, allowed)
    for required in ("action_type", "product_id", "reason"):
        if required not in args:
            errors.append(f"submit_final_action is missing required field: {required}")
    action_type = args.get("action_type")
    if action_type is not None and action_type not in ACTION_TYPES:
        errors.append(f"submit_final_action.action_type is invalid: {action_type}")
    product_id = args.get("product_id")
    if product_id is not None and not isinstance(product_id, str):
        errors.append("submit_final_action.product_id must be a string or null")
    reason = args.get("reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        errors.append("submit_final_action.reason must be a non-empty string")
    if "reasoning_summary" in args:
        errors.extend(reasoning_summary_errors(args["reasoning_summary"]))
    return errors


def _extra_key_errors(args: dict[str, Any], allowed: set[str]) -> list[str]:
    extra = sorted(set(args) - allowed)
    return [f"unexpected arguments: {extra}"] if extra else []
