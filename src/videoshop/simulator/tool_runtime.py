from __future__ import annotations

from dataclasses import asdict

from videoshop.simulator.schemas import AgentAction, EnvState, ToolCall, ToolCallRequest, ToolCallResult
from videoshop.simulator.tools import (
    explain_recommendation,
    find_substitute,
    get_coupon,
    rank_products,
    retrieve_candidates,
)


class ToolRuntime:
    """Executes closed-set VideoShop tools against the current environment state."""

    def execute(self, state: EnvState, request: ToolCallRequest) -> ToolCallResult:
        try:
            call = self._execute_tool(state, request)
            return ToolCallResult(tool=call.tool, input=call.input, output=call.output)
        except Exception as exc:  # Keep agent episodes alive after invalid tool use.
            return ToolCallResult(tool=request.tool, input=dict(request.args), output={}, success=False, error=str(exc))

    def execute_many(self, state: EnvState, requests: list[ToolCallRequest]) -> list[ToolCallResult]:
        return [self.execute(state, request) for request in requests]

    def validate_action_evidence(self, action: AgentAction, results: list[ToolCallResult]) -> list[str]:
        """Check that an action is grounded in the tool calls that preceded it.

        Matching the evidence against the tool output is not sufficient on its own:
        the agent could quote a real `find_substitute` result and then switch to a
        different product. Evidence is therefore bound on three sides, the tool input,
        the tool output, and the product the action finally commits to.
        """

        violations: list[str] = []
        successful = [result for result in results if result.success]
        tool_outputs = {result.tool: result.output for result in successful}
        tool_inputs = {result.tool: result.input for result in successful}

        if action.action_type == "show_coupon":
            coupon = action.evidence.get("coupon")
            if not coupon:
                violations.append("missing_coupon_evidence")
            elif coupon != tool_outputs.get("get_coupon"):
                violations.append("coupon_evidence_not_grounded_in_tool_result")
            elif tool_inputs.get("get_coupon", {}).get("product_id") != action.product_id:
                violations.append("coupon_evidence_is_for_a_different_product")
            elif not coupon.get("available"):
                violations.append("coupon_evidence_says_unavailable")

        if action.action_type == "switch_to_substitute":
            substitute = action.evidence.get("substitute")
            if not substitute:
                violations.append("missing_substitute_evidence")
            elif substitute != tool_outputs.get("find_substitute"):
                violations.append("substitute_evidence_not_grounded_in_tool_result")
            elif substitute.get("product_id") != action.product_id:
                violations.append("substitute_evidence_is_for_a_different_product")

        if action.action_type == "show_explanation" and not action.evidence:
            violations.append("missing_explanation_evidence")

        return violations

    def _execute_tool(self, state: EnvState, request: ToolCallRequest) -> ToolCall:
        if request.tool == "retrieve_candidates":
            limit = int(request.args.get("limit", 5))
            _, call = retrieve_candidates(state, limit=limit)
            return call

        if request.tool == "rank_products":
            product_ids = request.args.get("candidate_product_ids")
            candidates = _products_by_ids(state, product_ids) if product_ids else list(state.candidate_products)
            _, call = rank_products(state, candidates)
            return call

        if request.tool == "get_coupon":
            product = _product_by_id(state, str(request.args["product_id"]))
            return get_coupon(state, product)

        if request.tool == "find_substitute":
            product = _product_by_id(state, str(request.args["product_id"]))
            _, call = find_substitute(state, product)
            return call

        if request.tool == "explain_recommendation":
            product = _product_by_id(state, str(request.args["product_id"]))
            return explain_recommendation(state, product)

        raise ValueError(f"Unknown tool: {request.tool}")


def tool_results_to_calls(results: list[ToolCallResult]) -> list[ToolCall]:
    return [
        ToolCall(tool=result.tool, input=dict(result.input), output=dict(result.output))
        for result in results
        if result.success
    ]


def tool_results_as_dicts(results: list[ToolCallResult]) -> list[dict]:
    return [asdict(result) for result in results]


def _product_by_id(state: EnvState, product_id: str):
    for product in state.candidate_products:
        if product.product_id == product_id:
            return product
    raise ValueError(f"Unknown product_id: {product_id}")


def _products_by_ids(state: EnvState, product_ids: list[str]):
    return [_product_by_id(state, product_id) for product_id in product_ids]
