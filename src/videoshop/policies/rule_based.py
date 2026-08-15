from __future__ import annotations

from videoshop.simulator.schemas import AgentAction, EnvState
from videoshop.simulator.tools import (
    explain_recommendation,
    find_substitute,
    get_coupon,
    rank_products,
    retrieve_candidates,
)


class RuleBasedPolicy:
    def act(self, state: EnvState) -> AgentAction:
        if state.user_profile.ad_fatigue > 0.75 or state.session_state.recent_skips >= 3:
            return AgentAction(action_type="delay_recommendation", reason="High ad fatigue or repeated skips.")

        candidates, retrieve_call = retrieve_candidates(state)
        ranked, rank_call = rank_products(state, candidates)
        if not ranked:
            return AgentAction(action_type="delay_recommendation", reason="No candidate product.")

        product = ranked[0]
        tool_calls = [retrieve_call, rank_call]

        if product.review_risk >= 0.35:
            explain_call = explain_recommendation(state, product)
            tool_calls.append(explain_call)
            return AgentAction(
                action_type="show_explanation",
                product_id=product.product_id,
                reason="Product is relevant but has review risk, so show grounded explanation first.",
                tool_calls=tool_calls,
                evidence=explain_call.output["evidence"],
            )

        if state.user_profile.price_sensitivity > 0.65:
            coupon_call = get_coupon(state, product)
            tool_calls.append(coupon_call)
            if coupon_call.output["available"]:
                return AgentAction(
                    action_type="show_coupon",
                    product_id=product.product_id,
                    reason="User is price sensitive and a valid coupon is available.",
                    tool_calls=tool_calls,
                    evidence={"coupon": coupon_call.output},
                )

            substitute, substitute_call = find_substitute(state, product)
            tool_calls.append(substitute_call)
            if substitute:
                return AgentAction(
                    action_type="switch_to_substitute",
                    product_id=substitute.product_id,
                    reason="User is price sensitive and a lower-risk or cheaper substitute is available.",
                    tool_calls=tool_calls,
                    evidence={"substitute": substitute_call.output},
                )

        return AgentAction(
            action_type="show_product_card",
            product_id=product.product_id,
            reason="Top ranked product matches user and video context.",
            tool_calls=tool_calls,
        )
