from __future__ import annotations

import random

from videoshop.schemas import AgentAction, EnvState, Product
from videoshop.user_simulator import score_product_match


class RandomPolicy:
    action_types = [
        "delay_recommendation",
        "show_product_card",
        "show_review_summary",
        "show_coupon",
        "recommend_similar",
        "recommend_bundle",
        "switch_to_cheaper_item",
        "recommend_creator_video",
    ]

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def act(self, state: EnvState) -> AgentAction:
        action_type = self.rng.choice(self.action_types)
        product = self.rng.choice(state.candidate_products) if state.candidate_products else None
        return AgentAction(
            action_type=action_type,
            product_id=product.product_id if product else None,
            product_ids=[product.product_id] if product else [],
            reason="Random baseline action.",
        )


class RuleBasedPolicy:
    def act(self, state: EnvState) -> AgentAction:
        if state.user_profile.ad_fatigue > 0.75 or state.session_state.recent_skips >= 3:
            return AgentAction(action_type="delay_recommendation", reason="User has high ad fatigue or recent skips.")

        product = self._best_product(state)
        if product is None:
            return AgentAction(action_type="delay_recommendation", reason="No candidate product.")

        match_score = score_product_match(state, product)
        if match_score < 0.25:
            return AgentAction(action_type="delay_recommendation", reason="Best product is weakly related to current context.")
        if product.review_risk > 0.35:
            return AgentAction(action_type="show_review_summary", product_id=product.product_id, reason="Product matches context but has review risk.")
        if state.user_profile.price_sensitivity > 0.65 and state.user_profile.budget_level == "low":
            return AgentAction(action_type="show_coupon", product_id=product.product_id, reason="User is price sensitive.")
        if match_score > 0.65:
            return AgentAction(action_type="recommend_bundle", product_id=product.product_id, product_ids=[product.product_id], reason="Strong context match; try bundle conversion.")
        return AgentAction(action_type="show_product_card", product_id=product.product_id, reason="Product matches user and video context.")

    def _best_product(self, state: EnvState) -> Product | None:
        if not state.candidate_products:
            return None
        return max(
            state.candidate_products,
            key=lambda product: score_product_match(state, product) + 0.05 * product.rating - 0.2 * product.review_risk,
        )

