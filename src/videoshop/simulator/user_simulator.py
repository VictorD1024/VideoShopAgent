from __future__ import annotations

import random

from videoshop.simulator.schemas import AgentAction, EnvState, Product, UserResponse
from videoshop.simulator.tools import product_match_score


class UserSimulator:
    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def respond(self, state: EnvState, action: AgentAction, product: Product | None) -> UserResponse:
        if action.action_type == "delay_recommendation":
            return UserResponse(increased_watch_time=True)

        match_score = product_match_score(state, product) if product else 0.0
        irrelevant = match_score < 0.25
        interrupted = self.rng.random() < min(0.75, state.user_profile.ad_fatigue + 0.05 * state.session_state.recent_skips)

        explanation_boost = 0.08 if action.action_type == "show_explanation" and action.evidence else 0.0
        coupon_boost = 0.10 if action.action_type == "show_coupon" and product and product.has_coupon else 0.0
        substitute_boost = 0.05 if action.action_type == "switch_to_substitute" else 0.0

        click_prob = 0.08 + 0.55 * match_score + explanation_boost
        clicked = self.rng.random() < click_prob
        cart_prob = 0.10 + 0.35 * match_score + coupon_boost + substitute_boost
        added_to_cart = clicked and self.rng.random() < cart_prob
        purchase_prob = 0.04 + 0.25 * match_score + coupon_boost
        purchased = added_to_cart and self.rng.random() < purchase_prob
        returned = purchased and product is not None and self.rng.random() < product.review_risk

        return UserResponse(
            clicked=clicked,
            increased_watch_time=clicked and self.rng.random() < 0.35,
            added_to_cart=added_to_cart,
            purchased=purchased,
            skipped=not clicked and interrupted,
            returned_or_refunded=returned,
            irrelevant_recommendation=irrelevant,
            interrupted=interrupted,
            metadata={"match_score": match_score},
        )

