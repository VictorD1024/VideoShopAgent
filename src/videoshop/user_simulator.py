from __future__ import annotations

import random

from videoshop.schemas import AgentAction, EnvState, Product, UserResponse


def find_product(state: EnvState, product_id: str | None) -> Product | None:
    if product_id is None:
        return None
    for product in state.candidate_products:
        if product.product_id == product_id:
            return product
    return None


def score_product_match(state: EnvState, product: Product | None) -> float:
    if product is None:
        return 0.0

    user = state.user_profile
    video = state.current_video
    score = 0.0

    if product.category in user.category_interests:
        score += 0.35
    if product.category in state.session_state.recent_watch_categories:
        score += 0.15
    if product.category in video.scene:
        score += 0.15
    if any(tag in user.style_preferences for tag in product.tags):
        score += 0.15
    if any(obj in product.title.lower() for obj in video.objects):
        score += 0.10
    if product.rating >= 4.5:
        score += 0.10

    return min(score, 1.0)


class UserSimulator:
    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def respond(self, state: EnvState, action: AgentAction) -> UserResponse:
        product = find_product(state, action.product_id)
        match_score = score_product_match(state, product)

        if action.action_type == "delay_recommendation":
            return UserResponse(increased_watch_time=True)

        interruption_prob = min(0.65, state.user_profile.ad_fatigue + 0.08 * state.session_state.recent_skips)
        irrelevant = match_score < 0.25
        clicked = self.rng.random() < (0.08 + 0.55 * match_score)
        added_to_cart = clicked and self.rng.random() < (0.12 + 0.35 * match_score)

        coupon_boost = 0.12 if action.action_type == "show_coupon" else 0.0
        review_boost = 0.08 if action.action_type == "show_review_summary" else 0.0
        bundle_boost = 0.10 if action.action_type == "recommend_bundle" else 0.0
        purchase_prob = 0.05 + 0.25 * match_score + coupon_boost + review_boost
        purchased = added_to_cart and self.rng.random() < purchase_prob
        bundle_purchased = purchased and action.action_type == "recommend_bundle" and self.rng.random() < (0.25 + bundle_boost)

        return_risk = product.review_risk if product else 0.2
        returned = purchased and self.rng.random() < return_risk

        return UserResponse(
            clicked=clicked,
            increased_watch_time=clicked and self.rng.random() < 0.35,
            added_to_cart=added_to_cart,
            purchased=purchased,
            bundle_purchased=bundle_purchased,
            ad_clicked=clicked and self.rng.random() < 0.18,
            returned_or_refunded=returned,
            irrelevant_recommendation=irrelevant,
            interrupted=self.rng.random() < interruption_prob and action.action_type != "delay_recommendation",
            metadata={"match_score": match_score},
        )

