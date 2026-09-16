from __future__ import annotations

from dataclasses import asdict
from typing import Any

from videoshop.simulator.schemas import EnvState, Observation

ALLOWED_TOOLS = [
    "retrieve_candidates",
    "rank_products",
    "get_coupon",
    "find_substitute",
    "explain_recommendation",
]

ALLOWED_ACTIONS = [
    "delay_recommendation",
    "show_product_card",
    "show_coupon",
    "switch_to_substitute",
    "show_explanation",
]


def build_observation(
    state: EnvState,
    task: str = "Assist a short-video shopping session with grounded, low-risk recommendations.",
    last_step: dict[str, Any] | None = None,
    candidate_limit: int | None = None,
) -> Observation:
    user = state.user_profile
    session = state.session_state
    context = state.commerce_context

    return Observation(
        task=task,
        user_summary={
            "user_id": user.user_id,
            "country": user.country,
            "budget_level": user.budget_level,
            "style_preferences": list(user.style_preferences),
            "category_interests": dict(user.category_interests),
            "price_sensitivity": user.price_sensitivity,
            "risk_sensitivity": user.risk_sensitivity,
            "ad_fatigue": user.ad_fatigue,
        },
        session_summary={
            "step": session.step,
            "recent_watch_categories": list(session.recent_watch_categories),
            "recent_clicks": list(session.recent_clicks),
            "recent_carts": list(session.recent_carts),
            "recent_skips": session.recent_skips,
            "exposed_products": list(session.exposed_products),
            "used_coupons": list(session.used_coupons),
        },
        current_video=asdict(state.current_video),
        visible_candidates=[
            {
                "product_id": product.product_id,
                "title": product.title,
                "category": product.category,
                "price": product.price,
                "rating": product.rating,
                "inventory": product.inventory,
                "review_risk": product.review_risk,
                "has_coupon": product.has_coupon,
                "tags": list(product.tags),
            }
            for product in _visible_candidates(state, candidate_limit)
        ],
        commerce_signals={
            "coupon_inventory": dict(context.coupon_inventory),
            "coupon_thresholds": dict(context.coupon_thresholds),
            "coupon_expiry_steps": dict(context.coupon_expiry_steps),
            "stock_pressure": dict(context.stock_pressure),
            "risk_constraints": dict(context.risk_constraints),
        },
        allowed_tools=list(ALLOWED_TOOLS),
        allowed_actions=list(ALLOWED_ACTIONS),
        last_step=last_step,
    )


def _visible_candidates(state: EnvState, candidate_limit: int | None):
    if candidate_limit is None:
        return state.candidate_products
    return state.candidate_products[: max(0, candidate_limit)]
