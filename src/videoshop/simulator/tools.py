from __future__ import annotations

from videoshop.simulator.schemas import EnvState, Product, ToolCall


def coupon_is_available(state: EnvState, product: Product) -> bool:
    context = state.commerce_context
    threshold = context.coupon_thresholds.get(product.product_id, 0.0)
    expiry_step = context.coupon_expiry_steps.get(product.product_id)
    inventory = context.coupon_inventory.get(product.product_id, 0)
    return (
        product.has_coupon
        and inventory > 0
        and product.price >= threshold
        and (expiry_step is None or state.session_state.step <= expiry_step)
        and context.campaign_budget >= product.coupon_discount
        and product.product_id not in state.session_state.used_coupons
    )


def product_match_score(state: EnvState, product: Product) -> float:
    user = state.user_profile
    video = state.current_video
    score = 0.0

    score += 0.40 * user.category_interests.get(product.category, 0.0)
    if product.category == video.category:
        score += 0.25
    if any(tag in user.style_preferences for tag in product.tags):
        score += 0.15
    if any(obj in product.title.lower() for obj in video.objects):
        score += 0.10
    if product.rating >= 4.5:
        score += 0.10

    return min(score, 1.0)


def retrieve_candidates(state: EnvState, limit: int = 5) -> tuple[list[Product], ToolCall]:
    ranked = sorted(
        state.candidate_products,
        key=lambda product: product_match_score(state, product),
        reverse=True,
    )
    candidates = ranked[:limit]
    return candidates, ToolCall(
        tool="retrieve_candidates",
        input={"video_id": state.current_video.video_id, "user_id": state.user_profile.user_id},
        output={"candidate_product_ids": [product.product_id for product in candidates]},
    )


def rank_products(state: EnvState, candidates: list[Product]) -> tuple[list[Product], ToolCall]:
    ranked = sorted(
        candidates,
        key=lambda product: product_match_score(state, product) + 0.03 * product.rating - 0.25 * product.review_risk,
        reverse=True,
    )
    return ranked, ToolCall(
        tool="rank_products",
        input={"candidate_product_ids": [product.product_id for product in candidates]},
        output={"ranked_product_ids": [product.product_id for product in ranked]},
    )


def get_coupon(state: EnvState, product: Product) -> ToolCall:
    available = coupon_is_available(state, product)
    return ToolCall(
        tool="get_coupon",
        input={"product_id": product.product_id, "user_id": state.user_profile.user_id},
        output={
            "available": available,
            "discount": product.coupon_discount if available else 0.0,
            "inventory": state.commerce_context.coupon_inventory.get(product.product_id, 0),
            "threshold": state.commerce_context.coupon_thresholds.get(product.product_id, 0.0),
        },
    )


def find_substitute(state: EnvState, product: Product) -> tuple[Product | None, ToolCall]:
    max_review_risk = state.commerce_context.risk_constraints.get("max_review_risk", 0.35)
    substitutes = [
        candidate
        for candidate in state.candidate_products
        if candidate.product_id != product.product_id
        and candidate.category == product.category
        and candidate.price <= product.price
        and candidate.review_risk <= product.review_risk
        and candidate.review_risk <= max_review_risk
        and candidate.inventory > 0
    ]
    substitute = max(
        substitutes,
        key=lambda item: (
            product_match_score(state, item),
            state.commerce_context.stock_pressure.get(item.product_id, 0.0),
            -item.price,
        ),
        default=None,
    )
    return substitute, ToolCall(
        tool="find_substitute",
        input={
            "product_id": product.product_id,
            "constraints": {
                "lower_price": True,
                "lower_risk": True,
                "max_review_risk": max_review_risk,
                "in_stock": True,
            },
        },
        output={"product_id": substitute.product_id if substitute else None},
    )


def explain_recommendation(state: EnvState, product: Product) -> ToolCall:
    evidence = {
        "video_scene": state.current_video.scene,
        "product_category": product.category,
        "rating": product.rating,
        "review_risk": product.review_risk,
    }
    return ToolCall(
        tool="explain_recommendation",
        input={"product_id": product.product_id},
        output={"evidence": evidence},
    )
