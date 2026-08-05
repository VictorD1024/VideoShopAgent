from __future__ import annotations

from videoshop.simulator.schemas import AgentAction, EnvState, Product, StateUpdate, UserResponse


def apply_state_update(
    state: EnvState,
    action: AgentAction,
    response: UserResponse,
    product: Product | None,
) -> StateUpdate:
    update = StateUpdate()
    user = state.user_profile
    session = state.session_state
    session.step += 1

    if product and action.action_type != "delay_recommendation":
        session.exposed_products.append(product.product_id)

    if product and response.clicked:
        session.recent_clicks.append(product.product_id)
        update.purchase_intent_delta += 0.10
        update.ad_fatigue_delta -= 0.03
        update.interest_updates[product.category] = 0.10

    if product and response.added_to_cart:
        session.recent_carts.append(product.product_id)
        update.purchase_intent_delta += 0.20

    if response.skipped or response.interrupted or response.irrelevant_recommendation:
        session.recent_skips += 1
        update.ad_fatigue_delta += 0.12
        update.purchase_intent_delta -= 0.05

    if action.action_type == "delay_recommendation":
        update.ad_fatigue_delta -= 0.02

    if response.purchased:
        update.episode_done = True

    user.purchase_intent = _clamp(user.purchase_intent + update.purchase_intent_delta)
    user.ad_fatigue = _clamp(user.ad_fatigue + update.ad_fatigue_delta)
    for category, delta in update.interest_updates.items():
        user.category_interests[category] = _clamp(user.category_interests.get(category, 0.0) + delta)

    return update


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))

