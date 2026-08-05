from __future__ import annotations

from dataclasses import dataclass

from videoshop.simulator.schemas import AgentAction, Product, UserResponse


@dataclass(frozen=True)
class RewardConfig:
    click_product_card: float = 1.0
    add_to_cart: float = 3.0
    purchase_proxy: float = 10.0
    valid_coupon: float = 2.0
    valid_substitute: float = 2.0
    grounded_explanation: float = 1.0
    wrong_category: float = -4.0
    fake_coupon: float = -5.0
    unexplained_return_risk: float = -4.0
    unsupported_explanation: float = -3.0
    interrupt_user_experience: float = -2.0
    return_or_refund: float = -8.0


def compute_reward(
    response: UserResponse,
    action: AgentAction,
    product: Product | None = None,
    category_match: bool = True,
    config: RewardConfig | None = None,
) -> float:
    cfg = config or RewardConfig()
    reward = 0.0

    if response.clicked:
        reward += cfg.click_product_card
    if response.added_to_cart:
        reward += cfg.add_to_cart
    if response.purchased:
        reward += cfg.purchase_proxy
    if action.action_type == "show_coupon" and product and product.has_coupon:
        reward += cfg.valid_coupon
    if action.action_type == "switch_to_substitute" and product:
        reward += cfg.valid_substitute
    if action.action_type == "show_explanation" and action.evidence:
        reward += cfg.grounded_explanation

    if not category_match and action.action_type != "delay_recommendation":
        reward += cfg.wrong_category
    if action.action_type == "show_coupon" and product and not product.has_coupon:
        reward += cfg.fake_coupon
    if product and product.review_risk >= 0.35 and action.action_type != "show_explanation":
        reward += cfg.unexplained_return_risk
    if action.action_type == "show_explanation" and not action.evidence:
        reward += cfg.unsupported_explanation
    if response.interrupted:
        reward += cfg.interrupt_user_experience
    if response.returned_or_refunded:
        reward += cfg.return_or_refund

    return reward

