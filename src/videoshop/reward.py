from __future__ import annotations

from dataclasses import dataclass

from videoshop.schemas import Product, UserResponse


@dataclass(frozen=True)
class RewardConfig:
    click_product_card: float = 1.0
    increase_watch_time: float = 1.0
    add_to_cart: float = 3.0
    purchase: float = 10.0
    bundle_purchase: float = 15.0
    ad_click: float = 2.0
    high_margin_purchase: float = 2.0
    inventory_clearance_purchase: float = 2.0
    return_or_refund: float = -8.0
    irrelevant_recommendation: float = -3.0
    interrupt_user_experience: float = -2.0


def compute_reward(
    response: UserResponse,
    product: Product | None = None,
    config: RewardConfig | None = None,
) -> float:
    cfg = config or RewardConfig()
    reward = 0.0

    if response.clicked:
        reward += cfg.click_product_card
    if response.increased_watch_time:
        reward += cfg.increase_watch_time
    if response.added_to_cart:
        reward += cfg.add_to_cart
    if response.purchased:
        reward += cfg.purchase
    if response.bundle_purchased:
        reward += cfg.bundle_purchase
    if response.ad_clicked:
        reward += cfg.ad_click
    if response.returned_or_refunded:
        reward += cfg.return_or_refund
    if response.irrelevant_recommendation:
        reward += cfg.irrelevant_recommendation
    if response.interrupted:
        reward += cfg.interrupt_user_experience

    if product and response.purchased and product.is_high_margin:
        reward += cfg.high_margin_purchase
    if product and response.purchased and product.is_clearance:
        reward += cfg.inventory_clearance_purchase

    return reward

