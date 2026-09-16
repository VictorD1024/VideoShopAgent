from __future__ import annotations

import copy

from videoshop.data.mock import build_mock_catalog, build_mock_commerce_context, build_mock_users, build_mock_videos
from videoshop.simulator.scenario import VideoShopScenario
from videoshop.simulator.schemas import CommerceContext, Product, SessionState, UserProfile, VideoContext


def build_toy_scenarios() -> list[VideoShopScenario]:
    catalog = build_mock_catalog()
    videos = build_mock_videos()
    users = build_mock_users()

    return [
        _scenario(
            "coupon_valid",
            _user(users, "U001"),
            [_video(videos, "V001")],
            catalog,
            _commerce(coupon_inventory={"P001": 4}, coupon_thresholds={"P001": 20.0}, coupon_expiry_steps={"P001": 5}, campaign_budget=3.0),
            "Use a valid coupon when the top relevant product has an available coupon.",
            ["call_get_coupon_before_show_coupon", "avoid_fake_coupon"],
        ),
        _scenario(
            "fake_coupon_trap",
            _price_sensitive(_user(users, "U001")),
            [_video(videos, "V001")],
            catalog,
            _commerce(coupon_inventory={"P001": 0}, coupon_thresholds={"P001": 20.0}, coupon_expiry_steps={"P001": 5}, campaign_budget=0.0),
            "Do not show a coupon when inventory or campaign budget makes it unavailable.",
            ["avoid_fake_coupon", "prefer_product_card_or_substitute"],
        ),
        _scenario(
            "high_risk_product",
            _user(users, "U003"),
            [_video(videos, "V003")],
            catalog,
            _commerce(),
            "Explain risk before recommending a highly relevant but high-review-risk product.",
            ["ground_explanation", "avoid_unexplained_high_risk"],
        ),
        _scenario(
            "substitute_needed",
            _price_sensitive(_user(users, "U002")),
            [_video(videos, "V002")],
            _without_coupon(catalog, "P003"),
            _commerce(coupon_inventory={}, coupon_thresholds={}, coupon_expiry_steps={}, campaign_budget=0.0),
            "Find a cheaper and lower-risk substitute when the best product has no valid coupon.",
            ["call_find_substitute", "prefer_lower_risk_substitute"],
        ),
        _scenario(
            "delay_when_fatigue_high",
            _fatigued(_user(users, "U001")),
            [_video(videos, "V001")],
            catalog,
            _commerce(),
            "Delay recommendations when ad fatigue and recent skips are high.",
            ["delay_recommendation"],
            session_state=SessionState(recent_skips=3),
        ),
        _scenario(
            "category_mismatch_risk",
            _user(users, "U002"),
            [_video(videos, "V001")],
            catalog,
            _commerce(),
            "Avoid recommending a product whose category does not match the video and user intent.",
            ["avoid_category_mismatch"],
        ),
        _scenario(
            "coupon_expired",
            _price_sensitive(_user(users, "U001")),
            [_video(videos, "V001")],
            catalog,
            _commerce(coupon_inventory={"P001": 3}, coupon_thresholds={"P001": 20.0}, coupon_expiry_steps={"P001": -1}, campaign_budget=3.0),
            "Treat expired coupons as unavailable even when inventory remains.",
            ["avoid_fake_coupon"],
        ),
        _scenario(
            "stock_pressure_clearance",
            _price_sensitive(_user(users, "U001")),
            [_video(videos, "V001")],
            catalog,
            _commerce(coupon_inventory={}, stock_pressure={"P002": 0.9, "P005": 0.6}, campaign_budget=0.0),
            "Prefer a cheaper in-category substitute when stock pressure favors clearance inventory.",
            ["call_find_substitute", "prefer_clearance_substitute"],
        ),
        _scenario(
            "grounded_explanation",
            _user(users, "U004"),
            [_video(videos, "V003")],
            catalog,
            _commerce(),
            "Only provide recommendation explanations grounded in product, video, and risk evidence.",
            ["call_explain_recommendation", "avoid_unsupported_explanation"],
        ),
        _scenario(
            "low_inventory_guard",
            _user(users, "U002"),
            [_video(videos, "V002")],
            _with_inventory(catalog, {"P003": 0, "P006": 180}),
            _commerce(),
            "Avoid recommending an out-of-stock top product when an in-stock substitute exists.",
            ["avoid_out_of_stock", "prefer_available_product"],
        ),
    ]


def _scenario(
    scenario_id: str,
    user: UserProfile,
    videos: list[VideoContext],
    products: list[Product],
    commerce_context: CommerceContext,
    objective: str,
    expected_behaviors: list[str],
    session_state: SessionState | None = None,
) -> VideoShopScenario:
    return VideoShopScenario(
        scenario_id=scenario_id,
        user_profile=copy.deepcopy(user),
        initial_session_state=copy.deepcopy(session_state or SessionState()),
        video_feed=copy.deepcopy(videos),
        candidate_products=copy.deepcopy(products),
        commerce_context=copy.deepcopy(commerce_context),
        objective=objective,
        expected_behaviors=expected_behaviors,
    )


def _commerce(**overrides) -> CommerceContext:
    context = build_mock_commerce_context()
    for key, value in overrides.items():
        setattr(context, key, value)
    return context


def _user(users: list[UserProfile], user_id: str) -> UserProfile:
    return copy.deepcopy(next(user for user in users if user.user_id == user_id))


def _video(videos: list[VideoContext], video_id: str) -> VideoContext:
    return copy.deepcopy(next(video for video in videos if video.video_id == video_id))


def _price_sensitive(user: UserProfile) -> UserProfile:
    user.price_sensitivity = 0.9
    return user


def _fatigued(user: UserProfile) -> UserProfile:
    user.ad_fatigue = 0.9
    return user


def _without_coupon(products: list[Product], product_id: str) -> list[Product]:
    cloned = copy.deepcopy(products)
    product = next(item for item in cloned if item.product_id == product_id)
    product.has_coupon = False
    product.coupon_discount = 0.0
    return cloned


def _with_inventory(products: list[Product], inventory_by_id: dict[str, int]) -> list[Product]:
    cloned = copy.deepcopy(products)
    for product in cloned:
        if product.product_id in inventory_by_id:
            product.inventory = inventory_by_id[product.product_id]
    return cloned
