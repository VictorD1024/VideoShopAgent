from videoshop.simulator.schemas import CommerceContext, EnvState, Product, SessionState, UserProfile, VideoContext
from videoshop.simulator.tools import find_substitute, get_coupon


def _state(products: list[Product], commerce_context: CommerceContext) -> EnvState:
    return EnvState(
        user_profile=UserProfile(
            user_id="U001",
            country="US",
            budget_level="low",
            style_preferences=["desk"],
            category_interests={"home_organization": 0.8},
            price_sensitivity=0.8,
            risk_sensitivity=0.4,
            ad_fatigue=0.2,
            purchase_intent=0.4,
        ),
        session_state=SessionState(step=1),
        current_video=VideoContext(
            video_id="V001",
            caption="Desk setup",
            scene="home office",
            category="home_organization",
            objects=["organizer"],
            styles=["desk"],
            creator_type="home creator",
        ),
        candidate_products=products,
        commerce_context=commerce_context,
    )


def test_get_coupon_requires_context_inventory_budget_and_threshold():
    product = Product("P001", "Desk Organizer", "home_organization", 24.99, 4.6, 10, 0.1, has_coupon=True, coupon_discount=0.15)
    state = _state(
        [product],
        CommerceContext(
            coupon_inventory={"P001": 2},
            coupon_thresholds={"P001": 20.0},
            coupon_expiry_steps={"P001": 2},
            campaign_budget=1.0,
        ),
    )

    call = get_coupon(state, product)

    assert call.output["available"] is True
    assert call.output["discount"] == 0.15


def test_get_coupon_rejects_expired_or_depleted_coupon():
    product = Product("P001", "Desk Organizer", "home_organization", 24.99, 4.6, 10, 0.1, has_coupon=True, coupon_discount=0.15)
    state = _state(
        [product],
        CommerceContext(
            coupon_inventory={"P001": 0},
            coupon_thresholds={"P001": 20.0},
            coupon_expiry_steps={"P001": 0},
            campaign_budget=1.0,
        ),
    )

    call = get_coupon(state, product)

    assert call.output["available"] is False
    assert call.output["discount"] == 0.0


def test_find_substitute_respects_risk_constraints_and_inventory():
    risky = Product("P001", "Premium Desk Organizer", "home_organization", 30.0, 4.6, 5, 0.5)
    out_of_stock = Product("P002", "Cheap Desk Organizer", "home_organization", 10.0, 4.8, 0, 0.1)
    substitute = Product("P003", "Safe Desk Organizer", "home_organization", 20.0, 4.5, 10, 0.2, tags=["desk"])
    state = _state(
        [risky, out_of_stock, substitute],
        CommerceContext(
            stock_pressure={"P003": 0.7},
            risk_constraints={"max_review_risk": 0.35},
        ),
    )

    result, call = find_substitute(state, risky)

    assert result == substitute
    assert call.output["product_id"] == "P003"
