from videoshop.simulator.reward import compute_reward
from videoshop.simulator.schemas import AgentAction, Product, UserResponse


def test_compute_reward_with_purchase_and_high_margin_product():
    product = Product(
        product_id="P001",
        title="Desk Organizer",
        category="home organization",
        price=24.99,
        rating=4.6,
        inventory=100,
        review_risk=0.1,
        is_high_margin=True,
    )
    action = AgentAction(action_type="show_product_card", product_id="P001")
    response = UserResponse(clicked=True, added_to_cart=True, purchased=True)

    assert compute_reward(response, action, product) == 14.0


def test_coupon_reward_requires_grounded_coupon_evidence():
    product = Product(
        product_id="P001",
        title="Desk Organizer",
        category="home organization",
        price=24.99,
        rating=4.6,
        inventory=100,
        review_risk=0.1,
        has_coupon=True,
    )
    action = AgentAction(action_type="show_coupon", product_id="P001")
    response = UserResponse(clicked=True)

    assert compute_reward(response, action, product) == -4.0


def test_valid_coupon_reward_uses_tool_evidence():
    product = Product(
        product_id="P001",
        title="Desk Organizer",
        category="home organization",
        price=24.99,
        rating=4.6,
        inventory=100,
        review_risk=0.1,
        has_coupon=True,
    )
    action = AgentAction(
        action_type="show_coupon",
        product_id="P001",
        evidence={"coupon": {"available": True, "discount": 0.15}},
    )
    response = UserResponse(clicked=True)

    assert compute_reward(response, action, product) == 3.0


def test_refunded_purchase_does_not_receive_purchase_proxy_reward():
    product = Product(
        product_id="P001",
        title="Desk Organizer",
        category="home organization",
        price=24.99,
        rating=4.6,
        inventory=100,
        review_risk=0.1,
    )
    action = AgentAction(action_type="show_product_card", product_id="P001")
    response = UserResponse(clicked=True, added_to_cart=True, purchased=True, returned_or_refunded=True)

    assert compute_reward(response, action, product) == -4.0
