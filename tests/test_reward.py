from videoshop.reward import compute_reward
from videoshop.schemas import Product, UserResponse


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
    response = UserResponse(clicked=True, added_to_cart=True, purchased=True)

    assert compute_reward(response, product) == 16.0

