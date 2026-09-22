from __future__ import annotations

import copy
import random

import pytest

from videoshop.feed.candidate_provider import CandidateProviderConfig, SyntheticCandidateProvider
from videoshop.simulator.schemas import Product, SessionState, UserProfile, VideoContext


def make_user(**overrides) -> UserProfile:
    payload = {
        "user_id": "U1",
        "country": "US",
        "budget_level": "medium",
        "style_preferences": ["minimal"],
        "category_interests": {"kitchen": 0.8},
        "price_sensitivity": 0.3,
        "risk_sensitivity": 0.5,
        "ad_fatigue": 0.1,
        "purchase_intent": 0.6,
    }
    payload.update(overrides)
    return UserProfile(**payload)


def make_product(review_risk: float) -> Product:
    return Product(
        product_id="P1",
        title="Minimal Kitchen Knife",
        category="kitchen",
        price=40.0,
        rating=4.6,
        inventory=100,
        review_risk=review_risk,
        has_coupon=True,
        coupon_discount=0.15,
    )


VIDEO = VideoContext(
    video_id="V1",
    caption="minimal kitchen setup",
    scene="kitchen demo",
    category="kitchen",
    objects=["knife"],
    styles=["minimal"],
    creator_type="kitchen creator",
)


@pytest.fixture()
def provider():
    return SyntheticCandidateProvider([make_product(0.3)], [VIDEO])


def truth_for(provider, user, product):
    return provider.truth_for(user, SessionState(), VIDEO, product, "seller", "product_anchor")


def test_risk_sensitivity_changes_behaviour_not_just_the_label(provider):
    risky_product = make_product(0.5)
    averse = truth_for(provider, make_user(risk_sensitivity=0.95), risky_product)
    tolerant = truth_for(provider, make_user(risk_sensitivity=0.05), risky_product)

    assert averse.click_probability < tolerant.click_probability
    assert averse.purchase_probability < tolerant.purchase_probability


def test_risk_sensitivity_is_inert_on_a_clean_product(provider):
    clean_product = make_product(0.0)
    averse = truth_for(provider, make_user(risk_sensitivity=0.95), clean_product)
    tolerant = truth_for(provider, make_user(risk_sensitivity=0.05), clean_product)

    assert averse.click_probability == tolerant.click_probability
    assert averse.purchase_probability == tolerant.purchase_probability


def test_risk_averse_users_separate_low_from_high_risk_products(provider):
    user = make_user(risk_sensitivity=0.9)
    low = truth_for(provider, user, make_product(0.05))
    high = truth_for(provider, user, make_product(0.5))

    assert low.purchase_probability > high.purchase_probability
    assert low.refund_probability < high.refund_probability


def test_probabilities_stay_bounded_under_extreme_users(provider):
    for risk in (0.0, 1.0):
        for intent in (0.0, 1.0):
            for price_sensitivity in (0.0, 1.0):
                user = make_user(
                    risk_sensitivity=risk, purchase_intent=intent, price_sensitivity=price_sensitivity
                )
                truth = truth_for(provider, user, make_product(0.85))
                assert 0.0 <= truth.click_probability <= 1.0
                assert 0.0 <= truth.purchase_probability <= truth.click_probability


def test_generate_is_deterministic_for_a_seed(provider):
    user = make_user()
    first = provider.generate(user, SessionState(), 0, random.Random(5))
    second = provider.generate(copy.deepcopy(user), SessionState(), 0, random.Random(5))

    assert [candidate.to_dict() for candidate in first.candidates] == [
        candidate.to_dict() for candidate in second.candidates
    ]


def test_a_stale_coupon_buys_no_lift_and_no_discount(provider):
    """Both environments must agree that a coupon which will not redeem does nothing."""
    user, session = make_user(), SessionState()
    product = provider.products[0]

    real = provider.truth_for(user, session, VIDEO, product, "seller", "coupon")
    plain = provider.truth_for(user, session, VIDEO, product, "seller", "product_anchor")

    assert real.purchase_probability > plain.purchase_probability
    assert real.price < plain.price

    config = CandidateProviderConfig(coupon_treatment_rate=1.0, coupon_trap_rate=1.0)
    trapped = SyntheticCandidateProvider(provider.products, provider.videos, config)
    batch = trapped.generate(user, SessionState(), 0, random.Random(3))

    coupons = [c for c in batch.candidates if c.treatment == "coupon"]
    assert coupons, "no coupon candidates were generated"
    for candidate in coupons:
        truth = batch.truth[candidate.exposure_id]
        assert truth.coupon_available is False
        assert truth.price == pytest.approx(plain.price)
        assert truth.purchase_probability == pytest.approx(plain.purchase_probability)


def test_the_ranker_still_advertises_the_trapped_coupon(provider):
    """The trap must not be readable off the public scores, or it is not a trap."""
    user = make_user()
    honest = SyntheticCandidateProvider(
        provider.products,
        provider.videos,
        CandidateProviderConfig(coupon_treatment_rate=1.0, coupon_trap_rate=0.0, score_noise=0.0),
    )
    trapped = SyntheticCandidateProvider(
        provider.products,
        provider.videos,
        CandidateProviderConfig(coupon_treatment_rate=1.0, coupon_trap_rate=1.0, score_noise=0.0),
    )

    honest_batch = honest.generate(user, SessionState(), 0, random.Random(3))
    trapped_batch = trapped.generate(user, SessionState(), 0, random.Random(3))

    honest_scores = {c.exposure_id: c.base_scores.to_dict() for c in honest_batch.candidates}
    trapped_scores = {c.exposure_id: c.base_scores.to_dict() for c in trapped_batch.candidates}
    assert honest_scores == trapped_scores

    # ...while the private truth does differ.
    honest_truth = {k: v.purchase_probability for k, v in honest_batch.truth.items()}
    trapped_truth = {k: v.purchase_probability for k, v in trapped_batch.truth.items()}
    assert honest_truth != trapped_truth


def test_generate_respects_the_configured_mix(provider):
    config = CandidateProviderConfig(candidates_per_step=10, min_organic=3, min_commercial=4)
    provider = SyntheticCandidateProvider(provider.products, provider.videos, config)
    batch = provider.generate(make_user(), SessionState(), 0, random.Random(1))

    organic = [item for item in batch.candidates if item.is_organic]
    commercial = [item for item in batch.candidates if item.is_commercial]
    assert len(batch.candidates) >= 10
    assert len(organic) >= 3
    assert len(commercial) >= 4
    assert set(batch.truth) == {candidate.exposure_id for candidate in batch.candidates}
