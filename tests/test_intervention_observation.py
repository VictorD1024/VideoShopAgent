from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from videoshop.agents.prompt import observation_to_user_message
from videoshop.feed.observation import (
    COUPON_AUTHORITY_FIELDS,
    InterventionObservation,
    build_intervention_observation,
    build_intervention_system_prompt,
)
from videoshop.feed.schemas import BaseScores, Eligibility, ExposureCandidate
from videoshop.simulator.observation import build_observation
from videoshop.simulator.schemas import CommerceContext, EnvState, Product, SessionState, UserProfile, VideoContext
from videoshop.simulator.tools import coupon_is_available


@pytest.fixture()
def product() -> Product:
    return Product(
        product_id="P0001",
        title="Silent desk fan",
        category="home_appliance",
        price=89.0,
        rating=4.6,
        inventory=40,
        review_risk=0.12,
        has_coupon=True,
        tags=("quiet",),
    )


@pytest.fixture()
def state(product: Product) -> EnvState:
    return EnvState(
        user_profile=UserProfile(
            user_id="U001",
            country="US",
            budget_level="mid",
            style_preferences=("minimal",),
            category_interests={"home_appliance": 0.7},
            price_sensitivity=0.4,
            risk_sensitivity=0.3,
            ad_fatigue=0.2,
            purchase_intent=0.5,
        ),
        session_state=SessionState(step=1),
        current_video=VideoContext(
            video_id="V001",
            caption="Desk setup refresh",
            scene="home_office",
            category="home_appliance",
            objects=["fan"],
            styles=["minimal"],
            creator_type="reviewer",
        ),
        candidate_products=[product],
        commerce_context=CommerceContext(
            coupon_inventory={"P0001": 0},
            coupon_thresholds={"P0001": 50.0},
            coupon_expiry_steps={"P0001": 8},
            stock_pressure={"P0001": 0.2},
            risk_constraints={"max_review_risk": 0.6},
            campaign_budget=500.0,
        ),
    )


@pytest.fixture()
def exposure() -> ExposureCandidate:
    return ExposureCandidate(
        exposure_id="S:001:e0",
        video_id="V001",
        source_type="ad",
        product_id="P0001",
        treatment="coupon",
        placement="for_you",
        base_scores=BaseScores(),
        eligibility=Eligibility(),
    )


def test_the_v1_observation_gives_the_coupon_answer_away(state: EnvState, product: Product) -> None:
    """Regression guard for why the intervention view exists at all."""

    signals = build_observation(state).commerce_signals
    assert signals["coupon_inventory"]["P0001"] == 0
    assert coupon_is_available(state, product) is False


def test_the_intervention_view_withholds_the_coupon_authority(
    state: EnvState, exposure: ExposureCandidate
) -> None:
    observation = build_intervention_observation(state, exposure)

    for field in COUPON_AUTHORITY_FIELDS:
        assert field not in observation.commerce_signals
    blob = json.dumps(asdict(observation))
    assert "coupon_inventory" not in blob
    assert "coupon_expiry_steps" not in blob


def test_platform_context_survives_the_redaction(state: EnvState, exposure: ExposureCandidate) -> None:
    """Only the fields that decide redemption go; the rest of the brief stays."""

    observation = build_intervention_observation(state, exposure)

    assert observation.commerce_signals["stock_pressure"] == {"P0001": 0.2}
    assert observation.commerce_signals["risk_constraints"] == {"max_review_risk": 0.6}
    assert observation.visible_candidates[0]["has_coupon"] is True


def test_the_agent_learns_which_exposure_it_is_dressing(
    state: EnvState, exposure: ExposureCandidate
) -> None:
    observation = build_intervention_observation(state, exposure)

    assert observation.exposure["anchor_product_id"] == "P0001"
    assert observation.exposure["offered_treatment"] == "coupon"
    assert observation.exposure["source_type"] == "ad"
    assert observation.exposure["video_id"] == "V001"
    assert observation.exposure["placement"] == "for_you"


def test_the_rankers_claim_is_shown_as_a_claim(state: EnvState, exposure: ExposureCandidate) -> None:
    """The offer says the coupon is valid while the tool says it is not.

    That contradiction is the task, so the agent has to be able to see the claim.
    """

    observation = build_intervention_observation(state, exposure)

    assert observation.exposure["offered_eligibility"]["coupon_valid"] is True
    assert coupon_is_available(state, state.candidate_products[0]) is False


def test_an_organic_exposure_reports_no_anchor(state: EnvState) -> None:
    organic = ExposureCandidate(exposure_id="S:001:e1", video_id="V001", source_type="organic")

    observation = build_intervention_observation(state, organic)

    assert observation.exposure["anchor_product_id"] is None
    assert observation.exposure["offered_treatment"] == "none"


def test_it_serialises_into_a_prompt_like_the_v1_observation(
    state: EnvState, exposure: ExposureCandidate
) -> None:
    message = observation_to_user_message(build_intervention_observation(state, exposure))
    payload = json.loads(message["content"])["observation"]

    assert message["role"] == "user"
    assert payload["exposure"]["anchor_product_id"] == "P0001"
    assert "coupon_inventory" not in payload["commerce_signals"]


def test_the_brief_tells_the_model_the_anchor_is_fixed() -> None:
    prompt = build_intervention_system_prompt()

    assert "get_coupon" in prompt
    assert "find_substitute" in prompt
    assert "submit_final_action" in prompt
    assert "not in the observation" in prompt


def test_intervention_few_shots_start_from_the_exposure_not_a_product_search():
    from videoshop.feed.observation import intervention_few_shot_messages

    blob = " ".join(message["content"] for message in intervention_few_shot_messages())
    assert "anchor_product_id" in blob
    assert "offered_treatment" in blob
    assert "coupon_inventory" not in blob
    assert "get_coupon" in blob


def test_defaults_keep_the_dataclass_constructible() -> None:
    observation = InterventionObservation(task="t", exposure={})

    assert "get_coupon" in observation.allowed_tools
    assert observation.last_step is None
