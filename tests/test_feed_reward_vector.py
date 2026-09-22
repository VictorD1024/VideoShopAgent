from __future__ import annotations

import pytest

from videoshop.feed import (
    SCALARIZATIONS,
    FeedDecision,
    RewardConfig,
    RewardVector,
    RewardWeights,
    resolve_weights,
)


def sample_vector() -> RewardVector:
    return RewardVector(
        content_value=2.0,
        commerce_value=5.0,
        user_value=1.0,
        ecosystem_value=0.5,
        risk_cost=3.0,
    )


def test_risk_cost_is_subtracted_not_added():
    vector = sample_vector()
    weights = RewardWeights(content_value=1.0, commerce_value=1.0, user_value=1.0, ecosystem_value=1.0, risk_cost=1.0)

    assert vector.scalarize(weights) == pytest.approx(2.0 + 5.0 + 1.0 + 0.5 - 3.0)


def test_risk_cost_must_be_non_negative():
    with pytest.raises(ValueError, match="risk_cost must be non-negative"):
        RewardVector(risk_cost=-1.0)


def test_weights_must_be_non_negative():
    with pytest.raises(ValueError, match="must be non-negative"):
        RewardWeights(commerce_value=-1.0)


def test_gmv_first_and_retention_first_disagree_on_the_same_vector():
    vector = sample_vector()
    assert vector.scalarize("gmv_first") > vector.scalarize("retention_first")


def test_all_named_profiles_resolve():
    expected = {"content_first", "balanced", "gmv_first", "retention_first", "clearance_campaign"}
    assert set(SCALARIZATIONS) == expected
    for name in expected:
        assert isinstance(resolve_weights(name), RewardWeights)


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown scalarization profile"):
        resolve_weights("maximize_everything")


def test_contributions_sum_to_the_scalar_and_stay_auditable():
    vector = sample_vector()
    contributions = vector.contributions("gmv_first")

    assert sum(contributions.values()) == pytest.approx(vector.scalarize("gmv_first"))
    assert contributions["risk_cost"] < 0.0
    assert set(contributions) == set(vector.to_dict())


def test_vectors_add_component_wise():
    total = RewardVector(content_value=1.0, risk_cost=2.0) + RewardVector(commerce_value=3.0, risk_cost=0.5)

    assert total.content_value == 1.0
    assert total.commerce_value == 3.0
    assert total.risk_cost == 2.5


def test_vector_round_trips():
    vector = sample_vector()
    assert RewardVector.from_dict(vector.to_dict()) == vector
    with pytest.raises(ValueError, match="Unknown RewardVector fields"):
        RewardVector.from_dict({"gmv": 1.0})


def test_reward_config_round_trips_with_and_without_custom_weights():
    default_config = RewardConfig(profile="content_first")
    assert RewardConfig.from_dict(default_config.to_dict()) == default_config
    assert default_config.resolved_weights() == SCALARIZATIONS["content_first"]

    custom = RewardConfig(profile="custom", weights=RewardWeights(commerce_value=2.0))
    assert RewardConfig.from_dict(custom.to_dict()) == custom
    assert custom.scalarize(sample_vector()) == pytest.approx(
        1.0 * 2.0 + 2.0 * 5.0 + 1.0 * 1.0 + 1.0 * 0.5 - 1.0 * 3.0
    )


def test_reward_config_rejects_unknown_profile_without_weights():
    with pytest.raises(ValueError, match="Unknown scalarization profile"):
        RewardConfig(profile="custom")


def test_feed_decision_serve_helper_and_round_trip():
    decision = FeedDecision.serve(
        "X2",
        reason="Viewer is mid-session and the anchor matches the clip.",
        reasoning_summary={"observation_facts": ["ad_fatigue=0.2"], "confidence": 0.7},
    )

    assert decision.action_type == "serve_exposure"
    assert FeedDecision.from_dict(decision.to_dict()) == decision


def test_feed_decision_rejects_unsupported_actions_until_the_protocol_grows():
    with pytest.raises(ValueError, match="Unknown feed action_type"):
        FeedDecision(action_type="rank_exposures", exposure_id="X2")
    with pytest.raises(ValueError, match="requires an exposure_id"):
        FeedDecision.serve("")


def test_feed_decision_summary_is_copied_not_aliased():
    summary = {"observation_facts": ["a"]}
    decision = FeedDecision.serve("X1", reasoning_summary=summary)
    decision.to_dict()["reasoning_summary"]["observation_facts"].append("b")

    assert decision.reasoning_summary["observation_facts"] == ["a"]
    assert summary["observation_facts"] == ["a"]
