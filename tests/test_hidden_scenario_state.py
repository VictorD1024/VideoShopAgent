from __future__ import annotations

import json

import pytest

from videoshop.feed import (
    FEED_SCHEMA_VERSION,
    BaseScores,
    Eligibility,
    ExposureCandidate,
    RewardConfig,
    ScenarioSpec,
    find_forbidden_keys,
    find_leaks,
)


def build_spec(**overrides) -> ScenarioSpec:
    payload = {
        "scenario_id": "smoke_0001",
        "instruction": "Serve the next item in this session.",
        "public_context": {
            "user_summary": {"budget_level": "low", "ad_fatigue": 0.2},
            "session_summary": {"step": 0, "recent_skips": 1},
        },
        "hidden_world_state": {
            "true_purchase_intent": 0.71,
            "click_noise": 0.04,
        },
        "candidate_exposures": [
            ExposureCandidate(
                exposure_id="X1",
                video_id="V1",
                source_type="organic",
                base_scores=BaseScores(expected_watch_time=14.0, skip_probability=0.2),
            ),
            ExposureCandidate(
                exposure_id="X2",
                video_id="V2",
                source_type="seller",
                product_id="P1",
                treatment="product_anchor",
                base_scores=BaseScores(purchase_probability=0.08, expected_net_gmv=21.0),
            ),
            ExposureCandidate(
                exposure_id="X3",
                video_id="V3",
                source_type="ad",
                product_id="P2",
                treatment="coupon",
                eligibility=Eligibility(in_stock=False),
            ),
        ],
        "oracle": {"gold_exposure_id": "X2", "scenario_type": "anchor_preferred"},
        "reward_config": RewardConfig(profile="balanced"),
    }
    payload.update(overrides)
    return ScenarioSpec(**payload)


def test_agent_view_hides_oracle_and_hidden_world_state():
    view = build_spec().agent_view()
    serialized = json.dumps(view)

    assert "oracle" not in view
    assert "hidden_world_state" not in view
    assert "X2" in serialized  # the candidate itself is visible
    assert "gold_exposure_id" not in serialized
    assert "anchor_preferred" not in serialized
    assert "true_purchase_intent" not in serialized
    assert "0.71" not in serialized


def test_agent_view_exposes_candidates_and_objective_profile():
    view = build_spec().agent_view()

    assert view["schema_version"] == FEED_SCHEMA_VERSION
    assert view["objective_profile"] == "balanced"
    assert [candidate["exposure_id"] for candidate in view["candidate_exposures"]] == ["X1", "X2", "X3"]
    assert view["candidate_exposures"][0]["base_scores"]["expected_watch_time"] == 14.0


def test_agent_view_is_a_deep_copy():
    spec = build_spec()
    view = spec.agent_view()
    view["public_context"]["user_summary"]["ad_fatigue"] = 0.99

    assert spec.public_context["user_summary"]["ad_fatigue"] == 0.2


def test_public_context_cannot_carry_private_keys():
    with pytest.raises(ValueError, match="leaks private keys into public_context"):
        build_spec(public_context={"scenario_type": "fake_coupon_trap"})

    with pytest.raises(ValueError, match="leaks private keys into public_context"):
        build_spec(public_context={"expected_behaviors": ["call_get_coupon"]})


@pytest.mark.parametrize(
    "public_context,expected_path",
    [
        ({"user": {"true_purchase_intent": 0.9}}, "user.true_purchase_intent"),
        ({"a": {"b": {"c": {"gold_exposure_id": "X2"}}}}, "a.b.c.gold_exposure_id"),
        ({"items": [{"ok": 1}, {"oracle": {}}]}, "items[1].oracle"),
        ({"rows": [[{"scenario_type": "trap"}]]}, "rows[0][0].scenario_type"),
    ],
)
def test_nested_private_keys_are_caught(public_context, expected_path):
    assert find_forbidden_keys(public_context) == [expected_path]
    with pytest.raises(ValueError, match="leaks private keys into public_context"):
        build_spec(public_context=public_context)


def test_forbidden_key_scan_reports_every_hit_with_its_path():
    payload = {"oracle": 1, "deep": {"answer": 2, "fine": {"label": 3}}}
    assert sorted(find_forbidden_keys(payload)) == ["deep.answer", "deep.fine.label", "oracle"]


def test_clean_nested_context_passes():
    assert find_forbidden_keys({"user": {"ad_fatigue": 0.2}, "items": [{"price": 1.0}]}) == []


@pytest.mark.parametrize(
    "instruction",
    [
        "Use a valid coupon for a relevant kitchen product.",
        "Delay recommendations when ad fatigue and recent skips are high.",
        "Prefer a cheaper substitute when the leading product is expensive.",
        "Call show_coupon only when inventory allows.",
        "The correct answer is the organic clip.",
    ],
)
def test_leaky_instructions_are_rejected(instruction):
    with pytest.raises(ValueError, match="instruction leaks the answer"):
        build_spec(instruction=instruction)


def test_neutral_instruction_is_accepted():
    spec = build_spec(instruction="Choose what this viewer sees next in the for-you feed.")
    assert find_leaks(spec.instruction) == []


def test_find_leaks_reports_every_marker():
    leaks = find_leaks("First call get_coupon, then show_coupon for the gold product.")
    assert set(leaks) == {"get_coupon", "show_coupon", "gold"}


def test_spec_requires_candidates_and_unique_ids():
    with pytest.raises(ValueError, match="requires at least one candidate exposure"):
        build_spec(candidate_exposures=[])

    duplicate = ExposureCandidate(exposure_id="X1", video_id="V9", source_type="organic")
    with pytest.raises(ValueError, match="duplicate exposure_ids"):
        build_spec(candidate_exposures=[duplicate, duplicate])


def test_candidate_partitions():
    spec = build_spec()

    assert [candidate.exposure_id for candidate in spec.organic_candidates()] == ["X1"]
    assert [candidate.exposure_id for candidate in spec.commercial_candidates()] == ["X2", "X3"]
    assert [candidate.exposure_id for candidate in spec.eligible_candidates()] == ["X1", "X2"]
    assert spec.candidate("X3").eligibility.blocking_reasons() == ("in_stock",)
    assert spec.candidate("missing") is None


def test_spec_round_trips_with_private_state():
    spec = build_spec()
    restored = ScenarioSpec.from_dict(spec.to_dict())

    assert restored.to_dict() == spec.to_dict()
    assert restored.oracle == {"gold_exposure_id": "X2", "scenario_type": "anchor_preferred"}
    assert restored.hidden_world_state["true_purchase_intent"] == 0.71
    assert restored.candidate_exposures == spec.candidate_exposures


# --- smoke set audit -------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_scenarios():
    from videoshop.feed import build_smoke_scenarios

    return build_smoke_scenarios(count=100, seed=7)


def test_smoke_set_has_one_hundred_unique_scenarios(smoke_scenarios):
    assert len(smoke_scenarios) == 100
    assert len({scenario.scenario_id for scenario in smoke_scenarios}) == 100


def test_no_smoke_instruction_leaks_the_answer(smoke_scenarios):
    offenders = {
        scenario.scenario_id: find_leaks(scenario.instruction)
        for scenario in smoke_scenarios
        if find_leaks(scenario.instruction)
    }
    assert offenders == {}


def test_no_smoke_agent_view_leaks_private_state(smoke_scenarios):
    for scenario in smoke_scenarios:
        serialized = json.dumps(scenario.agent_view())
        assert "gold_exposure_id" not in serialized
        assert "expected_scalar_by_exposure" not in serialized
        assert "session_family" not in serialized
        assert "purchase_intent" not in serialized
        assert scenario.oracle["session_family"] not in serialized


def test_every_smoke_scenario_offers_organic_and_commercial_choices(smoke_scenarios):
    for scenario in smoke_scenarios:
        assert scenario.organic_candidates(), scenario.scenario_id
        assert scenario.commercial_candidates(), scenario.scenario_id
        servable_organic = [
            candidate for candidate in scenario.eligible_candidates() if candidate.is_organic
        ]
        assert servable_organic, f"{scenario.scenario_id} has no servable organic option"


def test_smoke_set_covers_every_source_type_and_objective_profile(smoke_scenarios):
    sources = {
        candidate.source_type for scenario in smoke_scenarios for candidate in scenario.candidate_exposures
    }
    profiles = {scenario.reward_config.profile for scenario in smoke_scenarios}
    families = {scenario.oracle["session_family"] for scenario in smoke_scenarios}

    assert sources == {"organic", "seller", "affiliate", "ad"}
    assert profiles == {"balanced", "content_first", "gmv_first", "retention_first", "clearance_campaign"}
    assert len(families) >= 5


def test_smoke_set_contains_candidates_the_environment_must_intercept(smoke_scenarios):
    blocked = [
        candidate
        for scenario in smoke_scenarios
        for candidate in scenario.candidate_exposures
        if not candidate.eligibility.is_eligible
    ]
    assert blocked, "the smoke set must exercise hard-constraint interception"


def test_oracle_gold_is_always_an_eligible_candidate(smoke_scenarios):
    for scenario in smoke_scenarios:
        gold = scenario.oracle["gold_exposure_id"]
        assert gold is not None, scenario.scenario_id
        candidate = scenario.candidate(gold)
        assert candidate is not None and candidate.eligibility.is_eligible


def test_smoke_set_is_deterministic_for_a_seed():
    from videoshop.feed import build_smoke_scenarios

    first = build_smoke_scenarios(count=10, seed=7)
    second = build_smoke_scenarios(count=10, seed=7)
    other = build_smoke_scenarios(count=10, seed=8)

    assert [spec.to_dict() for spec in first] == [spec.to_dict() for spec in second]
    assert [spec.to_dict() for spec in first] != [spec.to_dict() for spec in other]


def test_base_scores_are_not_the_hidden_truth(smoke_scenarios):
    """The agent sees a noisy ranker view, never the response parameters."""
    mismatches = 0
    total = 0
    for scenario in smoke_scenarios[:20]:
        truth = scenario.hidden_world_state["exposure_truth"]
        for candidate in scenario.candidate_exposures:
            total += 1
            if candidate.base_scores.skip_probability != truth[candidate.exposure_id]["skip_probability"]:
                mismatches += 1
    assert mismatches / total > 0.9
