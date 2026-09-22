from __future__ import annotations

import copy
import json
from dataclasses import fields

import pytest

from videoshop.feed import (
    FEED_SCHEMA_VERSION,
    FeedControlEnv,
    FeedDecision,
    build_default_provider,
    build_feed_policy,
    build_smoke_scenarios,
)
from videoshop.feed.candidate_provider import CandidateProviderConfig


@pytest.fixture(scope="module")
def provider():
    return build_default_provider()


@pytest.fixture(scope="module")
def scenarios(provider):
    # The scenarios must come from the provider that will also serve later steps,
    # otherwise the episode mixes two catalogs that share product ids.
    return build_smoke_scenarios(count=12, seed=7, provider=provider)


@pytest.fixture()
def env(provider):
    return FeedControlEnv(provider, max_steps=6, seed=123)


def run_episode(env, scenario, policy, seed=123):
    observation = env.reset(scenario, seed=seed)
    infos = []
    done = False
    while not done:
        observation, _vector, terminated, truncated, info = env.step(policy.act(observation))
        infos.append(info)
        done = terminated or truncated
    return observation, infos


def test_every_step_offers_both_organic_and_commercial(env, scenarios):
    policy = build_feed_policy("rule_based")
    for scenario in scenarios[:6]:
        observation = env.reset(scenario, seed=11)
        done = False
        while not done:
            sources = {candidate["source_type"] for candidate in observation.candidate_exposures}
            assert "organic" in sources, "organic content must always be a real option"
            assert sources & {"seller", "affiliate", "ad"}, "commercial content must always be offered"
            organic_servable = [
                candidate
                for candidate in observation.candidate_exposures
                if candidate["source_type"] == "organic" and _eligible(candidate)
            ]
            assert organic_servable, "doing nothing commercial must stay a legal move"
            observation, _reward, terminated, truncated, _info = env.step(policy.act(observation))
            done = terminated or truncated


def test_provider_emits_all_four_source_types(scenarios):
    seen = set()
    for scenario in scenarios:
        seen.update(candidate.source_type for candidate in scenario.candidate_exposures)
    assert seen == {"organic", "seller", "affiliate", "ad"}


def test_provider_config_rejects_a_mix_that_could_drop_organic():
    with pytest.raises(ValueError, match="must cover min_organic \\+ min_commercial"):
        CandidateProviderConfig(candidates_per_step=3, min_organic=2, min_commercial=3)


def test_agent_decides_the_next_exposure(env, scenarios):
    observation = env.reset(scenarios[0], seed=5)
    target = observation.candidate_exposures[-1]["exposure_id"]

    _observation, _vector, _terminated, _truncated, info = env.step(FeedDecision.serve(target))

    served = info["record"]["served_exposure"]
    assert served is not None
    assert served["exposure_id"] == target


def test_ineligible_candidates_are_intercepted_and_never_served(env, scenarios):
    scenario = next(
        spec
        for spec in scenarios
        if any(not candidate.eligibility.is_eligible for candidate in spec.candidate_exposures)
    )
    blocked_candidate = next(
        candidate for candidate in scenario.candidate_exposures if not candidate.eligibility.is_eligible
    )

    env.reset(scenario, seed=5)
    _observation, vector, terminated, _truncated, info = env.step(
        FeedDecision.serve(blocked_candidate.exposure_id)
    )

    assert info["blocked"]
    assert info["block_reasons"] == list(blocked_candidate.eligibility.blocking_reasons())
    assert info["record"]["served_exposure"] is None
    assert info["record"]["user_response"] is None
    assert vector.risk_cost > 0.0
    assert not terminated, "an illegal choice costs reward but does not end the session"


def test_unknown_exposure_id_is_blocked(env, scenarios):
    env.reset(scenarios[0], seed=5)
    _observation, _vector, _terminated, _truncated, info = env.step(FeedDecision.serve("not_a_real_exposure"))

    assert info["block_reasons"] == ["unknown_exposure_id"]
    assert info["record"]["served_exposure"] is None


def test_agent_cannot_rebind_a_product_to_another_video(env, scenarios):
    """The decision protocol carries only an exposure_id, so bindings are facts."""
    decision_fields = {field.name for field in fields(FeedDecision)}
    assert "product_id" not in decision_fields
    assert "video_id" not in decision_fields

    observation = env.reset(scenarios[0], seed=5)
    commercial = next(
        candidate
        for candidate in observation.candidate_exposures
        if candidate["source_type"] != "organic" and _eligible(candidate)
    )
    _observation, _vector, _terminated, _truncated, info = env.step(
        FeedDecision.serve(commercial["exposure_id"])
    )

    served = info["record"]["served_exposure"]
    assert (served["video_id"], served["product_id"]) == (commercial["video_id"], commercial["product_id"])


def test_observation_never_exposes_oracle_or_hidden_truth(env, scenarios):
    scenario = scenarios[0]
    observation = env.reset(scenario, seed=5)
    serialized = json.dumps(observation.to_dict())

    assert "gold_exposure_id" not in serialized
    assert "expected_scalar_by_exposure" not in serialized
    assert "session_family" not in serialized
    assert "purchase_intent" not in serialized
    assert "exposure_truth" not in serialized
    assert "hidden_world_state" not in serialized
    assert scenario.oracle["session_family"] not in serialized


def test_observation_stays_clean_after_stepping(env, scenarios):
    policy = build_feed_policy("rule_based")
    observation, _infos = run_episode(env, scenarios[1], policy)
    serialized = json.dumps(observation.to_dict())

    assert "gold" not in serialized
    assert "annoyance" not in serialized
    assert "click_probability" not in serialized or "product_click_probability" in serialized


def test_last_step_feedback_is_observable_but_limited(env, scenarios):
    observation = env.reset(scenarios[0], seed=5)
    assert observation.last_step is None

    target = observation.candidate_exposures[0]["exposure_id"]
    observation, _vector, _terminated, _truncated, _info = env.step(FeedDecision.serve(target))

    assert observation.last_step is not None
    assert observation.last_step["served_exposure_id"] in {target, None}
    assert "reward_vector" not in observation.last_step
    assert "reward_terms" not in observation.last_step


def test_same_seed_replays_identically(provider, scenarios):
    def rollout():
        env = FeedControlEnv(provider, max_steps=6, seed=77)
        policy = build_feed_policy("random", seed=4)
        observation = env.reset(scenarios[2], seed=77)
        done = False
        while not done:
            observation, _vector, terminated, truncated, _info = env.step(policy.act(observation))
            done = terminated or truncated
        return env.trajectory(episode_id="replay", policy="random")

    assert rollout() == rollout()


def test_trajectory_carries_the_v2_schema_version_and_audit_terms(env, scenarios):
    policy = build_feed_policy("rule_based")
    run_episode(env, scenarios[3], policy)
    trajectory = env.trajectory(episode_id="E1", policy="rule_based")

    assert trajectory["schema_version"] == FEED_SCHEMA_VERSION == "v2"
    assert trajectory["objective_profile"] == scenarios[3].reward_config.profile
    assert set(trajectory["total_reward_vector"]) == {
        "content_value",
        "commerce_value",
        "user_value",
        "ecosystem_value",
        "risk_cost",
    }

    served_step = next(step for step in trajectory["steps"] if not step["blocked"])
    assert served_step["schema_version"] == "v2"
    assert set(served_step["reward_terms"]) == set(served_step["reward_vector"])
    for component, terms in served_step["reward_terms"].items():
        assert pytest.approx(sum(terms.values()), abs=1e-6) == served_step["reward_vector"][component]


def test_total_reward_vector_is_the_sum_of_steps(env, scenarios):
    policy = build_feed_policy("rule_based")
    run_episode(env, scenarios[4], policy)
    trajectory = env.trajectory(episode_id="E1", policy="rule_based")

    for component, total in trajectory["total_reward_vector"].items():
        expected = sum(step["reward_vector"][component] for step in trajectory["steps"])
        assert pytest.approx(total, abs=1e-6) == expected


def test_episode_truncates_at_max_steps(provider, scenarios):
    env = FeedControlEnv(provider, max_steps=3, seed=9)
    observation = env.reset(scenarios[0], seed=9)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _vector, terminated, truncated, _info = env.step(
            FeedDecision.serve(observation.candidate_exposures[0]["exposure_id"])
        )
        steps += 1
    assert steps <= 3


def test_step_before_reset_is_rejected(provider):
    env = FeedControlEnv(provider)
    with pytest.raises(RuntimeError, match="Call reset\\(\\) before step\\(\\)"):
        env.step(FeedDecision.serve("X1"))


def test_rule_based_and_greedy_policies_never_pick_a_blocked_candidate(env, scenarios):
    for name in ("rule_based", "greedy_gmv"):
        policy = build_feed_policy(name)
        for scenario in scenarios[:6]:
            _observation, infos = run_episode(env, scenario, policy)
            assert not any(info["blocked"] for info in infos), f"{name} served an ineligible exposure"


def test_rule_based_beats_random_on_the_smoke_set(provider, scenarios):
    """Only the random margin is asserted here; 12 scenarios cannot resolve anything finer.

    The oracle ceiling is checked in test_feed_benchmark, where the sample is large
    enough for the ordering to be stable.
    """

    def average_reward(policy_name: str) -> float:
        env = FeedControlEnv(provider, max_steps=6, seed=123)
        totals = []
        for scenario in scenarios:
            policy = build_feed_policy(policy_name, seed=5)
            run_episode(env, scenario, policy)
            totals.append(env.trajectory(episode_id=scenario.scenario_id, policy=policy_name)["total_scalar_reward"])
        return sum(totals) / len(totals)

    assert average_reward("rule_based") > average_reward("random")


def test_greedy_gmv_over_exposes_commerce_relative_to_rule_based(provider, scenarios):
    def commercial_share(policy_name: str) -> float:
        env = FeedControlEnv(provider, max_steps=6, seed=123)
        commercial = total = 0
        for scenario in scenarios:
            policy = build_feed_policy(policy_name, seed=5)
            run_episode(env, scenario, policy)
            for step in env.trajectory(episode_id=scenario.scenario_id, policy=policy_name)["steps"]:
                served = step["served_exposure"]
                if served is None:
                    continue
                total += 1
                commercial += int(served["source_type"] != "organic")
        return commercial / max(total, 1)

    assert commercial_share("greedy_gmv") > commercial_share("rule_based")


@pytest.mark.parametrize("coupon_available", [True, False])
def test_only_a_coupon_that_redeems_is_booked_as_used(coupon_available):
    """`used_coupons` is public, so booking a trap there hands the answer over."""

    from videoshop.feed.candidate_provider import ExposureTruth
    from videoshop.feed.schemas import ExposureCandidate
    from videoshop.feed.user_model import FeedUserModel, FeedUserResponse
    from videoshop.simulator.schemas import SessionState, UserProfile

    candidate = ExposureCandidate(
        exposure_id="e0",
        video_id="V1",
        source_type="ad",
        product_id="P1",
        treatment="coupon",
    )
    truth = ExposureTruth(
        watch_seconds=8.0,
        skip_probability=0.2,
        click_probability=0.5,
        purchase_probability=0.1,
        refund_probability=0.0,
        annoyance=0.1,
        coupon_available=coupon_available,
    )
    session = SessionState(step=0)
    user = UserProfile(
        user_id="U1",
        country="US",
        budget_level="mid",
        style_preferences=["minimal"],
        category_interests={"home": 0.5},
        price_sensitivity=0.4,
        risk_sensitivity=0.3,
        ad_fatigue=0.2,
        purchase_intent=0.5,
    )

    FeedUserModel().apply(
        FeedUserResponse(watched_seconds=8.0, clicked=True),
        candidate,
        truth,
        user,
        session,
        "home",
    )

    assert session.recent_clicks == ["P1"]
    assert session.used_coupons == (["P1"] if coupon_available else [])


def test_the_session_summary_counts_aired_commercial_supply(provider, scenarios):
    env = FeedControlEnv(provider, max_steps=6, seed=123)
    policy = build_feed_policy("greedy_gmv")

    for scenario in scenarios:
        observation = env.reset(scenario, seed=123)
        assert observation.public_context["session_summary"]["commercial_exposures"] == 0
        done = False
        while not done:
            observation, _vector, terminated, truncated, _info = env.step(policy.act(observation))
            aired = sum(
                1
                for step in env.records
                if (step.get("served_exposure") or {}).get("source_type", "organic") != "organic"
            )
            assert observation.public_context["session_summary"]["commercial_exposures"] == aired
            done = terminated or truncated


def test_the_density_cap_reads_commercial_supply_not_products():
    from videoshop.feed.control_env import FeedObservation
    from videoshop.feed.policies import RuleBasedFeedPolicy

    policy = RuleBasedFeedPolicy()
    observation = FeedObservation(
        schema_version=FEED_SCHEMA_VERSION,
        scenario_id="s",
        instruction="",
        step=4,
        max_steps=6,
        objective_profile="balanced",
        public_context={
            "session_summary": {
                "step": 4,
                # Three ads aired; only one of them carried a product through.
                "commercial_exposures": 3,
                    "exposed_products": ["P1"],
                }
            },
            candidate_exposures=[],
        )

    assert policy._commercial_density(observation) == pytest.approx(0.75)


def test_scenario_requires_hidden_truth_for_every_candidate(scenarios):
    from videoshop.feed.control_env import _batch_from_scenario

    broken = copy.deepcopy(scenarios[0])
    broken.hidden_world_state["exposure_truth"].pop(broken.candidate_exposures[0].exposure_id)

    with pytest.raises(ValueError, match="missing hidden exposure truth"):
        _batch_from_scenario(broken)


def _eligible(candidate: dict) -> bool:
    eligibility = candidate["eligibility"]
    flags = ("in_stock", "coupon_valid", "policy_compliant", "risk_within_limit", "audience_allowed")
    return all(eligibility[flag] for flag in flags) and not eligibility["extra_blocks"]
