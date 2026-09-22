from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from videoshop.agents.llm_loop import FunctionCallingAgent, ScriptedToolCallingClient
from videoshop.feed.benchmark import build_split
from videoshop.feed.candidate_provider import CandidateProviderConfig
from videoshop.feed.composite_env import FeedInterventionEnv
from videoshop.feed.control_env import FeedControlEnv
from videoshop.feed.policies import (
    AnchorInterventionPolicy,
    TrustingInterventionPolicy,
    build_feed_policy,
    build_intervention_agent,
    build_intervention_policy,
)
from videoshop.feed.schemas import FEED_SCHEMA_VERSION
from videoshop.policies import RuleBasedPolicy
from videoshop.simulator.schemas import AgentAction
from videoshop.simulator.tools import find_substitute

# A coupon-heavy world so the stale-coupon trap is actually exercised.
COUPON_HEAVY = CandidateProviderConfig(coupon_treatment_rate=0.6)


@pytest.fixture(scope="module")
def split():
    return build_split("smoke", count=40, provider_config=COUPON_HEAVY)


def run_episodes(env, scenarios, feed_policy_name: str = "rule_based"):
    feed_policy = build_feed_policy(feed_policy_name)
    records = []
    rewards = []
    for scenario in scenarios:
        observation = env.reset(scenario, seed=42)
        done = False
        total = 0.0
        while not done:
            observation, vector, terminated, truncated, info = env.step(feed_policy.act(observation))
            total += scenario.reward_config.scalarize(vector)
            records.append(info["record"])
            done = terminated or truncated
        rewards.append(total)
    return records, rewards


def interventions(records):
    return [record["intervention"] for record in records if record.get("intervention", {}).get("ran")]


# --- the layers are actually connected ------------------------------------


def test_commercial_exposures_run_the_real_tool_runtime(split):
    env = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios)

    ran = interventions(records)
    assert ran, "no commercial exposure reached the intervention layer"

    tools = Counter(result["tool"] for item in ran for result in item["tool_results"])
    assert tools, "the intervention layer executed no tools"
    assert set(tools) <= {
        "retrieve_candidates",
        "rank_products",
        "get_coupon",
        "find_substitute",
        "explain_recommendation",
    }


def test_organic_exposures_skip_the_intervention_layer(split):
    env = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios)

    for record in records:
        if record["blocked"]:
            continue
        served_organic = record["served_exposure"]["source_type"] == "organic"
        offered_organic = record["offered_exposure"]["source_type"] == "organic"
        if offered_organic:
            assert record["intervention"]["ran"] is False
            assert served_organic


def test_feed_only_env_serves_the_offered_treatment_verbatim(split):
    env = FeedControlEnv(split.provider, max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios)

    for record in records:
        if record["blocked"]:
            continue
        assert record["offered_exposure"] == record["served_exposure"]
        assert "intervention" not in record


# --- the intervention layer changes outcomes ------------------------------


def test_verifying_policy_degrades_stale_coupons_instead_of_faking_them(split):
    env = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios)
    ran = interventions(records)

    offered_coupons = [item for item in ran if item["offered_treatment"] == "coupon"]
    downgraded = [item for item in offered_coupons if item["realized_treatment"] != "coupon"]

    assert offered_coupons, "the world offered no coupons"
    assert downgraded, "no stale coupon was caught; the trap is not reachable"
    assert all(not item["violations"] for item in ran)
    assert all("get_coupon" in {r["tool"] for r in item["tool_results"]} for item in offered_coupons)


def test_a_verifying_intervention_layer_costs_nothing_on_the_coupon_axis(split):
    """The intervention layer is a liability surface, not a bonus.

    A stale coupon buys nothing in either environment, so verifying and degrading lands
    back on the feed-only outcome; only ungrounded claims cost. The two are not bit
    identical because a grounded `switch_to_substitute` does legitimately change the
    product, so this asserts closeness rather than equality.
    """
    feed_only = FeedControlEnv(split.provider, max_steps=6, seed=42)
    composite = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    trusting = FeedInterventionEnv(split.provider, TrustingInterventionPolicy(), max_steps=6, seed=42)

    for feed_policy in ("rule_based", "greedy_gmv"):
        _, baseline = run_episodes(feed_only, split.scenarios, feed_policy)
        _, verified = run_episodes(composite, split.scenarios, feed_policy)
        _, unverified = run_episodes(trusting, split.scenarios, feed_policy)

        baseline_mean = sum(baseline) / len(baseline)
        assert sum(verified) / len(verified) == pytest.approx(baseline_mean, abs=0.05), feed_policy
        assert sum(unverified) / len(unverified) < baseline_mean - 0.05, feed_policy


def test_the_anchor_policy_does_not_drop_a_sellable_product(split):
    """Declining cannot help once the clip is airing: it pays the annoyance for nothing."""
    env = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios)

    for item in interventions(records):
        if item["action"]["action_type"] == "delay_recommendation":
            assert "unsellable" in item["action"]["reason"] or "unavailable" in item["action"]["reason"]


def test_trusting_policy_walks_into_the_trap_and_pays_for_it(split):
    verifying = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    trusting = FeedInterventionEnv(split.provider, TrustingInterventionPolicy(), max_steps=6, seed=42)

    verifying_records, verifying_rewards = run_episodes(verifying, split.scenarios)
    trusting_records, trusting_rewards = run_episodes(trusting, split.scenarios)

    trusting_violations = Counter(v for item in interventions(trusting_records) for v in item["violations"])
    verifying_violations = Counter(v for item in interventions(verifying_records) for v in item["violations"])

    assert trusting_violations["fake_coupon"] > 0
    assert verifying_violations["fake_coupon"] == 0
    assert sum(trusting_rewards) < sum(verifying_rewards), "verification must pay off"


def test_violations_are_charged_to_risk_cost_and_stay_auditable(split):
    env = FeedInterventionEnv(split.provider, TrustingInterventionPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios)

    penalised = [
        record
        for record in records
        if record.get("intervention", {}).get("violations")
    ]
    assert penalised

    for record in penalised:
        risk_terms = record["reward_terms"]["risk_cost"]
        charged = {key for key in risk_terms if key.startswith("intervention:")}
        assert charged, "an intervention violation was recorded but never priced"
        assert record["reward_vector"]["risk_cost"] >= sum(risk_terms[key] for key in charged)


# --- the feed layer's binding still holds ---------------------------------


class OffPoolPolicy:
    """Tries to rebind the exposure to a product the tools never surfaced."""

    def act(self, observation, state, exposure):
        del observation, state, exposure
        return AgentAction(
            action_type="show_product_card",
            product_id="P999999",
            reason="Rebind to something that is not on offer.",
        )


def test_intervention_cannot_rebind_to_a_product_outside_its_candidate_set(split):
    env = FeedInterventionEnv(split.provider, OffPoolPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios)
    ran = interventions(records)

    assert ran
    for item in ran:
        assert "product_outside_intervention_candidates" in item["violations"]
    for record in records:
        if record.get("intervention", {}).get("ran"):
            assert record["served_exposure"]["product_id"] != "P999999"


def test_intervention_never_changes_the_video_the_feed_layer_chose(split):
    env = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios)

    for record in records:
        if record["blocked"]:
            continue
        assert record["served_exposure"]["video_id"] == record["offered_exposure"]["video_id"]
        assert record["served_exposure"]["exposure_id"] == record["offered_exposure"]["exposure_id"]


def test_served_product_stays_in_the_video_category(split):
    env = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios)
    videos = {video.video_id: video for video in split.provider.videos}
    products = {product.product_id: product for product in split.provider.products}

    for record in records:
        served = record["served_exposure"]
        if record["blocked"] or not served["product_id"]:
            continue
        assert products[served["product_id"]].category == videos[served["video_id"]].category


# --- protocol guarantees survive composition ------------------------------


def test_composite_trajectories_keep_the_v2_schema(split):
    env = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    observation = env.reset(split.scenarios[0], seed=42)
    feed_policy = build_feed_policy("rule_based")
    done = False
    while not done:
        observation, _, terminated, truncated, _ = env.step(feed_policy.act(observation))
        done = terminated or truncated

    trajectory = env.trajectory(episode_id="E1", policy="rule_based+anchor_rule_based")
    assert trajectory["schema_version"] == FEED_SCHEMA_VERSION
    for step in trajectory["steps"]:
        assert step["schema_version"] == FEED_SCHEMA_VERSION
        if step.get("intervention", {}).get("ran"):
            assert step["intervention"]["action"]["action_type"]
            assert "tool_results" in step["intervention"]


def test_observations_never_expose_the_coupon_trap(split):
    env = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    feed_policy = build_feed_policy("rule_based")

    for scenario in split.scenarios[:10]:
        observation = env.reset(scenario, seed=42)
        done = False
        while not done:
            payload = repr(observation.to_dict())
            assert "coupon_available" not in payload
            assert "coupon_trap" not in payload
            observation, _, terminated, truncated, _ = env.step(feed_policy.act(observation))
            done = terminated or truncated


def test_stale_and_live_coupons_look_identical_to_the_intervention_agent(split):
    """The real test of the redaction: not "the field is gone" but "the two cases match".

    If any visible feature separated a trap from a real offer, a model could learn the
    shortcut and `get_coupon` would stop being the only authority.
    """

    from videoshop.feed.observation import build_intervention_observation
    from videoshop.simulator.tools import coupon_is_available

    seen: list[tuple[tuple, bool]] = []

    class Spy(AnchorInterventionPolicy):
        def act(self, observation, state, exposure):
            if exposure.treatment == "coupon":
                anchor = next(
                    item for item in state.candidate_products if item.product_id == exposure.product_id
                )
                view = build_intervention_observation(state, exposure)
                product = next(
                    item
                    for item in view.visible_candidates
                    if item["product_id"] == exposure.product_id
                )
                fingerprint = (
                    product["has_coupon"],
                    view.exposure["offered_treatment"],
                    view.exposure["offered_eligibility"]["coupon_valid"],
                    tuple(sorted(view.commerce_signals)),
                    exposure.product_id in view.session_summary["used_coupons"],
                )
                seen.append((fingerprint, coupon_is_available(state, anchor)))
            return super().act(observation, state, exposure)

    env = FeedInterventionEnv(split.provider, Spy(), max_steps=6, seed=42)
    run_episodes(env, split.scenarios, feed_policy_name="greedy_gmv")

    stale = {fingerprint for fingerprint, available in seen if not available}
    live = {fingerprint for fingerprint, available in seen if available}
    assert stale and live, "the run did not contain both a trap and a real coupon"
    assert stale == live


def test_composite_env_is_deterministic_under_a_fixed_seed(split):
    def trajectory():
        env = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=99)
        records, rewards = run_episodes(env, split.scenarios[:8])
        return [record["served_exposure"] for record in records], rewards

    assert trajectory() == trajectory()


# --- policy arity dispatch -------------------------------------------------


def test_v1_state_policies_are_accepted(split):
    env = FeedInterventionEnv(split.provider, RuleBasedPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios[:10])

    assert interventions(records), "the v1 policy never ran"


def test_tool_calling_agents_are_accepted(split):
    client = ScriptedToolCallingClient(
        [
            [
                {"name": "rank_products", "arguments": {"limit": 3}},
                {
                    "name": "submit_final_action",
                    "arguments": {
                        "action_type": "delay_recommendation",
                        "product_id": None,
                        "reason": "Hold off this step.",
                    },
                },
            ]
        ]
    )
    agent = FunctionCallingAgent(client)
    env = FeedInterventionEnv(split.provider, agent, max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios[:5])

    ran = interventions(records)
    assert ran
    assert all(item["action"]["action_type"] == "delay_recommendation" for item in ran)
    assert all(item["realized_treatment"] == "none" for item in ran)


def test_a_tool_calling_turn_is_recorded_in_full(split):
    """The record has to be trainable, not just scorable.

    Squashing an `AgentStep` down to its final action throws away the tool requests,
    the stated reasoning and the provider trace, which is most of the supervision.
    """

    client = ScriptedToolCallingClient(
        [
            [
                {"name": "rank_products", "arguments": {}},
                {
                    "name": "submit_final_action",
                    "arguments": {
                        "action_type": "delay_recommendation",
                        "product_id": None,
                        "reason": "Hold off this step.",
                        "reasoning_summary": {
                            "observation_facts": ["ad_fatigue is high"],
                            "evidence_used": ["rank_products"],
                            "decision_rule": "Delay when the viewer is tired.",
                            "confidence": 0.8,
                        },
                    },
                },
            ]
        ]
    )
    env = FeedInterventionEnv(split.provider, FunctionCallingAgent(client), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios[:5])

    ran = interventions(records)
    assert ran

    # The scripted client only has one turn; after that it falls back, so the scripted
    # detail is checked on the first turn and the trace on every turn.
    first = ran[0]["agent_step"]
    assert [request["tool"] for request in first["tool_requests"]] == ["rank_products"]
    assert first["final_action"]["reasoning_summary"]["decision_rule"] == (
        "Delay when the viewer is tired."
    )
    assert first["metadata"]["tool_call_trace"][0]["round"] == 1
    assert first["metadata"]["tool_call_trace"][0]["raw_calls"]
    assert all(item["agent_step"]["metadata"]["tool_call_trace"] for item in ran)


def test_the_prompt_the_agent_saw_is_kept_with_its_answer(split):
    env = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios[:8])

    for item in interventions(records):
        observation = item["observation"]
        assert observation["exposure"]["exposure_id"]
        assert "coupon_inventory" not in observation["commerce_signals"]


def test_a_rule_policy_records_no_agent_step(split):
    """`agent_step` means "there was an LLM turn here", so rule baselines leave it empty."""

    env = FeedInterventionEnv(split.provider, AnchorInterventionPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios[:5])

    ran = interventions(records)
    assert ran
    assert all(item["agent_step"] is None for item in ran)
    assert all(item["action"]["action_type"] for item in ran)


def test_repair_rounds_survive_into_the_record(split):
    """A rejected first attempt is training signal too, so it must not be dropped."""

    client = ScriptedToolCallingClient(
        [
            [{"name": "get_coupon", "arguments": {"product_id": "NOT_A_PRODUCT"}}],
            [
                {
                    "name": "submit_final_action",
                    "arguments": {
                        "action_type": "delay_recommendation",
                        "product_id": None,
                        "reason": "Recovered.",
                    },
                }
            ],
        ]
    )
    env = FeedInterventionEnv(split.provider, FunctionCallingAgent(client), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios[:3])

    ran = interventions(records)
    assert ran
    metadata = ran[0]["agent_step"]["metadata"]
    assert metadata["tool_call_trace"][0]["status"] == "repair"
    assert "NOT_A_PRODUCT" in metadata["tool_call_trace"][0]["error"]
    assert "invalid_tool_arguments" in metadata["agent_repair_warnings"]
    assert "You are the intervention layer" in metadata["system_prompt"]
    assert "coupon_inventory" not in metadata["system_prompt"]


class RecordingClient(ScriptedToolCallingClient):
    def __init__(self, turns):
        super().__init__(turns)
        self.messages = []

    def complete(self, messages, tools):
        self.messages.append([dict(message) for message in messages])
        return super().complete(messages, tools)


def test_a_bare_function_calling_agent_is_briefed_for_the_composite_env(split):
    """The obvious construction must not keep the v1 'pick a product' prompt."""

    client = RecordingClient(
        [
            [
                {
                    "name": "submit_final_action",
                    "arguments": {
                        "action_type": "delay_recommendation",
                        "product_id": None,
                        "reason": "Hold off.",
                    },
                }
            ]
        ]
    )
    env = FeedInterventionEnv(split.provider, FunctionCallingAgent(client), max_steps=6, seed=42)
    run_episodes(env, split.scenarios[:3])

    assert client.messages, "the agent was never asked to act"
    first = client.messages[0]
    system = first[0]["content"]
    user = next(message["content"] for message in reversed(first) if message["role"] == "user")

    assert "intervention layer" in system
    assert "Decide whether to recommend a product" not in system
    assert "anchor_product_id" in user
    assert "offered_treatment" in user
    assert "coupon_inventory" not in user
    assert "coupon_thresholds" not in user
    assert "coupon_expiry_steps" not in user


def test_build_intervention_agent_is_what_callers_should_use(split):
    from videoshop.agents.prompt import build_system_prompt
    from videoshop.feed.observation import build_intervention_system_prompt

    agent = build_intervention_agent(RecordingClient([]))
    assert agent.system_prompt == build_intervention_system_prompt()
    assert agent.system_prompt != build_system_prompt()
    assert agent.few_shots is not None
    assert "anchor_product_id" in agent.few_shots[0]["content"]


def test_declining_the_treatment_drops_the_product_but_keeps_the_supply_type(split):
    """Declining must not launder an ad into organic; that would dodge fatigue and risk."""

    class DecliningPolicy:
        def act(self, observation, state, exposure):
            del observation, state, exposure
            return AgentAction(action_type="delay_recommendation", reason="Not now.")

    env = FeedInterventionEnv(split.provider, DecliningPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios[:10])

    declined = 0
    for record in records:
        if not record.get("intervention", {}).get("ran"):
            continue
        offered, served = record["offered_exposure"], record["served_exposure"]
        assert offered["source_type"] != "organic"
        assert served["source_type"] == offered["source_type"]
        assert served["treatment"] == "none"
        assert served["product_id"] is None
        declined += 1

    assert declined


def test_a_declined_ad_still_counts_against_commercial_density(split):
    """The density cap used to read `exposed_products`, which a declined ad never enters."""

    class DecliningPolicy:
        def act(self, observation, state, exposure):
            del observation, state, exposure
            return AgentAction(action_type="delay_recommendation", reason="Not now.")

    env = FeedInterventionEnv(split.provider, DecliningPolicy(), max_steps=6, seed=42)
    feed_policy = build_feed_policy("greedy_gmv")

    saw_declined_ad = False
    for scenario in split.scenarios[:10]:
        observation = env.reset(scenario, seed=42)
        done = False
        while not done:
            observation, _, terminated, truncated, info = env.step(feed_policy.act(observation))
            served = info["record"]["served_exposure"]
            summary = observation.public_context["session_summary"]
            if served and served["source_type"] != "organic" and served["product_id"] is None:
                saw_declined_ad = True
                assert summary["commercial_exposures"] > len(summary["exposed_products"])
            done = terminated or truncated

    assert saw_declined_ad, "the declining policy never faced a commercial exposure"


def test_declining_never_converts_commercial_supply_into_organic(split):
    class DecliningPolicy:
        def act(self, observation, state, exposure):
            del observation, state, exposure
            return AgentAction(action_type="delay_recommendation", reason="Not now.")

    env = FeedInterventionEnv(split.provider, DecliningPolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios[:20])
    served = [r for r in records if not r["blocked"]]

    offered_organic = sum(r["offered_exposure"]["source_type"] == "organic" for r in served)
    served_organic = sum(r["served_exposure"]["source_type"] == "organic" for r in served)

    assert served_organic == offered_organic
    assert offered_organic < len(served), "the run must contain commercial exposures to be meaningful"


# --- evidence is bound to the product the action commits to ---------------


class UngroundedSubstitutePolicy:
    """Calls find_substitute, then switches to a different product than it returned."""

    def act(self, observation, state, exposure):
        del observation
        anchor = next(p for p in state.candidate_products if p.product_id == exposure.product_id)
        substitute, call = find_substitute(state, anchor)
        other = next(
            (
                p
                for p in state.candidate_products
                if p.product_id not in {anchor.product_id, getattr(substitute, "product_id", None)}
            ),
            None,
        )
        if other is None:
            return AgentAction(action_type="delay_recommendation", reason="Nothing to abuse.")
        return AgentAction(
            action_type="switch_to_substitute",
            product_id=other.product_id,
            reason="Quote a real tool result, then switch to something else.",
            tool_calls=[call],
            evidence={"substitute": call.output},
        )


def test_quoting_a_real_substitute_then_switching_elsewhere_is_caught(split):
    env = FeedInterventionEnv(split.provider, UngroundedSubstitutePolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios[:20])
    switches = [
        item
        for item in interventions(records)
        if item["action"]["action_type"] == "switch_to_substitute"
    ]

    assert switches, "the policy never attempted a switch"
    for item in switches:
        assert "substitute_evidence_is_for_a_different_product" in item["violations"]
        assert "intervention_substituted_without_substitute_tool" in item["violations"]


def test_a_grounded_substitute_is_not_penalised(split):
    class GroundedSubstitutePolicy:
        def act(self, observation, state, exposure):
            del observation
            anchor = next(p for p in state.candidate_products if p.product_id == exposure.product_id)
            substitute, call = find_substitute(state, anchor)
            if substitute is None:
                return AgentAction(action_type="delay_recommendation", reason="No substitute.")
            return AgentAction(
                action_type="switch_to_substitute",
                product_id=substitute.product_id,
                reason="Take exactly what the tool returned.",
                tool_calls=[call],
                evidence={"substitute": call.output},
            )

    env = FeedInterventionEnv(split.provider, GroundedSubstitutePolicy(), max_steps=6, seed=42)
    records, _ = run_episodes(env, split.scenarios[:20])
    switches = [
        item
        for item in interventions(records)
        if item["action"]["action_type"] == "switch_to_substitute"
    ]

    assert switches
    assert all(not item["violations"] for item in switches)


# --- determinism across processes -----------------------------------------


def test_coupon_inventory_does_not_depend_on_the_interpreter_hash_seed():
    script = (
        "import sys; sys.path.insert(0, 'src');"
        "from videoshop.feed.composite_env import _stable_hash;"
        "print([_stable_hash(p) % 3 for p in ('P0010001','P0010002','P0010003','P0010004')])"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
        for seed in ("0", "1", "2", "12345")
    }

    assert len(runs) == 1, f"coupon inventory varies with PYTHONHASHSEED: {runs}"


def test_intervention_policy_registry():
    assert isinstance(build_intervention_policy("anchor_rule_based"), AnchorInterventionPolicy)
    assert isinstance(build_intervention_policy("trusting"), TrustingInterventionPolicy)
    with pytest.raises(ValueError, match="Unknown intervention policy"):
        build_intervention_policy("nope")
