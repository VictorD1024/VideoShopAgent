import pytest

from videoshop.data.mock import build_mock_states, build_mock_videos
from videoshop.simulator.env import VideoShopEnv
from videoshop.simulator.metrics import evaluate_episode
from videoshop.simulator.scenario import VideoShopScenario
from videoshop.simulator.schemas import AgentStep, FinalAction, ToolCallRequest


def test_reset_agent_returns_observation_and_info_state():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=2, seed=42)

    observation, info = env.reset_agent(seed=7)

    assert observation.task
    assert observation.current_video["video_id"]
    assert "retrieve_candidates" in observation.allowed_tools
    assert info["state"].session_state.step == 0


def test_observation_candidate_limit_keeps_full_state_candidates():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=2, seed=42, observation_candidate_limit=3)

    observation, info = env.reset_agent(seed=7)

    assert len(observation.visible_candidates) == 3
    assert len(info["state"].candidate_products) > 3


def test_step_agent_executes_tools_and_builds_grounded_coupon_evidence():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=2, seed=42)
    observation, _ = env.reset_agent(seed=42)
    product_id = observation.visible_candidates[0]["product_id"]

    step = AgentStep(
        tool_requests=[ToolCallRequest("get_coupon", {"product_id": product_id})],
        final_action=FinalAction("show_coupon", product_id=product_id, reason="Use available coupon if valid."),
    )
    next_observation, reward, terminated, truncated, info = env.step_agent(step)

    assert next_observation.session_summary["step"] == 1
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["tool_results"][0].tool == "get_coupon"
    assert info["violations"] == []


def test_step_agent_splits_truncated_from_terminated():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=1, seed=42)
    env.reset_agent(seed=42)

    _, _, terminated, truncated, _ = env.step_agent(
        AgentStep(final_action=FinalAction("delay_recommendation", reason="Wait for stronger buying intent."))
    )

    assert terminated is False
    assert truncated is True


def test_step_agent_rejects_product_action_without_product_before_state_transition():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=2, seed=42)
    env.reset_agent(seed=42)

    with pytest.raises(ValueError, match="requires a product_id"):
        env.step_agent(AgentStep(final_action=FinalAction("show_product_card", product_id=None, reason="Invalid.")))

    assert env.state.session_state.step == 0


def test_video_shop_scenario_can_seed_a_fixed_episode():
    states = build_mock_states()
    videos = build_mock_videos()
    scenario = VideoShopScenario.from_state("coupon_case", states[0], videos, objective="Validate coupon behavior.")
    env = VideoShopEnv(states, videos, scenarios=[scenario], max_steps=2, seed=42)

    observation, info = env.reset_agent(scenario_id="coupon_case")

    assert observation.task == "Validate coupon behavior."
    assert info["scenario_id"] == "coupon_case"
    assert observation.user_summary["user_id"] == states[0].user_profile.user_id


def test_evaluate_episode_reports_constraint_counts():
    episode = {
        "episode_id": "E000001",
        "total_reward": -5.0,
        "outcome": "no_purchase",
        "steps": [
            {
                "action": {"action_type": "show_coupon", "evidence": {"coupon": {"available": False}}},
                "user_response": {"clicked": False, "added_to_cart": False, "purchased": False, "interrupted": True, "irrelevant_recommendation": True},
            }
        ],
    }

    metrics = evaluate_episode(episode)

    assert metrics["fake_coupon_count"] == 1
    assert metrics["category_mismatch_count"] == 1
    assert metrics["constraint_violations"] == 2


def test_evaluate_episode_separates_gross_and_net_purchase():
    episode = {
        "episode_id": "E000002",
        "total_reward": -4.0,
        "outcome": "returned_or_refunded",
        "steps": [
            {
                "action": {"action_type": "show_product_card", "evidence": {}},
                "user_response": {
                    "clicked": True,
                    "added_to_cart": True,
                    "purchased": True,
                    "returned_or_refunded": True,
                    "interrupted": False,
                    "irrelevant_recommendation": False,
                },
            }
        ],
    }

    metrics = evaluate_episode(episode)

    assert metrics["gross_purchase"] is True
    assert metrics["net_purchase"] is False
    assert metrics["gross_purchase_steps"] == 1
    assert metrics["net_purchase_steps"] == 0
