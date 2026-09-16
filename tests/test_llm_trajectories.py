from scripts.generate_llm_trajectories import run_llm_episode, summarize
from videoshop.agents.llm_loop import FunctionCallingAgent, ScriptedToolCallingClient
from videoshop.agents.tool_specs import FINAL_ACTION_TOOL
from videoshop.data.mock import build_mock_states, build_mock_videos
from videoshop.data.scenarios import build_toy_scenarios
from videoshop.simulator.env import VideoShopEnv


def test_run_llm_episode_records_tool_calling_trace():
    scenarios = build_toy_scenarios()[:1]
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), scenarios=scenarios, max_steps=2, seed=42)
    client = ScriptedToolCallingClient(
        [
            [{"name": "get_coupon", "arguments": {"product_id": "P001"}}],
            [
                {
                    "name": FINAL_ACTION_TOOL,
                    "arguments": {
                        "action_type": "show_coupon",
                        "product_id": "P001",
                        "reason": "Coupon checked.",
                        "reasoning_summary": {
                            "observation_facts": ["P001 is visible"],
                            "evidence_used": ["get_coupon returned a valid coupon"],
                            "candidate_comparison": ["P001 has a valid coupon"],
                            "rejected_options": [],
                            "decision_rule": "Show valid coupon after checking coupon tool.",
                            "confidence": 0.9,
                        },
                    },
                }
            ],
        ]
    )
    agent = FunctionCallingAgent(client, include_few_shots=False)

    episode = run_llm_episode(env, agent, scenarios[0].scenario_id, "llm:coupon_valid:001")

    assert episode["episode_id"] == "llm:coupon_valid:001"
    assert episode["steps"][0]["agent_step"]["tool_requests"][0]["tool"] == "get_coupon"
    assert episode["steps"][0]["tool_results"][0]["tool"] == "get_coupon"
    assert episode["steps"][0]["reasoning_summary"]["decision_rule"] == "Show valid coupon after checking coupon tool."
    assert episode["steps"][0]["violations"] == []
    assert "metrics" in episode


def test_summarize_llm_episodes_counts_scenarios():
    episode = {
        "scenario_id": "coupon_valid",
        "steps": [{"reward": 1.0, "reasoning_summary": {"decision_rule": "test"}}],
        "total_reward": 1.0,
        "outcome": "no_purchase",
        "metrics": {"constraint_violations": 0, "fake_coupon_count": 0, "unsupported_explanation_count": 0, "category_mismatch_count": 0, "return_or_refund_count": 0},
    }

    report = summarize([episode], model="Qwen3.8-27B")

    assert report["model"] == "Qwen3.8-27B"
    assert report["episodes"] == 1
    assert report["scenarios"] == 1
    assert report["avg_reward"] == 1.0
    assert report["reasoning_summary_coverage"] == 1.0
