import urllib.error

from scripts.generate_llm_trajectories import run_llm_episode, run_llm_episode_with_retries, summarize
from videoshop.agents.openai_compatible import LLMHTTPError
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
    assert episode["steps"][0]["tool_call_trace"][-1]["status"] == "accepted"
    assert episode["steps"][0]["violations"] == []
    assert "metrics" in episode


def test_summarize_llm_episodes_counts_scenarios():
    episode = {
        "scenario_id": "coupon_valid",
        "steps": [
            {
                "reward": 1.0,
                "reasoning_summary": {
                    "observation_facts": ["test"],
                    "evidence_used": [],
                    "decision_rule": "test",
                },
                "agent_step": {"metadata": {}},
                "tool_results": [],
            }
        ],
        "total_reward": 1.0,
        "outcome": "no_purchase",
        "metrics": {
            "constraint_violations": 0,
            "fake_coupon_count": 0,
            "unsupported_explanation_count": 0,
            "category_mismatch_count": 0,
            "return_or_refund_count": 0,
            "gross_purchase": False,
            "net_purchase": False,
        },
    }

    report = summarize([episode], model="Qwen3.8-27B")

    assert report["model"] == "Qwen3.8-27B"
    assert report["episodes"] == 1
    assert report["scenarios"] == 1
    assert report["avg_reward"] == 1.0
    assert report["reasoning_summary_coverage"] == 1.0
    assert report["valid_reasoning_summary_coverage"] == 1.0


def test_episode_retry_reuses_seed_after_transient_infrastructure_error(monkeypatch):
    scenarios = build_toy_scenarios()[:1]
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), scenarios=scenarios, max_steps=1, seed=42)
    attempts = []

    def fake_run(env, agent, scenario_id, episode_id, *, seed=None):
        del env, agent, scenario_id
        attempts.append(seed)
        if len(attempts) == 1:
            raise TimeoutError("temporary timeout")
        return {
            "episode_id": episode_id,
            "scenario_id": "coupon_valid",
            "steps": [],
            "total_reward": 0.0,
            "outcome": "no_purchase",
            "metrics": {},
        }

    monkeypatch.setattr("scripts.generate_llm_trajectories.run_llm_episode", fake_run)
    episode = run_llm_episode_with_retries(
        env,
        object(),
        scenarios[0].scenario_id,
        "llm:retry:001",
        seed=77,
        retries=2,
    )

    assert attempts == [77, 77]
    assert episode["run_metadata"]["attempts"] == 2
    assert episode["run_metadata"]["attempt_errors"][0]["category"] == "infrastructure"


def test_failed_infrastructure_episode_is_not_a_constraint_violation(monkeypatch):
    scenarios = build_toy_scenarios()[:1]
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), scenarios=scenarios, max_steps=1, seed=42)

    def always_fail(*args, **kwargs):
        del args, kwargs
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr("scripts.generate_llm_trajectories.run_llm_episode", always_fail)
    episode = run_llm_episode_with_retries(
        env,
        object(),
        scenarios[0].scenario_id,
        "llm:retry:002",
        seed=78,
        retries=1,
    )

    assert episode["outcome"] == "failed"
    assert episode["error"]["category"] == "infrastructure"
    assert episode["metrics"]["constraint_violations"] == 0
    assert episode["run_metadata"]["attempts"] == 2


def test_retryable_provider_error_retries_http_429(monkeypatch):
    scenarios = build_toy_scenarios()[:1]
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), scenarios=scenarios, max_steps=1, seed=42)
    attempts = 0

    def rate_limited_then_ok(env, agent, scenario_id, episode_id, *, seed=None):
        nonlocal attempts
        del env, agent, scenario_id, seed
        attempts += 1
        if attempts == 1:
            raise LLMHTTPError(429, "rate limited")
        return {
            "episode_id": episode_id,
            "scenario_id": "coupon_valid",
            "steps": [],
            "total_reward": 0.0,
            "outcome": "no_purchase",
            "metrics": {},
        }

    monkeypatch.setattr("scripts.generate_llm_trajectories.run_llm_episode", rate_limited_then_ok)
    episode = run_llm_episode_with_retries(
        env,
        object(),
        scenarios[0].scenario_id,
        "llm:retry:003",
        seed=79,
        retries=1,
    )

    assert attempts == 2
    assert episode["run_metadata"]["attempt_errors"][0]["category"] == "provider"
    assert episode["run_metadata"]["attempt_errors"][0]["status_code"] == 429
