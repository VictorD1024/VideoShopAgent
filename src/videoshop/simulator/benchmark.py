from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict

from videoshop.data.mock import build_mock_states, build_mock_videos
from videoshop.data.scenarios import build_toy_scenarios
from videoshop.data.synthetic import SyntheticCommerceConfig, build_synthetic_scenarios, build_synthetic_states, build_synthetic_videos
from videoshop.policies import RandomPolicy, RuleBasedPolicy
from videoshop.simulator.env import VideoShopEnv
from videoshop.simulator.metrics import evaluate_episode


def build_policy(name: str, seed: int):
    if name == "rule_based":
        return RuleBasedPolicy()
    if name == "random":
        return RandomPolicy(random.Random(seed))
    raise ValueError(f"Unsupported policy: {name}")


def run_scenario_episode(env: VideoShopEnv, policy, scenario_id: str, episode_id: str) -> dict:
    state = env.reset(scenario_id=scenario_id)
    steps = []
    total_reward = 0.0
    done = False
    outcome = "no_purchase"
    scenario = env.scenarios[scenario_id]

    while not done:
        state_snapshot = asdict(state)
        action = policy.act(state)
        next_state, response, reward, done, update = env.step(action)
        total_reward += reward
        if response.purchased:
            outcome = "purchase"

        steps.append(
            {
                "t": next_state.session_state.step,
                "scenario_id": scenario_id,
                "state": state_snapshot,
                "tool_calls": [asdict(call) for call in action.tool_calls],
                "action": asdict(action),
                "user_response": asdict(response),
                "reward": reward,
                "state_update": asdict(update),
            }
        )
        state = next_state

    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "objective": scenario.objective,
        "expected_behaviors": list(scenario.expected_behaviors),
        "user_profile": asdict(state.user_profile),
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
    }


def summarize_benchmark(episodes: list[dict], policy_name: str) -> dict:
    per_episode = [evaluate_episode(episode) for episode in episodes]
    total_steps = sum(metrics.get("steps", 0) for metrics in per_episode)
    total_reward = sum(metrics.get("total_reward", 0.0) for metrics in per_episode)
    purchases = sum(metrics.get("outcome") == "purchase" for metrics in per_episode)
    violations = sum(metrics.get("constraint_violations", 0) for metrics in per_episode)

    return {
        "policy": policy_name,
        "episodes": len(episodes),
        "scenarios": len({episode["scenario_id"] for episode in episodes}),
        "avg_reward": total_reward / len(episodes) if episodes else 0.0,
        "avg_steps": total_steps / len(episodes) if episodes else 0.0,
        "purchase_rate": purchases / len(episodes) if episodes else 0.0,
        "constraint_violations": violations,
        "fake_coupon_count": sum(metrics.get("fake_coupon_count", 0) for metrics in per_episode),
        "unsupported_explanation_count": sum(metrics.get("unsupported_explanation_count", 0) for metrics in per_episode),
        "category_mismatch_count": sum(metrics.get("category_mismatch_count", 0) for metrics in per_episode),
        "return_or_refund_count": sum(metrics.get("return_or_refund_count", 0) for metrics in per_episode),
    }


def run_benchmark(
    policy_name: str,
    episodes_per_scenario: int,
    seed: int,
    *,
    split: str = "toy",
    synthetic_config: SyntheticCommerceConfig | None = None,
) -> tuple[list[dict], dict]:
    scenarios, initial_states, videos = _build_split(split, seed, synthetic_config)
    env = VideoShopEnv(
        initial_states,
        videos,
        scenarios=scenarios,
        max_steps=8,
        seed=seed,
    )
    policy = build_policy(policy_name, seed)
    episodes = []

    for scenario in scenarios:
        for index in range(episodes_per_scenario):
            episode_id = f"{scenario.scenario_id}:{index + 1:03d}"
            episodes.append(run_scenario_episode(env, policy, scenario.scenario_id, episode_id))

    summary = summarize_benchmark(episodes, policy_name=policy_name)
    summary["split"] = split
    summary["scenario_type_counts"] = dict(Counter(_scenario_type(episode["scenario_id"]) for episode in episodes))
    if split == "synthetic":
        config = synthetic_config or SyntheticCommerceConfig(seed=seed)
        summary["synthetic_config"] = {
            **asdict(config),
            "catalog_size": config.category_count * config.products_per_category,
            "video_count": config.category_count * config.videos_per_category,
        }
    return episodes, summary


def _build_split(
    split: str,
    seed: int,
    synthetic_config: SyntheticCommerceConfig | None,
):
    if split == "toy":
        return build_toy_scenarios(), build_mock_states(), build_mock_videos()
    if split == "synthetic":
        config = synthetic_config or SyntheticCommerceConfig(seed=seed)
        return build_synthetic_scenarios(config), build_synthetic_states(config), build_synthetic_videos(config)
    raise ValueError(f"Unsupported benchmark split: {split}")


def _scenario_type(scenario_id: str) -> str:
    if scenario_id.startswith("synthetic_"):
        parts = scenario_id.split("_")
        return "_".join(parts[1:-1]) if len(parts) > 2 else "synthetic"
    return scenario_id
