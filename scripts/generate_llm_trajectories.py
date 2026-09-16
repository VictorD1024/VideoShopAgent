from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from videoshop.agents import FunctionCallingAgent, OpenAICompatibleToolClient
from videoshop.data.mock import build_mock_states, build_mock_videos
from videoshop.data.scenarios import build_toy_scenarios
from videoshop.data.synthetic import SyntheticCommerceConfig, build_synthetic_scenarios, build_synthetic_states, build_synthetic_videos
from videoshop.simulator.env import VideoShopEnv
from videoshop.simulator.metrics import evaluate_episode


def run_llm_episode(env: VideoShopEnv, agent: FunctionCallingAgent, scenario_id: str, episode_id: str) -> dict:
    observation, info = env.reset_agent(scenario_id=scenario_id)
    steps = []
    total_reward = 0.0
    terminated = False
    truncated = False
    outcome = "no_purchase"
    scenario = env.scenarios[scenario_id]

    while not (terminated or truncated):
        observation_snapshot = asdict(observation)
        state_snapshot = asdict(info["state"])
        agent_step = agent.act(observation, info["state"])
        final_action_snapshot = asdict(agent_step.final_action) if agent_step.final_action else {}
        observation, reward, terminated, truncated, step_info = env.step_agent(agent_step)
        total_reward += reward
        if step_info["user_response"].purchased:
            outcome = "purchase"

        steps.append(
            {
                "t": observation.session_summary["step"],
                "scenario_id": scenario_id,
                "observation": observation_snapshot,
                "state": state_snapshot,
                "agent_step": asdict(agent_step),
                "reasoning_summary": final_action_snapshot.get("reasoning_summary", {}),
                "tool_results": [asdict(result) for result in step_info["tool_results"]],
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "violations": list(step_info["violations"]),
                "user_response": asdict(step_info["user_response"]),
                "state_update": asdict(step_info["state_update"]),
            }
        )
        info = {"state": step_info["state"]}

    episode = {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "objective": scenario.objective,
        "expected_behaviors": list(scenario.expected_behaviors),
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
    }
    # Compatibility for existing metrics helpers that expect action/user_response fields.
    episode["metrics"] = evaluate_llm_episode(episode)
    return episode


def evaluate_llm_episode(episode: dict) -> dict:
    compatible_steps = []
    for step in episode["steps"]:
        final_action = step["agent_step"].get("final_action") or {}
        compatible_steps.append(
            {
                "action": {
                    "action_type": final_action.get("action_type"),
                    "product_id": final_action.get("product_id"),
                    "reason": final_action.get("reason", ""),
                    "tool_calls": step["agent_step"].get("tool_requests", []),
                    "evidence": _evidence_from_tool_results(final_action.get("action_type"), step.get("tool_results", [])),
                },
                "user_response": step["user_response"],
                "reward": step["reward"],
            }
        )
    return evaluate_episode(
        {
            "episode_id": episode["episode_id"],
            "total_reward": episode["total_reward"],
            "outcome": episode["outcome"],
            "steps": compatible_steps,
        }
    )


def _evidence_from_tool_results(action_type: str | None, tool_results: list[dict]) -> dict:
    outputs = {result.get("tool"): result.get("output", {}) for result in tool_results if result.get("success", True)}
    if action_type == "show_coupon" and "get_coupon" in outputs:
        return {"coupon": outputs["get_coupon"]}
    if action_type == "switch_to_substitute" and "find_substitute" in outputs:
        return {"substitute": outputs["find_substitute"]}
    if action_type == "show_explanation" and "explain_recommendation" in outputs:
        return outputs["explain_recommendation"].get("evidence", {})
    return {}


def summarize(episodes: list[dict], model: str) -> dict:
    metrics = [episode["metrics"] for episode in episodes]
    successful = [episode for episode in episodes if episode.get("outcome") != "failed"]
    steps = [step for episode in successful for step in episode.get("steps", [])]
    reasoning_steps = [step for step in steps if step.get("reasoning_summary")]
    return {
        "model": model,
        "episodes": len(episodes),
        "successful_episodes": len(successful),
        "failed_episodes": len(episodes) - len(successful),
        "scenarios": len({episode["scenario_id"] for episode in episodes}),
        "avg_reward": sum(episode["total_reward"] for episode in successful) / len(successful) if successful else 0.0,
        "avg_steps": sum(len(episode["steps"]) for episode in successful) / len(successful) if successful else 0.0,
        "purchase_rate": sum(episode["outcome"] == "purchase" for episode in successful) / len(successful) if successful else 0.0,
        "constraint_violations": sum(metric.get("constraint_violations", 0) for metric in metrics),
        "fake_coupon_count": sum(metric.get("fake_coupon_count", 0) for metric in metrics),
        "unsupported_explanation_count": sum(metric.get("unsupported_explanation_count", 0) for metric in metrics),
        "category_mismatch_count": sum(metric.get("category_mismatch_count", 0) for metric in metrics),
        "return_or_refund_count": sum(metric.get("return_or_refund_count", 0) for metric in metrics),
        "reasoning_summary_steps": len(reasoning_steps),
        "reasoning_summary_coverage": len(reasoning_steps) / len(steps) if steps else 0.0,
    }


def _build_split(split: str, synthetic_config: SyntheticCommerceConfig):
    if split == "toy":
        return build_toy_scenarios(), build_mock_states(), build_mock_videos()
    if split == "synthetic":
        return (
            build_synthetic_scenarios(synthetic_config),
            build_synthetic_states(synthetic_config),
            build_synthetic_videos(synthetic_config),
        )
    raise ValueError(f"Unsupported split: {split}")


def _scenario_type_counts(episodes: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in episodes:
        scenario_type = _scenario_type(episode.get("scenario_id", ""))
        counts[scenario_type] = counts.get(scenario_type, 0) + 1
    return counts


def _scenario_type(scenario_id: str) -> str:
    if scenario_id.startswith("synthetic_"):
        parts = scenario_id.split("_")
        return "_".join(parts[1:-1]) if len(parts) > 2 else "synthetic"
    return scenario_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM tool-calling trajectories for VideoShopEnv.")
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--split", choices=["toy", "synthetic"], default="toy")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/llm_trajectories/qwen_tool_calling.jsonl")
    parser.add_argument("--report", default="outputs/llm_trajectories/qwen_tool_calling_summary.json")
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-env-steps", type=int, default=8)
    parser.add_argument("--observation-candidate-limit", type=int, default=12)
    parser.add_argument("--no-few-shots", action="store_true")
    parser.add_argument("--synthetic-categories", type=int, default=12)
    parser.add_argument("--synthetic-products-per-category", type=int, default=80)
    parser.add_argument("--synthetic-users", type=int, default=40)
    parser.add_argument("--synthetic-videos-per-category", type=int, default=8)
    parser.add_argument("--synthetic-scenarios", type=int, default=50)
    parser.add_argument("--synthetic-candidate-pool-size", type=int, default=32)
    parser.add_argument("--synthetic-coupon-coverage", type=float, default=0.22)
    args = parser.parse_args()

    model = os.environ.get("VIDEOSHOP_LLM_MODEL", "Qwen3.8-27B")
    synthetic_config = SyntheticCommerceConfig(
        seed=args.seed,
        category_count=args.synthetic_categories,
        products_per_category=args.synthetic_products_per_category,
        users=args.synthetic_users,
        videos_per_category=args.synthetic_videos_per_category,
        scenario_count=args.synthetic_scenarios,
        candidate_pool_size=args.synthetic_candidate_pool_size,
        coupon_coverage=args.synthetic_coupon_coverage,
    )
    scenarios, initial_states, videos = _build_split(args.split, synthetic_config)
    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]

    env = VideoShopEnv(
        initial_states,
        videos,
        scenarios=scenarios,
        max_steps=args.max_env_steps,
        seed=args.seed,
        observation_candidate_limit=args.observation_candidate_limit,
    )
    client = OpenAICompatibleToolClient(model=model, timeout=args.timeout, max_tokens=args.max_tokens)
    agent = FunctionCallingAgent(client, max_tool_rounds=args.max_tool_rounds, include_few_shots=not args.no_few_shots)

    episodes = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            for index in range(args.episodes_per_scenario):
                episode_id = f"llm:{scenario.scenario_id}:{index + 1:03d}"
                print(f"[llm] running {episode_id}", flush=True)
                try:
                    episode = run_llm_episode(env, agent, scenario.scenario_id, episode_id)
                    print(f"[llm] saved {episode_id} reward={episode['total_reward']} steps={len(episode['steps'])}", flush=True)
                except Exception as exc:
                    episode = {
                        "episode_id": episode_id,
                        "scenario_id": scenario.scenario_id,
                        "objective": scenario.objective,
                        "expected_behaviors": list(scenario.expected_behaviors),
                        "steps": [],
                        "total_reward": 0.0,
                        "outcome": "failed",
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "metrics": {"constraint_violations": 1},
                    }
                    print(f"[llm] failed {episode_id}: {type(exc).__name__}: {exc}", flush=True)
                episodes.append(episode)
                handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
                handle.flush()

    report = summarize(episodes, model=model)
    report["split"] = args.split
    report["scenario_type_counts"] = _scenario_type_counts(episodes)
    if args.split == "synthetic":
        report["synthetic_config"] = {
            **asdict(synthetic_config),
            "catalog_size": synthetic_config.category_count * synthetic_config.products_per_category,
            "video_count": synthetic_config.category_count * synthetic_config.videos_per_category,
            "observation_candidate_limit": args.observation_candidate_limit,
        }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
