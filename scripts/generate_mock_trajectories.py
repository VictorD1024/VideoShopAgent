from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from videoshop.data.mock import build_mock_states, build_mock_videos
from videoshop.policies import RuleBasedPolicy
from videoshop.simulator.env import VideoShopEnv
from videoshop.simulator.evaluator import summarize_episodes
from videoshop.simulator.rollout import run_batch
from videoshop.simulator.trajectory import write_jsonl


def generate(episodes: int, seed: int) -> tuple[list[dict], dict]:
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=8, seed=seed)
    policy = RuleBasedPolicy()
    rows = run_batch(env, policy, episodes)
    summary = summarize_episodes(rows, policy_name="RuleBasedPolicy")
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/trajectories/mock_trajectories.jsonl")
    parser.add_argument("--report", default="outputs/reports/mock_eval_summary.json")
    args = parser.parse_args()

    rows, summary = generate(args.episodes, args.seed)
    write_jsonl(args.out, rows)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
