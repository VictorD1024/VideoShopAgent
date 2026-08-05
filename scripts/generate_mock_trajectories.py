from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from videoshop.env import VideoShopEnv
from videoshop.policies import RuleBasedPolicy
from videoshop.schemas import EnvState, Product, SessionState, UserProfile, VideoContext
from videoshop.trajectory import write_jsonl


def build_mock_states() -> list[EnvState]:
    products = [
        Product("P001", "Acrylic Desk Organizer", "home organization", 24.99, 4.6, 430, 0.18, tags=["minimal", "desk"]),
        Product("P002", "Cable Organizer Box", "home organization", 15.99, 4.4, 210, 0.12, is_clearance=True, tags=["clean", "desk"]),
        Product("P003", "Portable Camping Lantern", "camping", 29.99, 4.7, 120, 0.10, is_high_margin=True, tags=["outdoor"]),
        Product("P004", "Makeup Brush Organizer", "beauty", 19.99, 4.5, 260, 0.22, tags=["aesthetic"]),
    ]
    users = [
        UserProfile("U001", "US", "medium", ["minimal", "clean"], ["home organization"], 0.45, 0.2, 0.35),
        UserProfile("U002", "US", "low", ["outdoor"], ["camping"], 0.8, 0.35, 0.4),
        UserProfile("U003", "UK", "medium", ["aesthetic"], ["beauty"], 0.55, 0.25, 0.3),
    ]
    videos = [
        VideoContext("V001", "Small desk makeover for tiny apartment", "home office desk setup", ["organizer", "lamp", "keyboard"], ["minimal", "clean"], "home lifestyle"),
        VideoContext("V002", "Weekend camping gear setup", "camping outdoor setup", ["lantern", "chair", "tent"], ["outdoor"], "travel creator"),
        VideoContext("V003", "Minimal vanity organization", "beauty desk organization", ["brush", "mirror", "organizer"], ["aesthetic"], "beauty creator"),
    ]

    states: list[EnvState] = []
    for user in users:
        for video in videos:
            states.append(
                EnvState(
                    user_profile=user,
                    session_state=SessionState(step=0, recent_watch_categories=[video.scene.split()[0]]),
                    current_video=video,
                    candidate_products=products,
                )
            )
    return states


def generate(episodes: int, seed: int) -> tuple[list[dict], dict]:
    random.seed(seed)
    env = VideoShopEnv(build_mock_states(), max_steps=8, seed=seed)
    policy = RuleBasedPolicy()
    rows: list[dict] = []
    total_rewards: list[float] = []
    purchases = 0

    for episode_idx in range(episodes):
        state = env.reset()
        steps = []
        total_reward = 0.0
        outcome = "no_purchase"

        done = False
        while not done:
            action = policy.act(state)
            next_state, response, reward, done = env.step(action)
            total_reward += reward
            if response.purchased:
                purchases += 1
                outcome = "purchase"
            steps.append(
                {
                    "t": next_state.session_state.step,
                    "state": asdict(state),
                    "action": asdict(action),
                    "user_response": asdict(response),
                    "reward": reward,
                }
            )
            state = next_state

        total_rewards.append(total_reward)
        rows.append(
            {
                "episode_id": f"E{episode_idx + 1:06d}",
                "steps": steps,
                "total_reward": total_reward,
                "outcome": outcome,
            }
        )

    summary = {
        "episodes": episodes,
        "avg_reward": sum(total_rewards) / len(total_rewards),
        "purchase_rate": purchases / episodes,
        "policy": "RuleBasedPolicy",
    }
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
