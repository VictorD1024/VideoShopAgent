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
from videoshop.simulator.rollout import run_batch
from videoshop.training import export_dpo_pairs, export_rl_rollouts, export_sft_samples
from videoshop.training.exporters import load_jsonl, write_jsonl


def load_or_generate_episodes(input_path: Path, episodes: int, seed: int) -> list[dict]:
    if input_path.exists():
        return load_jsonl(input_path)

    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=8, seed=seed)
    return run_batch(env, RuleBasedPolicy(), episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/trajectories/mock_trajectories.jsonl")
    parser.add_argument("--out-dir", default="outputs/datasets")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    episodes = load_or_generate_episodes(Path(args.input), args.episodes, args.seed)
    out_dir = Path(args.out_dir)

    sft_samples = export_sft_samples(episodes)
    dpo_pairs = export_dpo_pairs(episodes)
    rl_rollouts = export_rl_rollouts(episodes)

    write_jsonl(out_dir / "sft.jsonl", sft_samples)
    write_jsonl(out_dir / "dpo_pairs.jsonl", dpo_pairs)
    write_jsonl(out_dir / "rl_rollouts.jsonl", rl_rollouts)

    summary = {
        "episodes": len(episodes),
        "sft_samples": len(sft_samples),
        "dpo_pairs": len(dpo_pairs),
        "rl_rollouts": len(rl_rollouts),
        "output_dir": str(out_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
