from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from videoshop.simulator.benchmark import run_benchmark
from videoshop.data.synthetic import SyntheticCommerceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a VideoShopEnv benchmark split.")
    parser.add_argument("--policy", choices=["rule_based", "random"], default="rule_based")
    parser.add_argument("--split", choices=["toy", "synthetic"], default="toy")
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/benchmarks/toy_episodes.jsonl")
    parser.add_argument("--report", default="outputs/benchmarks/toy_summary.json")
    parser.add_argument("--synthetic-categories", type=int, default=12)
    parser.add_argument("--synthetic-products-per-category", type=int, default=80)
    parser.add_argument("--synthetic-users", type=int, default=40)
    parser.add_argument("--synthetic-videos-per-category", type=int, default=8)
    parser.add_argument("--synthetic-scenarios", type=int, default=50)
    parser.add_argument("--synthetic-candidate-pool-size", type=int, default=32)
    parser.add_argument("--synthetic-coupon-coverage", type=float, default=0.22)
    args = parser.parse_args()

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
    episodes, summary = run_benchmark(
        args.policy,
        args.episodes_per_scenario,
        args.seed,
        split=args.split,
        synthetic_config=synthetic_config,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
