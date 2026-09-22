from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from videoshop.feed.benchmark import FEED_SPLITS, OracleStepGreedyPolicy, build_split, run_feed_benchmark
from videoshop.feed.candidate_provider import CandidateProviderConfig
from videoshop.feed.policies import FEED_POLICIES, INTERVENTION_POLICIES

ORACLE = OracleStepGreedyPolicy.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v0.2 feed-control benchmark.")
    parser.add_argument(
        "--policy",
        default="rule_based",
        choices=[*sorted(FEED_POLICIES), ORACLE, "all"],
        help=f"'{ORACLE}' reads private world state and is a ceiling reference, not a legal agent.",
    )
    parser.add_argument(
        "--intervention",
        default=None,
        choices=sorted(INTERVENTION_POLICIES),
        help="Run the composite env: the feed policy picks the exposure, this policy picks the treatment.",
    )
    parser.add_argument("--split", default="smoke", choices=list(FEED_SPLITS))
    parser.add_argument("--scenarios", type=int, default=None, help="Override the split size.")
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates-per-step", type=int, default=8)
    parser.add_argument("--ineligible-rate", type=float, default=0.22)
    parser.add_argument("--commercial-optimism", type=float, default=0.35)
    parser.add_argument(
        "--coupon-treatment-rate",
        type=float,
        default=CandidateProviderConfig.coupon_treatment_rate,
        help="Share of product candidates offered as coupons. At the default, coupons are "
        "only ~5%% of candidates, too thin to measure the intervention layer against.",
    )
    parser.add_argument(
        "--coupon-trap-rate",
        type=float,
        default=CandidateProviderConfig.coupon_trap_rate,
        help="Share of coupon offers that will not actually redeem.",
    )
    parser.add_argument("--out", default="outputs/feed_benchmarks/episodes.jsonl")
    parser.add_argument("--report", default="outputs/feed_benchmarks/summary.json")
    args = parser.parse_args()

    provider_config = CandidateProviderConfig(
        candidates_per_step=args.candidates_per_step,
        ineligible_rate=args.ineligible_rate,
        commercial_optimism=args.commercial_optimism,
        coupon_treatment_rate=args.coupon_treatment_rate,
        coupon_trap_rate=args.coupon_trap_rate,
    )
    split = build_split(args.split, count=args.scenarios, provider_config=provider_config)
    scenarios, provider = split.scenarios, split.provider

    policy_names = [*sorted(FEED_POLICIES), ORACLE] if args.policy == "all" else [args.policy]
    all_episodes = []
    summaries = []
    for policy_name in policy_names:
        episodes, summary = run_feed_benchmark(
            policy_name,
            split=args.split,
            scenarios=scenarios,
            provider=provider,
            seed=args.seed,
            max_steps=args.max_steps,
            episodes_per_scenario=args.episodes_per_scenario,
            intervention=args.intervention,
        )
        all_episodes.extend(episodes)
        summaries.append(summary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for episode in all_episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    report = summaries[0] if len(summaries) == 1 else {"split": args.split, "policies": summaries}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
