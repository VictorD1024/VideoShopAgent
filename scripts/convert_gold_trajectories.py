from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from videoshop.training.exporters import load_jsonl, write_jsonl
from videoshop.training.gold_converter import (
    build_action_json_sft_samples,
    build_conversion_report,
    build_openai_tool_sft_samples,
    select_gold_episodes,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter strict Gold LLM trajectories and convert them into trainable SFT JSONL files."
    )
    parser.add_argument(
        "--input",
        default="outputs/llm_trajectories/qwen_synthetic_tool_calling_v2.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/datasets/qwen_synthetic_gold",
    )
    parser.add_argument(
        "--expected-gold",
        type=int,
        default=None,
        help="Fail if the strict Gold filter does not select this many episodes.",
    )
    args = parser.parse_args()

    episodes = load_jsonl(args.input)
    gold, rejected = select_gold_episodes(episodes)
    if args.expected_gold is not None and len(gold) != args.expected_gold:
        raise SystemExit(
            f"Expected {args.expected_gold} Gold episodes, but strict filtering selected {len(gold)}."
        )

    openai_samples = build_openai_tool_sft_samples(gold)
    action_samples = build_action_json_sft_samples(gold)
    report = build_conversion_report(episodes, gold, rejected, len(openai_samples))
    report["input"] = str(Path(args.input))
    report["output_dir"] = str(Path(args.out_dir))

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "gold_episodes.jsonl", gold)
    write_jsonl(out_dir / "openai_tool_sft.jsonl", openai_samples)
    write_jsonl(out_dir / "action_sft.jsonl", action_samples)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
