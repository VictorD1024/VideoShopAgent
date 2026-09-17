import json
import subprocess
import sys

from videoshop.data.scenarios import build_toy_scenarios
from videoshop.data.synthetic import (
    SyntheticCommerceConfig,
    build_synthetic_catalog,
    build_synthetic_scenarios,
    build_synthetic_videos,
)
from videoshop.simulator.benchmark import run_benchmark


def test_toy_scenarios_are_fixed_and_named():
    scenarios = build_toy_scenarios()
    scenario_ids = [scenario.scenario_id for scenario in scenarios]

    assert len(scenarios) == 10
    assert len(set(scenario_ids)) == 10
    assert "coupon_valid" in scenario_ids
    assert "delay_when_fatigue_high" in scenario_ids
    assert all(scenario.video_feed for scenario in scenarios)
    assert all(scenario.expected_behaviors for scenario in scenarios)


def test_run_benchmark_returns_episode_rows_and_summary():
    episodes, summary = run_benchmark("rule_based", episodes_per_scenario=1, seed=42)

    assert len(episodes) == 10
    assert summary["policy"] == "rule_based"
    assert summary["episodes"] == 10
    assert summary["scenarios"] == 10
    assert "avg_reward" in summary
    assert "gross_purchase_rate" in summary
    assert "net_purchase_rate" in summary
    assert "constraint_violations" in summary
    assert all("scenario_id" in episode for episode in episodes)


def test_synthetic_split_builds_large_catalog_with_limited_candidate_pools():
    config = SyntheticCommerceConfig(
        seed=7,
        category_count=6,
        products_per_category=20,
        users=8,
        videos_per_category=3,
        scenario_count=5,
        candidate_pool_size=16,
        coupon_coverage=0.35,
    )

    catalog = build_synthetic_catalog(config)
    videos = build_synthetic_videos(config)
    scenarios = build_synthetic_scenarios(config)

    assert len(catalog) == 120
    assert len(videos) == 18
    assert len(scenarios) == 5
    assert all(len(scenario.candidate_products) == 16 for scenario in scenarios)
    assert any(product.has_coupon for product in catalog)
    assert any(scenario.commerce_context.coupon_inventory for scenario in scenarios)


def test_run_benchmark_supports_synthetic_split():
    config = SyntheticCommerceConfig(
        seed=11,
        category_count=4,
        products_per_category=12,
        users=5,
        videos_per_category=2,
        scenario_count=3,
        candidate_pool_size=12,
        coupon_coverage=0.3,
    )

    episodes, summary = run_benchmark(
        "rule_based",
        episodes_per_scenario=1,
        seed=11,
        split="synthetic",
        synthetic_config=config,
    )

    assert len(episodes) == 3
    assert summary["split"] == "synthetic"
    assert summary["scenarios"] == 3
    assert "avg_reward" in summary


def test_run_benchmark_cli_writes_outputs(tmp_path):
    episodes_path = tmp_path / "episodes.jsonl"
    report_path = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_benchmark.py",
            "--policy",
            "rule_based",
            "--episodes-per-scenario",
            "1",
            "--out",
            str(episodes_path),
            "--report",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(report_path.read_text(encoding="utf-8"))
    assert summary["episodes"] == 10
    assert episodes_path.read_text(encoding="utf-8").count("\n") == 10
    assert '"policy": "rule_based"' in result.stdout
