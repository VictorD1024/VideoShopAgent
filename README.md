# VideoShopAgent

[English](README.md) | [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

VideoShopAgent is a simulator and benchmark for agents in a short-video shopping feed. One episode is one viewing session. The agent picks the next clip, or decides what to show once a clip is already playing.

![VideoShopAgent architecture](videoshopagent-architecture.png)

## Installation

Python 3.10 or newer.

```bash
pip install -e .
```

## Quick start

Serve the first candidate for two steps:

```python
from videoshop.feed import FeedControlEnv, FeedDecision, build_default_provider, build_smoke_scenarios

provider = build_default_provider()
scenarios = build_smoke_scenarios(count=1, seed=7, provider=provider)
env = FeedControlEnv(provider, max_steps=2, seed=42)

observation = env.reset(scenarios[0], seed=42)
done = False
while not done:
    exposure_id = observation.candidate_exposures[0]["exposure_id"]
    observation, reward, terminated, truncated, info = env.step(
        FeedDecision.serve(exposure_id)
    )
    done = terminated or truncated
```

`reward` is a `RewardVector` (`content_value`, `commerce_value`, `user_value`, `ecosystem_value`, `risk_cost`).

Run the tests and a small benchmark:

```bash
python -m pytest
python scripts/run_feed_benchmark.py --policy rule_based --split smoke --scenarios 20
```

## Environments

| Environment | Schema | Who picks the video | Agent output |
| --- | --- | --- | --- |
| [`VideoShopEnv`](#videoshopenv) | v1 | the environment | one of `show_product_card`, `show_coupon`, `switch_to_substitute`, `show_explanation`, `delay_recommendation` |
| [`FeedControlEnv`](#feedcontrolenv) | v2 | the agent | `serve_exposure(exposure_id)` |
| [`FeedInterventionEnv`](#feedinterventionenv) | v2 | the feed agent, then a second agent | a clip, then a v1 action on that clip |

v1 and v2 trajectory files are not interchangeable. Convert a v1 file with `videoshop.feed.legacy.upgrade_v1_episode`.

### VideoShopEnv

The environment sets `current_video`. The agent calls tools, then submits one action. The simulated user clicks, skips, buys, or leaves. The environment picks the next video.

```text
retrieve_candidates(limit?)
rank_products(candidate_product_ids?)
get_coupon(product_id)
find_substitute(product_id)
explain_recommendation(product_id)
```

```bash
python scripts/run_benchmark.py --policy rule_based --episodes-per-scenario 1 --seed 42
python scripts/generate_mock_trajectories.py --episodes 100 --seed 42
```

The fixed toy split has 10 scenarios (valid coupon, fake coupon, high-risk product, substitute, ad fatigue, category mismatch, expired coupon, stock pressure, explanation, low inventory). For a larger catalog:

```bash
python scripts/run_benchmark.py \
  --split synthetic \
  --policy rule_based \
  --synthetic-categories 12 \
  --synthetic-products-per-category 80 \
  --synthetic-scenarios 50 \
  --out outputs/benchmarks/synthetic_episodes.jsonl \
  --report outputs/benchmarks/synthetic_summary.json
```

### FeedControlEnv

Each step the agent receives about 8 candidates. At least one is organic (no product) and one is commercial (`seller`, `affiliate`, or `ad`). The action is only an `exposure_id`: the agent cannot attach a different product to the clip. Candidates that fail eligibility are not shown; that step still costs reward.

`base_scores` on a candidate are noisy and biased toward commerce. True click and purchase rates are not in the observation.

```bash
python scripts/run_feed_benchmark.py --policy all --split smoke
```

Smoke split, 100 scenarios, seed 42, 6 steps, mean scalar reward:

| Policy | Reward | 95% CI | Commercial share | Gold@1 |
| --- | --- | --- | --- | --- |
| `greedy_gmv` | -1.98 | [-2.52, -1.45] | 0.97 | 0.11 |
| `random` | -1.50 | [-1.93, -1.08] | 0.59 | 0.20 |
| `rule_based` | 2.27 | [1.54, 3.04] | 0.27 | 0.68 |
| `always_organic` | 2.38 | [1.70, 3.01] | 0.00 | 0.67 |
| `oracle_step_greedy` | 2.84 | [2.12, 3.64] | 0.27 | 1.00 |

`greedy_gmv` always serves the highest predicted GMV and ends sessions early. `always_organic` and `rule_based` overlap on this split. `oracle_step_greedy` reads hidden state; it is a ceiling, not a trainable policy.

Splits do not share product, video, or user ids:

| Split | Scenarios | Products | Videos | Users |
| --- | --- | --- | --- | --- |
| `train` | 500 | 648 | 180 | 90 |
| `smoke` | 100 | 216 | 60 | 30 |
| `frozen_eval` | 200 | 216 | 60 | 30 |

Every split still covers all 12 categories. A report includes `dataset_hash` and the provider config. Pass a split name to `run_feed_benchmark`. Passing scenarios without the provider that built them raises an error.

Reward profiles: `content_first`, `balanced`, `gmv_first`, `retention_first`, `clearance_campaign`. `risk_cost` is subtracted.

### FeedInterventionEnv

Same candidate list. If the chosen clip is commercial, a second agent runs the v1 tools on that clip and its anchor product. The user sees the action that survives the tools, not the treatment printed on the candidate.

```text
serve_exposure(X03_02)          # offered treatment: coupon
get_coupon(P1) -> available=false
show_product_card(P1)           # coupon dropped
```

`coupon_trap_rate` (default 0.3) is the fraction of printed coupons that will not redeem. `get_coupon` is the only way to tell. Showing one anyway records `fake_coupon`. Build the second agent with `build_intervention_agent`.

```bash
python scripts/run_feed_benchmark.py --policy all --split smoke --intervention anchor_rule_based
python scripts/run_feed_benchmark.py --policy greedy_gmv --split smoke --coupon-treatment-rate 0.6 --intervention trusting
```

Coupons are about 5% of candidates at the default mix. Raise `--coupon-treatment-rate` to measure the intervention agent. On a coupon-heavy smoke run, `anchor_rule_based` matches feed-only and `trusting` scores worse.

## Training data

`generate_mock_trajectories.py` writes `outputs/trajectories/mock_trajectories.jsonl`.

```bash
python scripts/export_training_data.py \
  --input outputs/trajectories/mock_trajectories.jsonl \
  --out-dir outputs/datasets
```

| File | Contents |
| --- | --- |
| `sft.jsonl` | state, tool calls, final action |
| `dpo_pairs.jsonl` | chosen action and a hard negative (`fake_coupon`, `missed_substitute`, `high_risk_unexplained`, `category_mismatch`, `premature_intervention`, `unsupported_explanation`) |
| `rl_rollouts.jsonl` | state, action, reward, user response, next state |

These exporters read v1 episodes.

## LLM agents

`videoshop.agents` turns provider tool calls into `AgentStep`.

```bash
python examples/function_calling_agent.py
```

```bash
export VIDEOSHOP_LLM_BASE_URL="http://127.0.0.1:8001/v1"
export VIDEOSHOP_LLM_MODEL="Qwen3.8-27B"
export VIDEOSHOP_LLM_API_KEY="<your-api-key>"
python examples/qwen_sglang_agent.py
python scripts/generate_llm_trajectories.py --episodes-per-scenario 1 --max-scenarios 10
```

`--resume` appends only missing episode ids. Filter and convert a run:

```bash
python scripts/convert_gold_trajectories.py \
  --input outputs/llm_trajectories/qwen_synthetic_tool_calling_v2.jsonl \
  --out-dir outputs/datasets/qwen_synthetic_gold
```

## Trajectory files

A file with no `schema_version` is v1.

| Version | Written by | Action field |
| --- | --- | --- |
| v1 | `VideoShopEnv` | `step["action"]`, or `step["agent_step"]["final_action"]` for LLM runs |
| v2 | `FeedControlEnv`, `FeedInterventionEnv` | `serve_exposure` |

`upgrade_v1_episode` keeps the original scalar in `legacy_reward`. A step with neither action field raises. Recorded Qwen and DeepSeek files in `outputs/llm_trajectories/` are the regression set for this converter.

## Layout

```text
examples/          scripted and SGLang agents
scripts/           benchmarks, trajectory export, LLM runs
src/videoshop/
  agents/          tool-calling loop
  data/            mock and synthetic catalogs
  feed/            v0.2 environments
  policies/        v1 rule and random policies
  simulator/       VideoShopEnv
  training/        SFT, DPO, and gold converters
tests/
```

## License

[Apache-2.0](LICENSE).
