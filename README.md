# VideoShopAgent

[English](README.md) | [简体中文](README.zh-CN.md)

VideoShopAgent is a short-video commerce agent environment for generating and evaluating tool-use trajectories. Its core environment, `VideoShopEnv`, models how a shopping agent decides when to intervene in a content feed, what product to recommend, whether to use coupons or substitutes, and how to ground recommendations with evidence.

The project focuses on the agentic decision layer above retrieval, ranking, ads, and multimodal understanding. It produces structured trajectories for SFT, DPO, RL rollouts, and offline agent evaluation.

![VideoShopAgent architecture](videoshopagent-architecture.png)

## Highlights

- Short-video commerce environment with user, session, video, product, and commerce state.
- Closed tool/action space for controllable agent behavior.
- Reward model covering conversion, coupon correctness, substitution, grounding, interruption, category mismatch, and return risk.
- Rule-based and random baseline policies.
- JSONL trajectory generation with state, tool calls, action, user response, reward, and state transition.
- Training data exporters for SFT, DPO preference pairs, and RL rollouts.
- DPO hard negatives for realistic commerce-agent failures.

## Environment

Each `VideoShopEnv` episode simulates one short-video shopping session.

```text
state
  -> tool calls
  -> final action
  -> simulated user response
  -> reward
  -> state update
  -> next video
```

State includes:

- `user_profile`: country, budget level, style preferences, category interests, price sensitivity, risk sensitivity, ad fatigue, purchase intent.
- `session_state`: step, recent watched categories, clicks, carts, skips, exposed products, used coupons.
- `current_video`: caption, scene, category, detected objects, styles, creator type.
- `candidate_products`: title, category, price, rating, inventory, review risk, margin/clearance tags, coupon info.
- `commerce_context`: coupon inventory, coupon thresholds, coupon expiry, campaign budget, stock pressure, risk constraints.

## Tools And Actions

Closed tools:

```text
retrieve_candidates(video_context, user_summary)
rank_products(candidates, state)
get_coupon(product_id, user_id)
find_substitute(product_id, constraints)
explain_recommendation(product_id, evidence)
```

Final actions:

```text
delay_recommendation
show_product_card(product_id)
show_coupon(product_id)
switch_to_substitute(product_id)
show_explanation(product_id)
```

## Reward

Positive signals:

```text
click product card
add to cart
purchase proxy
valid coupon usage
valid substitute
grounded explanation
```

Penalties:

```text
wrong category recommendation
fake coupon
unexplained high return risk
unsupported explanation
user interruption
return or refund
```

The reward is computed from structured logs and tool evidence instead of relying only on LLM-as-judge.

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Generate mock trajectories:

```bash
python scripts/generate_mock_trajectories.py --episodes 100 --seed 42
```

Export training data:

```bash
python scripts/export_training_data.py \
  --input outputs/trajectories/mock_trajectories.jsonl \
  --out-dir outputs/datasets
```

Run tests:

```bash
python -m pytest
```

## Outputs

Trajectory generation writes:

```text
outputs/trajectories/mock_trajectories.jsonl
outputs/reports/mock_eval_summary.json
```

Training data export writes:

```text
outputs/datasets/sft.jsonl
outputs/datasets/dpo_pairs.jsonl
outputs/datasets/rl_rollouts.jsonl
```

Evaluation summary example:

```json
{
  "episodes": 100,
  "avg_reward": 10.25,
  "avg_steps": 6.35,
  "ctr": 0.4724,
  "add_to_cart_rate": 0.2063,
  "purchase_rate": 0.39,
  "interruption_rate": 0.2409,
  "policy": "RuleBasedPolicy"
}
```

## Training Data

SFT samples train action imitation:

```text
messages:
  system: agent role and constraints
  user: current VideoShopEnv state
  assistant: tool calls and final action
```

DPO pairs train preference over commerce-aware actions:

```text
prompt: current state
chosen: policy action
rejected: hard negative action
metadata: reward, outcome, negative_type
```

Supported hard negative types:

```text
fake_coupon
missed_substitute
high_risk_unexplained
category_mismatch
premature_intervention
unsupported_explanation
```

RL rollouts provide transition-level data:

```text
state
tool_calls
action
reward
user_response
state_update
next_state
done
```

## Project Structure

```text
VideoShopAgent/
  configs/
  data/
    raw/
    processed/
    synthetic/
  docs/
  notebooks/
  outputs/
    trajectories/
    reports/
    datasets/
  scripts/
    generate_mock_trajectories.py
    export_training_data.py
  src/videoshop/
    data/
    policies/
    simulator/
    training/
  tests/
```

## Roadmap

- Add more baseline policies such as risk-aware, coupon-aware, and conversion-greedy policies.
- Add counterfactual reward estimation for DPO rejected actions.
- Expand mock scenarios with richer product catalogs and user cohorts.
- Calibrate user response probabilities with public behavior datasets.
- Add multimodal video/product embeddings for content-product grounding.
- Add LLM tool-calling policies for offline comparison.

## License

This project is released under the Apache-2.0 License. External datasets should be used according to their own licenses and terms.

## Toy Benchmark

Run the fixed VideoShopEnv toy benchmark split:

```bash
python scripts/run_benchmark.py --policy rule_based --episodes-per-scenario 1 --seed 42
```

Outputs are written to:

```text
outputs/benchmarks/toy_episodes.jsonl
outputs/benchmarks/toy_summary.json
```

The toy split contains 10 fixed scenarios covering valid coupons, fake coupon traps, high-risk products, substitutes, ad fatigue, category mismatch, expired coupons, stock pressure, grounded explanations, and low inventory guards.

For larger cold-start data generation, use the synthetic commerce split:

```bash
python scripts/run_benchmark.py \
  --split synthetic \
  --policy rule_based \
  --synthetic-categories 12 \
  --synthetic-products-per-category 80 \
  --synthetic-scenarios 50 \
  --synthetic-candidate-pool-size 32 \
  --out outputs/benchmarks/synthetic_episodes.jsonl \
  --report outputs/benchmarks/synthetic_summary.json
```

The synthetic split creates a large catalog, coupon inventory, stock pressure, user profiles, videos, and typed scenarios such as valid coupon, fake coupon trap, high-risk explanation, substitute, fatigue delay, low inventory guard, clearance pressure, and category mismatch trap. Each scenario exposes a bounded candidate pool so prompts stay tractable while the underlying catalog can scale.

## Function Calling Agent

VideoShopEnv keeps `AgentStep` as the internal protocol, and adds provider-style function/tool calling adapters under `videoshop.agents`. A minimal scripted example is available at:

```bash
python examples/function_calling_agent.py
```

The tool schema includes `retrieve_candidates`, `rank_products`, `get_coupon`, `find_substitute`, `explain_recommendation`, and `submit_final_action`. Provider tool calls are normalized into `AgentStep` before being passed to `env.step_agent(...)`.

### SGLang / OpenAI-compatible endpoint

For an OpenAI-compatible SGLang endpoint, configure credentials with environment variables instead of hard-coding them:

```bash
export VIDEOSHOP_LLM_BASE_URL="http://113.128.201.101:8001/v1"
export VIDEOSHOP_LLM_MODEL="Qwen3.8-27B"
export VIDEOSHOP_LLM_API_KEY="<your-api-key>"
python examples/qwen_sglang_agent.py
```

### Generate LLM tool-calling trajectories

After configuring the OpenAI-compatible endpoint, save LLM tool-calling traces with:

```bash
python scripts/generate_llm_trajectories.py --episodes-per-scenario 1 --max-scenarios 10
```

Synthetic LLM trajectory generation uses the same split controls:

```bash
python scripts/generate_llm_trajectories.py \
  --split synthetic \
  --episodes-per-scenario 1 \
  --synthetic-scenarios 50 \
  --synthetic-categories 12 \
  --synthetic-products-per-category 80 \
  --synthetic-candidate-pool-size 32 \
  --out outputs/llm_trajectories/qwen_synthetic_tool_calling.jsonl \
  --report outputs/llm_trajectories/qwen_synthetic_tool_calling_summary.json
```

Outputs are written to:

```text
outputs/llm_trajectories/qwen_tool_calling.jsonl
outputs/llm_trajectories/qwen_tool_calling_summary.json
```

Each step records the observation, model tool requests, final action, explicit `reasoning_summary`, executed tool results, reward, user response, state update, termination flags, and constraint violations. `reasoning_summary` is an audit artifact requested from the model, not hidden chain-of-thought.
