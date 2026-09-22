# VideoShopAgent

[English](README.md) | [简体中文](README.zh-CN.md)

VideoShopAgent is an open-source short-video commerce simulator and benchmark for training agents to optimize content distribution, product exposure, commercial interventions, and long-term user value.

Its core environment, `VideoShopEnv`, models how an agent decides when and how to intervene in a content feed under user, product, inventory, promotion, and risk constraints. The project focuses on the agentic decision layer above retrieval, ranking, ads, and multimodal understanding, and produces structured trajectories for SFT, DPO, RL rollouts, and offline agent evaluation.

![VideoShopAgent architecture](videoshopagent-architecture.png)

## Highlights

- Short-video commerce environment with user, session, video, product, and commerce state.
- Closed tool/action space for controllable agent behavior.
- Reward model covering conversion, coupon correctness, substitution, grounding, interruption, category mismatch, and return risk.
- Rule-based and random baseline policies.
- JSONL trajectory generation with state, tool calls, action, user response, reward, and state transition.
- Training data exporters for SFT, DPO preference pairs, and RL rollouts.
- DPO hard negatives for realistic commerce-agent failures.
- v0.2 feed-control layer where the agent selects the next `user x video x product x treatment` exposure, with a multi-objective reward vector and an explicit public/private scenario boundary.

## Two Layers

The project has three environments built from two decision units. They are versioned separately and are not interchangeable.

```text
FeedControlEnv      (v0.2, schema v2)  -> which exposure to serve next
VideoShopEnv        (v0.1, schema v1)  -> how to intervene on a fixed video
FeedInterventionEnv (v0.2, schema v2)  -> both, composed
```

`VideoShopEnv` is the intervention layer: the environment picks `current_video`, and the agent decides the commercial treatment on that video. `FeedControlEnv` sits above it and gives the agent control over the exposure itself, including whether to serve organic content at all.

`FeedInterventionEnv` composes the two. The feed policy picks the exposure; for commercial exposures the intervention layer then runs the real `ToolRuntime` and evidence validation to decide how to present it. The candidate's `treatment` becomes an *offer* rather than a fact:

```text
feed policy      -> serve_exposure(X03_02)          video + anchor product + offered treatment
intervention     -> get_coupon(...) -> unavailable  tools resolve what is actually true
                 -> show_product_card               realized treatment, coupon declined
user model       -> responds to the realized exposure
```

This is what makes the coupon axis non-trivial. A share of coupon exposures (`coupon_trap_rate`) advertise a coupon that will not redeem, and `ExposureTruth.coupon_available` is the only authority. The trap is invisible from the outside: the ranker's `base_scores` describe the coupon as advertised, so a feed-only policy cannot tell a real offer from a stale one, while a coupon that does not redeem grants no conversion lift and no discounted price in *either* environment. Only `get_coupon` settles it.

The intervention layer is therefore a liability surface rather than a bonus. Showing an unverified coupon books a `fake_coupon` violation, priced into `risk_cost`, which carries the v1 grounding penalties into the v2 reward vector instead of quietly dropping them. Two baselines ship: `anchor_rule_based` verifies before it shows, and `trusting` applies the offer verbatim.

```text
smoke, --coupon-treatment-rate 0.6   feed-only   +anchor   +trusting
greedy_gmv                              -2.54     -2.53      -3.57
rule_based                              +2.36     +2.36      +2.01
always_organic                          +2.21     +2.21      +2.21
```

An agent that verifies lands back on the feed-only outcome; an agent that does not, pays. `always_organic` is identical across all three because it never serves a commercial exposure, so the intervention layer never runs.

Coupons are only about 5% of candidates at default settings, which is too thin to measure this against; `--coupon-treatment-rate` and `--coupon-trap-rate` stress the axis.

Evidence is bound on three sides, the tool input, the tool output, and the product the action finally commits to. Quoting a real `find_substitute` result and then switching to a different product is a violation, not a pass. The feed layer's binding also holds: the video is fixed, and leaving the anchor product is only legitimate through `find_substitute`.

Declining the product keeps the exposure's `source_type`. A declined ad is still ad supply with `treatment="none"`; it does not get reclassified as organic, which would otherwise let an agent shed ad fatigue and risk cost for free.

The intervention agent does not see the v1 `Observation`. `FeedInterventionEnv` builds an `InterventionObservation` that publishes `anchor_product_id`, `offered_treatment` and `source_type`, and strips `coupon_inventory` / threshold / expiry. A coupon's true availability is only reachable through `get_coupon`. Dropping a bare `FunctionCallingAgent` into the composite env rewrites its v1 brief automatically; the explicit factory is `build_intervention_agent`. Each intervention record keeps the full `agent_step` (tool requests, `reasoning_summary`, tool-call trace, the system prompt that was used) so the episode is trainable, not just scorable.

## Intervention Environment (VideoShopEnv)

Each `VideoShopEnv` episode simulates one short-video shopping session. The environment, not the agent, chooses the next video.

```text
state
  -> tool calls
  -> final action
  -> simulated user response
  -> reward
  -> state update
  -> next video (chosen by the environment)
```

State includes:

- `user_profile`: country, budget level, style preferences, category interests, price sensitivity, risk sensitivity, ad fatigue, purchase intent.
- `session_state`: step, recent watched categories, clicks, carts, skips, exposed products, used coupons.
- `current_video`: caption, scene, category, detected objects, styles, creator type.
- `candidate_products`: title, category, price, rating, inventory, review risk, margin/clearance tags, coupon info.
- `commerce_context`: coupon inventory, coupon thresholds, coupon expiry, campaign budget, stock pressure, risk constraints.

## Tools And Actions

Closed tools, as the agent may call them. Arguments the environment fills in itself (user, video, constraints) are not part of the callable signature:

```text
retrieve_candidates(limit?)
rank_products(candidate_product_ids?)
get_coupon(product_id)
find_substitute(product_id)
explain_recommendation(product_id)
```

`find_substitute` applies the scenario's constraints and `explain_recommendation` returns evidence; neither accepts them as input.

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

## Feed Decision Core (v0.2)

`FeedControlEnv` makes exposure selection the decision. Every step the agent receives a bounded candidate set that always contains both organic and commercial options, and serves exactly one.

```text
CandidateProvider
  -> FeedControlEnv (serve_exposure)
  -> user response
  -> RewardVector
  -> next candidate set
```

An `ExposureCandidate` is one `user x video x product x treatment` unit:

```text
exposure_id, video_id
source_type:  organic | seller | affiliate | ad
product_id:   null for organic
treatment:    none | product_anchor | coupon
placement:    for_you | search | shop_tab
base_scores:  expected_watch_time, skip_probability, product_click_probability,
              purchase_probability, expected_net_gmv, refund_probability
eligibility:  in_stock, coupon_valid, policy_compliant, risk_within_limit, audience_allowed
```

Protocol guarantees, all covered by tests:

- Organic content is a formal candidate. "No commercial intervention" means serving organic content, not idling, and at least one servable organic candidate always exists.
- Video/product bindings are candidate facts. `FeedDecision` carries only an `exposure_id`, so an agent structurally cannot re-bind a product to another video.
- Candidates failing a hard constraint are intercepted by the environment and never reach the user; the decision costs reward instead.
- `base_scores` are a noisy, commerce-optimistic view of the private response parameters, so a policy cannot read the truth off the observation.
- `ScenarioSpec` separates `public_context` from `hidden_world_state` and `oracle`. Instructions are scanned for answer leakage at construction time.
- Only Top-1 `serve_exposure` is supported. `rank_exposures` is deliberately deferred until the Top-1 protocol is stable.

The reward is a vector, scalarized by a named objective profile:

```text
content_value, commerce_value, user_value, ecosystem_value, risk_cost
profiles: content_first | balanced | gmv_first | retention_first | clearance_campaign
```

`risk_cost` is a non-negative magnitude that scalarization subtracts, and every component ships with a per-term breakdown so a reward number can be audited back to the events that produced it.

Run the benchmark:

```bash
python scripts/run_feed_benchmark.py --policy all --split smoke
python scripts/run_feed_benchmark.py --policy all --split smoke --intervention anchor_rule_based
```

Baselines on the 100-scenario smoke split (seed 42, 6 steps, average scalar reward):

| Policy | Reward | 95% CI | Commercial share | Gold@1 |
| --- | --- | --- | --- | --- |
| `greedy_gmv` | -1.98 | [-2.52, -1.45] | 0.97 | 0.11 |
| `random` | -1.50 | [-1.93, -1.08] | 0.59 | 0.20 |
| `rule_based` | 2.27 | [1.54, 3.04] | 0.27 | 0.68 |
| `always_organic` | 2.38 | [1.70, 3.01] | 0.00 | 0.67 |
| `oracle_step_greedy` | 2.84 | [2.12, 3.64] | 0.27 | 1.00 |

`greedy_gmv` trusts the ranker's optimistic scores and loses: over-exposing commerce ends sessions early. `always_organic` is a strong floor whose confidence interval overlaps `rule_based`, so the hand-written policy is not convincingly better than serving no commerce at all; the two separate on `frozen_eval` (2.33 vs 2.24) and on the `gmv_first` and `clearance_campaign` profiles. The oracle beats both by serving roughly 27% commercial exposures, so the commerce axis is not decorative. `oracle_step_greedy` reads private world state and is a ceiling reference, not a legal agent.

### Splits

Splits are `smoke` (100 scenarios), `train` (500), and `frozen_eval` (200). They partition **entities**, not just random seeds: one world of 1080 products, 300 videos and 150 users is carved into disjoint pools, stratified within category so every split still covers all 12 categories.

```text
train        648 products   180 videos   90 users
smoke        216 products    60 videos   30 users
frozen_eval  216 products    60 videos   30 users
```

No product, video, or user id appears in two splits. Each split owns the provider that generated it, so step 0 and every later step of an episode draw from the same catalog. Each report carries a `dataset_hash`, the provider config, and the pool sizes.

Pairing is enforced rather than assumed. `run_feed_benchmark` takes a `FeedSplit`, or both `scenarios` and `provider`, and rejects one without the other; passing scenarios alone used to fall back to the default split's provider and quietly run frozen_eval scenarios against the smoke catalog. `FeedControlEnv.reset` is the backstop: it refuses a scenario whose entities the provider does not own, including the subtle case where the ids line up but the attributes behind them differ.

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
  examples/
  outputs/
    trajectories/
    reports/
    datasets/
    benchmarks/
    feed_benchmarks/
  scripts/
    generate_mock_trajectories.py
    export_training_data.py
    run_benchmark.py
    run_feed_benchmark.py
    generate_llm_trajectories.py
    convert_gold_trajectories.py
  src/videoshop/
    agents/
    data/
    feed/
    policies/
    simulator/
    training/
  tests/
```

`src/videoshop/feed/` holds the v0.2 layer: `schemas.py`, `candidate_provider.py`, `control_env.py`, `composite_env.py`, `user_model.py`, `reward.py`, `policies.py`, `scenarios.py`, `splits.py`, `benchmark.py`, and `legacy.py`.

## Schema Versions

Trajectories are self-describing. Episodes written before versioning are v1 by definition.

```text
v1  VideoShopEnv intervention episodes (frozen)
v2  FeedControlEnv / FeedInterventionEnv exposure episodes
```

`videoshop.feed.legacy.upgrade_v1_episode` projects a v1 episode into the v2 envelope, mapping each legacy action to a treatment (`show_coupon` to `coupon`, `delay_recommendation` to organic, and so on). It keeps the original scalar under `legacy_reward` rather than fabricating a reward vector v1 never recorded.

v1 episodes come in two shapes and the adapter reads both: rollout and benchmark episodes store the decision under `action`, while LLM tool-calling episodes store it under `agent_step.final_action` and have no `action` key at all. A step matching neither raises rather than defaulting, because defaulting silently rewrote every commercial decision as an organic no-op. Recoverable-but-malformed records, such as a `show_product_card` with a null `product_id`, are downgraded and listed in `conversion_warnings`. The recorded Qwen and DeepSeek runs under `outputs/llm_trajectories/` are used directly as regression fixtures.

## Roadmap

- Add `rank_exposures` once the Top-1 feed protocol is stable.
- Let the feed layer offer several treatments for the same `video x product` pair, so treatment becomes a feed-level choice rather than only an intervention-level one.
- Benchmark LLM agents on `FeedInterventionEnv`. The observation protocol and trainable traces are in place; live Qwen/DeepSeek runs are the next step, ahead of `rank_exposures`.
- Remove answer leakage from v1 synthetic scenario objectives, which still name the expected action.
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
  --episode-retries 2 \
  --max-tool-rounds 5 \
  --max-tokens 2048 \
  --out outputs/llm_trajectories/qwen_synthetic_tool_calling.jsonl \
  --report outputs/llm_trajectories/qwen_synthetic_tool_calling_summary.json
```

Use `--resume` to append only missing episode IDs after an interrupted run. Each episode receives an independent deterministic seed, and transient network, HTTP 429, and HTTP 5xx failures are retried with the same seed.

Outputs are written to:

```text
outputs/llm_trajectories/qwen_tool_calling.jsonl
outputs/llm_trajectories/qwen_tool_calling_summary.json
```

Each step records the observation, validated model tool requests, final action, explicit `reasoning_summary`, per-round raw tool-call trace and provider metadata, executed tool results, reward, user response, state update, termination flags, and constraint violations. Invalid protocol output is repaired before the environment state advances. `reasoning_summary` is an audit artifact requested from the model, not hidden chain-of-thought. Reports separate gross purchases from net purchases after returns; `purchase_rate` is the net purchase rate.

### Convert strict Gold trajectories to SFT data

Filter failed, repaired, malformed, or ungrounded episodes and convert the remaining trajectories with:

```bash
python scripts/convert_gold_trajectories.py \
  --input outputs/llm_trajectories/qwen_synthetic_tool_calling_v2.jsonl \
  --out-dir outputs/datasets/qwen_synthetic_gold \
  --expected-gold 12
```

The converter emits the selected source episodes, reconstructed OpenAI-compatible tool-calling messages, a flattened action-JSON variant, and a rejection report. The original trajectory stores flattened tool requests rather than raw provider turn boundaries, so reconstructed tool sequences are marked in sample metadata.

```text
outputs/datasets/qwen_synthetic_gold/gold_episodes.jsonl
outputs/datasets/qwen_synthetic_gold/openai_tool_sft.jsonl
outputs/datasets/qwen_synthetic_gold/action_sft.jsonl
outputs/datasets/qwen_synthetic_gold/conversion_report.json
```
