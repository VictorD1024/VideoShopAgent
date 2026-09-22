# VideoShopAgent

[English](README.md) | [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

VideoShopAgent 是短视频购物信息流的 Agent 仿真和评测环境。一个 episode 是一次观看会话。Agent 选择下一条视频，或者在视频已经播放时决定展示什么。

![VideoShopAgent 架构图](videoshopagent-architecture.png)

## 安装

需要 Python 3.10 及以上。

```bash
pip install -e .
```

## 快速开始

下面的例子连续投放两条候选里的第一条：

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

`reward` 是 `RewardVector`，包含 `content_value`、`commerce_value`、`user_value`、`ecosystem_value`、`risk_cost`。

跑测试和一个小评测：

```bash
python -m pytest
python scripts/run_feed_benchmark.py --policy rule_based --split smoke --scenarios 20
```

## 环境

| 环境 | Schema | 谁选视频 | Agent 输出 |
| --- | --- | --- | --- |
| [`VideoShopEnv`](#videoshopenv) | v1 | 环境 | `show_product_card`、`show_coupon`、`switch_to_substitute`、`show_explanation`、`delay_recommendation` 之一 |
| [`FeedControlEnv`](#feedcontrolenv) | v2 | Agent | `serve_exposure(exposure_id)` |
| [`FeedInterventionEnv`](#feedinterventionenv) | v2 | 先由 feed Agent 选，再由第二个 Agent 处理 | 一条视频，然后这条视频上的一个 v1 动作 |

v1 和 v2 轨迹不能混用。转换用 `videoshop.feed.legacy.upgrade_v1_episode`。

### VideoShopEnv

环境写入 `current_video`。Agent 调用工具后提交一个动作。模拟用户会点击、跳过、购买或离开。下一条视频仍由环境选择。

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

固定 toy split 有 10 个场景：有效优惠券、假券、高风险商品、替代品、广告疲劳、类目错配、过期券、库存压力、解释、低库存。更大的目录：

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

每一步大约 8 个候选，至少一条普通视频（不挂商品）和一条商业视频（`seller`、`affiliate` 或 `ad`）。动作只有 `exposure_id`，不能把别的商品挂到这条视频上。不满足资格的候选不会播出，这一步仍然扣分。

候选上的 `base_scores` 有噪声，并且偏向商业。真实点击率和购买率不在 observation 里。

```bash
python scripts/run_feed_benchmark.py --policy all --split smoke
```

smoke split，100 个场景，seed 42，6 步，平均标量 reward：

| 策略 | Reward | 95% CI | 商业曝光占比 | Gold@1 |
| --- | --- | --- | --- | --- |
| `greedy_gmv` | -1.98 | [-2.52, -1.45] | 0.97 | 0.11 |
| `random` | -1.50 | [-1.93, -1.08] | 0.59 | 0.20 |
| `rule_based` | 2.27 | [1.54, 3.04] | 0.27 | 0.68 |
| `always_organic` | 2.38 | [1.70, 3.01] | 0.00 | 0.67 |
| `oracle_step_greedy` | 2.84 | [2.12, 3.64] | 0.27 | 1.00 |

`greedy_gmv` 总是投放预测 GMV 最高的候选，会话会提前结束。`always_organic` 和 `rule_based` 在这个 split 上重叠。`oracle_step_greedy` 读取隐藏状态，是上限，不是可训练策略。

三个 split 不共享商品、视频和用户 id：

| Split | 场景数 | 商品 | 视频 | 用户 |
| --- | --- | --- | --- | --- |
| `train` | 500 | 648 | 180 | 90 |
| `smoke` | 100 | 216 | 60 | 30 |
| `frozen_eval` | 200 | 216 | 60 | 30 |

每个 split 仍覆盖全部 12 个品类。报告里有 `dataset_hash` 和 provider 配置。给 `run_feed_benchmark` 传 split 名字。只传 scenarios、不传生成它们的 provider，会报错。

Reward 配置：`content_first`、`balanced`、`gmv_first`、`retention_first`、`clearance_campaign`。`risk_cost` 是减项。

### FeedInterventionEnv

候选与上面相同。选中的如果是商业视频，第二个 Agent 会在这条视频和锚定商品上调用 v1 工具。用户看到的是工具校验后留下的动作，不是候选上印的 treatment。

```text
serve_exposure(X03_02)          # 报价 treatment：coupon
get_coupon(P1) -> available=false
show_product_card(P1)           # 券被拿掉
```

`coupon_trap_rate`（默认 0.3）是印出来的券里实际核销不了的比例。只有 `get_coupon` 能区分。照样展示会记 `fake_coupon`。第二个 Agent 用 `build_intervention_agent` 构造。

```bash
python scripts/run_feed_benchmark.py --policy all --split smoke --intervention anchor_rule_based
python scripts/run_feed_benchmark.py --policy greedy_gmv --split smoke --coupon-treatment-rate 0.6 --intervention trusting
```

默认配比里优惠券大约只占候选的 5%。要测干预 Agent，调高 `--coupon-treatment-rate`。券较多时，`anchor_rule_based` 与只跑 feed 的分数一致，`trusting` 更低。

## 训练数据

`generate_mock_trajectories.py` 写出 `outputs/trajectories/mock_trajectories.jsonl`。

```bash
python scripts/export_training_data.py \
  --input outputs/trajectories/mock_trajectories.jsonl \
  --out-dir outputs/datasets
```

| 文件 | 内容 |
| --- | --- |
| `sft.jsonl` | 状态、工具调用、最终动作 |
| `dpo_pairs.jsonl` | 选中的动作和一个 hard negative（`fake_coupon`、`missed_substitute`、`high_risk_unexplained`、`category_mismatch`、`premature_intervention`、`unsupported_explanation`） |
| `rl_rollouts.jsonl` | 状态、动作、reward、用户反应、下一状态 |

这些导出读的是 v1 episode。

## LLM Agent

`videoshop.agents` 把模型的 tool call 转成 `AgentStep`。

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

`--resume` 只补还没有的 episode id。过滤并转换一次运行：

```bash
python scripts/convert_gold_trajectories.py \
  --input outputs/llm_trajectories/qwen_synthetic_tool_calling_v2.jsonl \
  --out-dir outputs/datasets/qwen_synthetic_gold
```

## 轨迹文件

没有 `schema_version` 的文件按 v1 读。

| 版本 | 谁写出 | 动作字段 |
| --- | --- | --- |
| v1 | `VideoShopEnv` | `step["action"]`；LLM 运行是 `step["agent_step"]["final_action"]` |
| v2 | `FeedControlEnv`、`FeedInterventionEnv` | `serve_exposure` |

`upgrade_v1_episode` 把原来的标量留在 `legacy_reward`。两种动作字段都没有的 step 会报错。`outputs/llm_trajectories/` 里已有的 Qwen、DeepSeek 文件是这套转换的回归样本。

## 目录

```text
examples/          脚本 Agent 和 SGLang 示例
scripts/           评测、轨迹导出、LLM 运行
src/videoshop/
  agents/          tool-calling 循环
  data/            mock 和合成目录
  feed/            v0.2 环境
  policies/        v1 规则策略和随机策略
  simulator/       VideoShopEnv
  training/        SFT、DPO 和 gold 转换
tests/
```

## 许可证

[Apache-2.0](LICENSE)。
