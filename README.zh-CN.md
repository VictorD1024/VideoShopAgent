# VideoShopAgent

[English](README.md) | [简体中文](README.zh-CN.md)

VideoShopAgent 是一个面向短视频电商场景的 Agent 环境，用于生成和评测工具调用轨迹。核心环境 `VideoShopEnv` 模拟购物 Agent 在内容流中如何决策：何时介入、推荐什么商品、是否使用优惠券或替代品，以及如何基于证据解释推荐。

本项目聚焦在搜索、推荐、广告和多模态理解之上的 Agentic decision layer。它可以生成结构化轨迹，用于 SFT、DPO、RL rollout 和离线 Agent 评测。

![VideoShopAgent 架构图](videoshopagent-architecture.png)

## 核心特性

- 面向短视频电商的环境状态，包含用户、会话、视频、商品和商业上下文。
- 闭集工具和动作空间，便于控制 Agent 行为并进行可重复评测。
- Reward 覆盖转化、优惠券正确性、替代品选择、证据 grounding、用户打扰、类目错配和退货风险。
- 提供规则策略和随机策略 baseline。
- 生成 JSONL 轨迹，包含 state、tool calls、action、user response、reward 和 state transition。
- 支持导出 SFT、DPO preference pairs 和 RL rollouts。
- DPO hard negatives 覆盖真实电商 Agent 常见错误。

## 环境

每个 `VideoShopEnv` episode 表示一次短视频购物会话。

```text
state
  -> tool calls
  -> final action
  -> simulated user response
  -> reward
  -> state update
  -> next video
```

State 包含：

- `user_profile`：国家、预算水平、风格偏好、类目兴趣、价格敏感度、风险敏感度、广告疲劳度、购买意图。
- `session_state`：当前步数、最近观看类目、点击、加购、跳过、已曝光商品、已使用优惠券。
- `current_video`：标题、场景、类目、检测物体、风格、创作者类型。
- `candidate_products`：标题、类目、价格、评分、库存、评论风险、毛利/清仓标签、优惠券信息。
- `commerce_context`：券库存、券门槛、券过期步数、活动预算、库存压力、风险约束。

## 工具与动作

闭集工具：

```text
retrieve_candidates(video_context, user_summary)
rank_products(candidates, state)
get_coupon(product_id, user_id)
find_substitute(product_id, constraints)
explain_recommendation(product_id, evidence)
```

最终动作：

```text
delay_recommendation
show_product_card(product_id)
show_coupon(product_id)
switch_to_substitute(product_id)
show_explanation(product_id)
```

## Reward

正向信号：

```text
点击商品卡
加入购物车
成单代理分
有效使用优惠券
有效替代商品
有证据的推荐解释
```

惩罚项：

```text
错类目推荐
虚假优惠券
高退货风险未解释
无证据解释
打扰用户体验
退货或退款
```

Reward 由结构化日志和工具证据计算，不依赖单一 LLM-as-judge。

## 快速开始

以 editable 模式安装：

```bash
pip install -e .
```

生成 mock trajectories：

```bash
python scripts/generate_mock_trajectories.py --episodes 100 --seed 42
```

导出训练数据：

```bash
python scripts/export_training_data.py \
  --input outputs/trajectories/mock_trajectories.jsonl \
  --out-dir outputs/datasets
```

运行测试：

```bash
python -m pytest
```

## 输出

轨迹生成输出：

```text
outputs/trajectories/mock_trajectories.jsonl
outputs/reports/mock_eval_summary.json
```

训练数据导出输出：

```text
outputs/datasets/sft.jsonl
outputs/datasets/dpo_pairs.jsonl
outputs/datasets/rl_rollouts.jsonl
```

评测摘要示例：

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

## 训练数据

SFT 样本用于动作模仿：

```text
messages:
  system: Agent 角色和约束
  user: 当前 VideoShopEnv state
  assistant: 工具调用和最终动作
```

DPO pairs 用于训练电商约束下的动作偏好：

```text
prompt: 当前状态
chosen: 策略动作
rejected: hard negative 动作
metadata: reward, outcome, negative_type
```

支持的 hard negative 类型：

```text
fake_coupon
missed_substitute
high_risk_unexplained
category_mismatch
premature_intervention
unsupported_explanation
```

RL rollouts 提供 transition 级数据：

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

## 项目结构

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

- 增加 risk-aware、coupon-aware、conversion-greedy 等 baseline policy。
- 为 DPO rejected actions 增加 counterfactual reward 估计。
- 扩展 mock 场景，加入更丰富的商品池和用户群体。
- 使用公开行为数据校准用户响应概率。
- 加入视频和商品多模态 embedding，增强内容-商品 grounding。
- 增加 LLM tool-calling policy，用于离线对比。

## License

本项目使用 Apache-2.0 License。外部数据集需遵守其各自的 license 和使用条款。

## Toy Benchmark

运行固定的 VideoShopEnv toy benchmark：

```bash
python scripts/run_benchmark.py --policy rule_based --episodes-per-scenario 1 --seed 42
```

输出文件：

```text
outputs/benchmarks/toy_episodes.jsonl
outputs/benchmarks/toy_summary.json
```

当前 toy split 包含 10 个固定场景，覆盖有效优惠券、虚假优惠券陷阱、高风险商品、替代品、广告疲劳、类目错配、过期优惠券、库存压力、grounded explanation 和低库存保护。

如果要生成更大规模冷启动数据，可以使用 synthetic commerce split：

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

synthetic split 会生成更大的商品目录、优惠券库存、库存压力、用户画像、视频流和类型化场景，包括有效优惠券、虚假优惠券陷阱、高风险解释、替代品、疲劳延迟、低库存保护、清仓库存压力和类目错配陷阱。每个 scenario 只暴露有限候选池，避免 prompt 过长，同时底层 catalog 可以继续扩大。

## Function Calling Agent

VideoShopEnv 内部仍以 `AgentStep` 作为标准协议，同时在 `videoshop.agents` 下新增了 provider 风格 function/tool calling 适配层。最小脚本示例：

```bash
python examples/function_calling_agent.py
```

当前工具 schema 包括 `retrieve_candidates`、`rank_products`、`get_coupon`、`find_substitute`、`explain_recommendation` 和 `submit_final_action`。外部模型返回的 tool calls 会先归一化为 `AgentStep`，再传给 `env.step_agent(...)`。

### SGLang / OpenAI-compatible endpoint

如果使用 OpenAI-compatible 的 SGLang 内网接口，建议用环境变量配置，不要把 Key 写入代码：

```bash
export VIDEOSHOP_LLM_BASE_URL="http://113.128.201.101:8001/v1"
export VIDEOSHOP_LLM_MODEL="Qwen3.8-27B"
export VIDEOSHOP_LLM_API_KEY="<your-api-key>"
python examples/qwen_sglang_agent.py
```

### 生成 LLM tool-calling 轨迹

配置 OpenAI-compatible 接口后，可以保存真实 LLM tool-calling 轨迹：

```bash
python scripts/generate_llm_trajectories.py --episodes-per-scenario 1 --max-scenarios 10
```

synthetic LLM 轨迹也使用同一套 split 参数：

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

输出文件：

```text
outputs/llm_trajectories/qwen_tool_calling.jsonl
outputs/llm_trajectories/qwen_tool_calling_summary.json
```

每个 step 会记录 observation、模型 tool requests、final action、显式 `reasoning_summary`、实际执行的 tool results、reward、user response、state update、终止标记和 constraint violations。`reasoning_summary` 是要求模型输出的可审计推理摘要，不是隐藏 chain-of-thought。
