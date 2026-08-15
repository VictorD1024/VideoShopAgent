# VideoShopAgent

面向短视频电商的多模态购物推荐 Agent 项目，核心包含 `VideoShopEnv` 仿真环境，用于生成用户-智能体交互轨迹，并支持 SFT、DPO 和 Agentic RL 训练评测。

系统模拟用户在短视频流中的观看、点击、加购、购买、组合购买、跳过和退货等行为，用于生成 Agent 轨迹数据，并评测短视频电商推荐智能体的多步决策能力。

核心问题：

> 在短视频内容流里，给定当前视频和用户状态，Agent 多步调用工具，决定“推什么、是否推券、是否换替代、如何解释”，并用可计算 reward 优化转化与风险。

## 1. 项目定位

传统搜广推通常解决：

```text
给定用户、场景和候选商品，如何排序 Top-K 商品/广告/内容。
```

VideoShopAgent 关注更高层的决策问题：

```text
用户正在刷短视频时，Agent 应该推什么？
是否需要推券？
是否应该换成低价替代品？
如何引用工具证据解释推荐？
这次推荐会带来转化，还是带来错类目、虚假优惠、高退货风险或用户打扰？
```

因此，本项目不是替代搜广推，而是在搜索、推荐、广告和多模态理解之上构建一个 **Agentic decision layer**。

## 2. 核心能力

- 构建短视频电商用户仿真环境。
- 模拟用户画像、观看历史、视频场景、商品图文、价格、评论和库存。
- 支持闭集工具调用，MVP 阶段只保留 5 个工具，避免一开始做成全能导购。
- 根据点击、加购、成单代理分、错类目、虚假优惠、高退货风险、证据引用和打扰成本计算 reward。
- 生成可用于 SFT、DPO、RL 或 Agent 评测的多轮交互轨迹。
- 支持规则策略、相似度策略、LLM 工具调用策略和后续 Agentic RL 策略对比。

## 3. 目录结构

```text
VideoShopAgent/
  README.md
  pyproject.toml
  configs/
    default.yaml
  data/
    raw/              # 原始公开数据，如 ABO、ESCI、Reviews、OTTO/RetailRocket
    processed/        # 清洗后的商品、视频场景、用户画像和行为表
    synthetic/        # 合成用户、合成视频场景、合成 ground truth
  docs/
    architecture.md
  notebooks/
    README.md
  outputs/
    trajectories/     # 生成的 episode 轨迹
    reports/          # 离线评测报告
    datasets/         # SFT、DPO、RL 训练数据
  scripts/
    generate_mock_trajectories.py
    export_training_data.py
  src/
    videoshop/
      __init__.py
      simulator/
        schemas.py
        env.py
        feed.py
        tools.py
        state.py
        reward.py
        user_simulator.py
        rollout.py
        evaluator.py
        trajectory.py
      policies/
        random.py
        rule_based.py
      data/
        mock.py
      training/
        exporters.py
  tests/
    test_reward.py
```

## 4. 环境抽象

### State

```text
user_profile:
  用户国家、预算水平、风格偏好、品类兴趣、价格敏感度、广告疲劳度、购买意图
  来源：仿真

session_state:
  当前步数、最近观看品类、最近点击、最近加购、最近跳过
  来源：仿真 rollout 产生

current_video:
  视频标题、检测场景、检测物体、风格、创作者类型、视频/文本向量
  来源：第一版仿真；后续可由视频帧、标题、标签和多模态模型生成

candidate_products:
  商品编号、标题、品类、价格、评分、库存、评论风险、利润标签、商品图文向量
  来源：公开商品数据 + 仿真补充

commerce_context:
  券库存、券门槛、商品库存、是否高退货风险、是否高毛利/清库存
  来源：仿真
```

### Action / Tool 闭集

```text
retrieve_candidates(video_context, user_summary)
  根据当前视频和用户兴趣召回候选商品。

rank_products(candidates, state)
  对候选商品做相关性、转化和风险排序。

get_coupon(product_id, user_id)
  查询商品是否有可用优惠券，以及是否适合给当前用户。

find_substitute(product_id, constraints)
  当商品价格过高、风险过高或库存不足时，寻找替代商品。

explain_recommendation(product_id, evidence)
  基于视频、用户、商品、评论和优惠券证据生成推荐解释。
```

MVP 的最终动作从工具结果中产生：

```text
show_product_card(product_id)
show_coupon(product_id)
switch_to_substitute(product_id)
show_explanation(product_id)
delay_recommendation
```

### Reward

```text
点击商品卡：+1
加入购物车：+3
成单代理分：+10
正确使用优惠券：+2
成功替换高风险/高价商品：+2
解释引用工具证据：+1

错类目推荐：-4
虚假优惠：-5
高退货风险未解释：-4
无证据解释：-3
打扰用户体验：-2
```

不把“用户满意度 LLM-as-judge”作为唯一主 reward。LLM 可以辅助生成场景、用户画像和解释文本，但主 reward 必须能由结构化日志、规则和工具证据自动计算。

## 5. 数据规划

第一阶段使用 mock 数据跑通仿真闭环。

第二阶段逐步接入公开数据：

| 数据集 | 用途 |
|---|---|
| Amazon Berkeley Objects | 商品图片、标题、类目、颜色、材质、尺寸等多模态商品信息 |
| Amazon ESCI | Query-Product 相关性标签，用于搜索相关性和替代品/互补品构造 |
| Amazon Reviews / UCSD | 评论、评分、卖点、痛点和退货风险估计 |
| RetailRocket / OTTO | 浏览、加购、购买行为，用于校准用户响应概率和 reward |

## 6. 最小运行目标

第一版目标是生成 100 条 mock episode：

```bash
python scripts/generate_mock_trajectories.py --episodes 100
```

预期输出：

```text
outputs/trajectories/mock_trajectories.jsonl
outputs/reports/mock_eval_summary.json
```

评测摘要示例：

```json
{
  "episodes": 100,
  "avg_reward": 12.4,
  "avg_steps": 5.8,
  "ctr": 0.42,
  "add_to_cart_rate": 0.18,
  "purchase_rate": 0.08,
  "interruption_rate": 0.12,
  "policy": "RuleBasedPolicy"
}
```

每条 episode 包含：

```json
{
  "episode_id": "E000001",
  "steps": [
    {
      "t": 1,
      "state": {},
      "action": {},
      "user_response": {},
      "reward": 1.0
    }
  ],
  "total_reward": 8.5,
  "outcome": "purchase"
}
```

## 7. Baseline 策略

MVP 阶段至少实现三类 baseline：

1. **RandomPolicy**
   - 随机选择动作和商品，用作最低基线。

2. **PopularityPolicy**
   - 推荐评分高、库存充足或模拟销量高的商品。

3. **RuleBasedPolicy**
   - 根据用户兴趣、视频场景、商品品类、价格、评论风险和广告疲劳度选择动作。

后续增加：

- SimilarityPolicy
- LLMToolCallingPolicy
- SFTPolicy
- AgenticRLPolicy

## 8. Agentic RL 方向

当前已经支持从 `VideoShopEnv` 生成三类训练数据：

```text
SFT:
  state -> tool calls -> final action
  用于训练 Agent 模仿专家策略完成工具调用和动作选择。

Preference / DPO:
  prompt -> chosen action / rejected hard negative
  hard negative 覆盖虚假优惠、错过替代、高风险未解释、错类目、过早打扰和无证据解释。

RL:
  state, action, reward, next_state, done
  用于在环境中优化推荐动作、工具调用顺序和长期转化策略。
```

导出命令：

```bash
python scripts/generate_mock_trajectories.py --episodes 100 --seed 42
python scripts/export_training_data.py \
  --input outputs/trajectories/mock_trajectories.jsonl \
  --out-dir outputs/datasets
```

训练目标不是单纯提升点击，而是联合优化：

```text
点击 + 加购 + 成单代理分 + 正确用券 + 合理替代 + 证据解释
- 错类目 - 虚假优惠 - 高退货风险 - 无证据解释 - 用户打扰
```

## 9. 协作约定

```text
main:
  稳定可运行版本。

dev:
  日常集成分支。

feature/*:
  功能开发分支，例如 feature/simulator-core、feature/reward-model、feature/tool-policy。
```

## 10. License

本仓库代码使用 Apache-2.0 License。外部公开数据集需遵守其各自的 license 和使用条款。
