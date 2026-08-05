# VideoShopSimulator 设计文档

## 1. 核心定义

VideoShopSimulator 是 VideoShopAgent 的可 rollout 环境，用于模拟短视频电商内容流中的多轮用户-智能体交互。

它要回答的问题不是传统推荐系统里的：

```text
给定候选商品，哪个商品应该排第一？
```

而是：

```text
在用户刷短视频的过程中，Agent 应该何时介入、推什么商品、是否推券、是否切换替代品，以及如何基于证据解释推荐？
```

一句话定义：

> 在短视频内容流里，给定当前视频和用户状态，Agent 多步调用工具，决定“推什么、是否推券、是否换替代、如何解释”，并用可计算 reward 优化转化与风险。

## 2. Multi-turn 交互

这里的 multi-turn 不是传统对话多轮，而是短视频内容流里的连续决策。

一个 episode 表示一个用户的一次短视频浏览会话：

```text
Episode = 一个用户打开短视频电商 App 后连续刷 N 条视频的 session
Step = 一条当前视频 + 一次 Agent 决策 + 一次用户反馈 + 一次状态更新
```

每一轮交互：

```text
用户看到当前视频
  -> Env 暴露当前 state
  -> Agent 调用闭集工具
  -> Agent 选择最终动作
  -> UserSimulator 生成用户反馈
  -> RewardModel 计算 reward
  -> StateTransition 更新用户状态
  -> VideoFeedSimulator 切换下一条视频
```

## 3. State

State 必须可观测、可复现，并且明确区分真实信号和仿真信号。

### 用户状态

```text
user_profile:
  user_id
  country
  budget_level
  style_preferences
  category_interests
  price_sensitivity
  risk_sensitivity
  ad_fatigue
  purchase_intent
```

来源：

```text
仿真生成。
后续可用真实行为数据统计分布校准。
```

### 会话状态

```text
session_state:
  step
  recent_watch_categories
  recent_clicks
  recent_carts
  recent_skips
  exposed_products
  used_coupons
```

来源：

```text
rollout 过程中由环境更新。
```

### 当前视频

```text
current_video:
  video_id
  caption
  scene
  detected_objects
  styles
  creator_type
  topic
  video_embedding
```

来源：

```text
第一版使用仿真视频场景。
后续可由视频标题、标签、封面图、关键帧和多模态模型生成。
```

### 候选商品

```text
candidate_products:
  product_id
  title
  category
  price
  rating
  inventory
  review_risk
  margin_tag
  clearance_tag
  tags
  image_embedding
  text_embedding
```

来源：

```text
公开商品数据 + 仿真补充。
可接入 ABO、Amazon ESCI、Amazon Reviews、RetailRocket/OTTO。
```

### 商业上下文

```text
commerce_context:
  coupon_inventory
  coupon_threshold
  coupon_expiry
  campaign_budget
  stock_pressure
  risk_constraints
```

来源：

```text
仿真生成。
用于模拟平台或商家的转化策略约束。
```

## 4. Tool / Action 闭集

MVP 不做开放式全能导购，只保留可评测的闭集工具。

### 工具集合

```text
retrieve_candidates(video_context, user_summary)
  根据当前视频和用户兴趣召回候选商品。

rank_products(candidates, state)
  对候选商品进行相关性、转化收益和风险排序。

get_coupon(product_id, user_id)
  查询商品是否有可用优惠券，以及是否适合给当前用户。

find_substitute(product_id, constraints)
  当商品价格过高、风险过高、库存不足或类目不完全匹配时，寻找替代商品。

explain_recommendation(product_id, evidence)
  基于视频、用户、商品、评论、优惠券和替代品证据生成推荐解释。
```

### 最终动作集合

```text
delay_recommendation
show_product_card(product_id)
show_coupon(product_id)
switch_to_substitute(product_id)
show_explanation(product_id)
```

工具调用可以是多步的，但最终动作必须落在闭集里，便于做 reward 计算和策略学习。

## 5. 用户反馈

UserSimulator 根据当前 state、Agent action 和商品匹配度生成用户反馈。

反馈事件：

```text
continue_watch
click_product
add_to_cart
purchase
skip
negative_feedback
return_or_refund
```

用户反馈可以由概率模型生成：

```text
p_click = f(视频商品相关性, 用户兴趣, 价格匹配, 解释质量, 广告疲劳)
p_cart = f(p_click, 评分, 评论风险, 优惠券)
p_purchase = f(p_cart, 价格敏感度, 优惠券, 信任度, 库存压力)
p_return = f(评论风险, 类目错配, 无证据解释, 过度营销)
```

第一版使用规则和随机采样即可，后续使用 RetailRocket/OTTO 等行为数据校准概率。

## 6. State Transition

每一步用户反馈都会改变下一轮状态。

示例规则：

```text
点击相关商品:
  对应品类兴趣 +0.1
  purchase_intent +0.1
  ad_fatigue -0.03

加入购物车:
  purchase_intent +0.2
  recent_carts 加入商品

购买:
  episode_done = true

跳过推荐:
  recent_skips +1
  ad_fatigue +0.12
  purchase_intent -0.05

无关推荐:
  ad_fatigue +0.2
  对应品类兴趣不变或下降

延迟推荐:
  ad_fatigue -0.02
  用户继续观看
```

Multi-turn 的价值在于模拟：

```text
推荐时机
策略递进
广告疲劳
购买意图演化
纠错能力
长期 reward
```

## 7. Reward

Reward 必须自动可计算，不能把 LLM-as-judge 作为唯一主 reward。

### 正向收益

```text
点击商品卡: +1
加入购物车: +3
成单代理分: +10
正确使用优惠券: +2
成功替换高价/高风险商品: +2
解释引用工具证据: +1
```

### 约束惩罚

```text
错类目推荐: -4
虚假优惠: -5
高退货风险未解释: -4
无证据解释: -3
打扰用户体验: -2
退货或退款: -8
```

### 解释证据检查

解释必须能被工具证据支持：

```text
类目证据来自商品表或视频标签。
优惠券证据来自 get_coupon。
替代品证据来自 find_substitute。
评论风险证据来自评论摘要或风险字段。
不能声明未出现在 evidence 中的材质、折扣、库存、功效或评论结论。
```

## 8. Trajectory 格式

每条 trajectory 是一个 episode。

```json
{
  "episode_id": "E000001",
  "user_profile": {
    "user_id": "U001",
    "country": "US",
    "budget_level": "medium"
  },
  "steps": [
    {
      "t": 1,
      "state": {
        "current_video": {
          "scene": "home office desk setup",
          "detected_objects": ["organizer", "lamp", "keyboard"]
        },
        "session_state": {
          "recent_skips": 0
        }
      },
      "tool_calls": [
        {
          "tool": "retrieve_candidates",
          "input": {
            "scene": "home office desk setup"
          },
          "output": {
            "candidate_product_ids": ["P001", "P002"]
          }
        }
      ],
      "action": {
        "type": "show_product_card",
        "product_id": "P001",
        "reason": "Matches desk setup scene and user's minimal style preference."
      },
      "user_response": {
        "clicked": true,
        "added_to_cart": false,
        "purchased": false,
        "skipped": false
      },
      "reward": 1.0,
      "state_update": {
        "purchase_intent_delta": 0.1,
        "ad_fatigue_delta": -0.03
      }
    }
  ],
  "total_reward": 8.5,
  "outcome": "purchase"
}
```

## 9. Ground Truth 构造

不要只靠 LLM 编标签。Ground Truth 分三层：

```text
强标签:
  Amazon ESCI 相关性标签
  RetailRocket/OTTO 的浏览、加购、购买行为

弱标签:
  ABO 商品类目、属性、图文相似度
  Amazon Reviews 评论风险和卖点/痛点
  相似商品迁移

合成标签:
  用户画像
  视频场景
  券库存
  best_action
  reward 采样
```

文档和实验报告中必须标明每个字段是真实信号、弱信号还是仿真信号。

## 10. Baseline 和训练路线

### Baseline

```text
RandomPolicy:
  随机动作。

RuleBasedPolicy:
  根据类目、价格、风险和疲劳度做规则决策。

SimilarityPolicy:
  根据视频-商品、用户-商品相似度推荐。

LLMToolCallingPolicy:
  使用闭集工具调用进行多步决策。
```

### 训练路线

```text
1. 用规则策略和相似度策略生成高质量轨迹。
2. 从高 reward 轨迹构造 SFT 数据，训练 action/tool-call policy。
3. 根据 total reward 或人工偏好构造好坏轨迹对，做 DPO/偏好优化。
4. 在 VideoShopSimulator 中 rollout，做 Agentic RL。
5. 在 held-out 用户、视频场景和商品集合上评测泛化。
```

## 11. 最小实现模块

```text
src/videoshop/simulator/feed.py
  视频流生成和下一条视频采样。

src/videoshop/simulator/state.py
  根据用户反馈更新兴趣、购买意图、广告疲劳和购物车。

src/videoshop/simulator/tools.py
  5 个闭集工具的 mock 实现。

src/videoshop/simulator/rollout.py
  episode rollout loop。

src/videoshop/simulator/evaluator.py
  reward、转化率、风险和解释证据评测。
```

## 12. 与 ShopSimulator 的关系

ShopSimulator 最值得借鉴的是：

```text
金标/约束
可 rollout 环境
SFT 冷启动
再用 RL 优化策略
```

VideoShopSimulator 不复刻对话导购场景，而是将这套工程骨架迁移到短视频内容流：

```text
ShopSimulator:
  用户主动购物，Agent 搜索、点击、询问、购买。

VideoShopSimulator:
  用户刷视频，Agent 决定商品曝光、优惠券、替代品、解释和延迟推荐。
```
