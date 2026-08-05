# 设计原则：从 ShopSimulator 到短视频电商内容流

## 1. 一句话定义

在短视频内容流里，给定当前视频和用户状态，Agent 多步调用工具，决定“推什么、是否推券、是否换替代、如何解释”，并用可计算 reward 优化转化与风险。

## 2. 不做什么

第一版不做全能导购，不做开放动作空间，不把 LLM-as-judge 当唯一主 reward。

MVP 需要先跑通一个可复现、可 rollout、可自动评测的闭环：

```text
state -> tool calls -> action -> env response -> reward -> trajectory
```

## 3. State 必须可观测、可复现

### 真实或来自公开数据的部分

```text
product:
  商品标题、类目、价格、评分、评论、图片、属性、库存代理字段

review:
  评论摘要、正向卖点、负向风险、退货/质量风险关键词

behavior:
  浏览、点击、加购、购买的统计分布，用于校准响应概率
```

可用数据：

```text
ABO:
  商品图文、属性、多模态商品表征

Amazon ESCI:
  Query-Product 相关性、替代/互补/无关标签

Amazon Reviews:
  评论、评分、卖点、痛点、风险

RetailRocket / OTTO:
  浏览、加购、购买行为
```

### 仿真的部分

```text
user_profile:
  国家、预算、兴趣、风格、价格敏感度、广告疲劳度

current_video:
  视频场景、检测物体、主题、风格、创作者类型

commerce_context:
  券库存、券门槛、库存压力、高毛利/清库存标签

session_state:
  最近点击、最近跳过、最近加购、当前会话深度
```

仿真字段必须有生成配置和随机种子，保证同一批 episode 可以复现。

## 4. Action / Tool 闭集

MVP 只保留 5 个工具：

```text
retrieve_candidates(video_context, user_summary)
rank_products(candidates, state)
get_coupon(product_id, user_id)
find_substitute(product_id, constraints)
explain_recommendation(product_id, evidence)
```

工具输出必须结构化，方便 reward 检查：

```json
{
  "tool": "get_coupon",
  "product_id": "P001",
  "available": true,
  "coupon_id": "C001",
  "discount": 0.15,
  "expires_in_hours": 12
}
```

最终动作也保持闭集：

```text
show_product_card
show_coupon
switch_to_substitute
show_explanation
delay_recommendation
```

## 5. Reward 必须自动可算

### 短期转化

```text
点击商品卡：+1
加入购物车：+3
成单代理分：+10
```

### 策略收益

```text
正确使用优惠券：+2
成功替换高价/高风险商品：+2
解释引用工具证据：+1
```

### 约束惩罚

```text
错类目推荐：-4
虚假优惠：-5
高退货风险未解释：-4
无证据解释：-3
打扰用户体验：-2
```

### 解释检查

解释不能只由 LLM 主观判断。第一版用规则检查：

```text
解释中引用的类目必须来自商品表或视频标签。
解释中引用的优惠券必须来自 get_coupon 工具。
解释中引用的评论风险必须来自 review/risk 工具字段。
解释不能声明未出现在工具证据中的材质、折扣、库存、功效。
```

## 6. Ground Truth 构造

三层标签：

```text
强标签：
  ESCI 相关性、RetailRocket/OTTO 行为事件。

弱标签：
  ABO 属性匹配、图文相似度、评论风险、相似商品迁移。

合成标签：
  用户画像、视频场景、券库存、best action、reward 采样。
```

报告和 README 里必须明确哪些是真实信号，哪些是仿真信号。

## 7. 训练路线

```text
1. 用规则策略生成可解释、高约束轨迹。
2. 从高 reward 轨迹构造 SFT 数据，训练 action/tool-call policy。
3. 构造好坏轨迹对，用 DPO 或偏好优化校准策略。
4. 在可 rollout 环境中做 RL，优化长期 reward。
5. 用 held-out 用户、视频场景和商品集合评测泛化。
```

## 8. 与 ShopSimulator 的关系

ShopSimulator 的重点不是“对话导购”这个具体场景，而是：

```text
金标/约束 + 可 rollout 环境 + SFT 冷启动 + RL 优化
```

VideoShopAgent 搬用这套工程骨架，但把场景换成短视频内容流，并用 VideoShopSimulator 作为可 rollout 环境：

```text
ShopSimulator:
  用户主动购物，Agent 搜索、点击、询问、购买。

VideoShopAgent / VideoShopSimulator:
  用户刷视频，Agent 决定商品曝光、用券、替代、解释和延迟推荐。
```
