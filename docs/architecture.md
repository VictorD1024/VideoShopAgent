# 架构草图

```text
ProductCatalog + VideoFeed + UserProfiles
        ↓
VideoShopEnv.reset()
        ↓
Policy.act(state)
        ↓
VideoShopEnv.step(action)
        ↓
UserSimulator.respond(state, action)
        ↓
RewardModel.compute(response)
        ↓
TrajectoryLogger.write(episode)
```

当前代码结构：

```text
src/videoshop/
  simulator/
    env.py             # reset/step，多轮环境入口
    feed.py            # 短视频流采样
    tools.py           # retrieve/rank/coupon/substitute/explain 闭集工具
    state.py           # 用户兴趣、购买意图、广告疲劳状态更新
    reward.py          # 可计算 reward
    user_simulator.py  # 用户反馈模拟
    rollout.py         # episode/batch rollout
    evaluator.py       # 离线指标汇总

  policies/
    random.py
    rule_based.py

  data/
    mock.py
```

后续模块扩展：

```text
Multimodal Encoder:
  商品图片、视频帧、标题、评论向量化

Retrieval Engine:
  文本检索、图片检索、视频场景到商品召回

LLM Agent:
  工具调用、动作规划、推荐解释

RL Trainer:
  SFT、偏好优化、策略优化
```
