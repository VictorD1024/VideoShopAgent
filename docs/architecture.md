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

