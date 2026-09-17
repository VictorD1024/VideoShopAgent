# VideoShopAgent 项目优化规划

> 版本：v1.0  
> 更新日期：2026-09-17  
> 目标：将 VideoShopAgent 从可运行的合成环境，升级为可训练、可评测、可扩展的视频原生电商 Agent 基础设施。

## 1. 项目定位

VideoShopAgent 不应成为另一个商品搜索或网页购物环境。项目应聚焦于短视频内容消费中的 Agent 决策问题：

1. 什么时候介入用户的内容消费过程。
2. 推荐什么商品或替代品。
3. 通过商品卡、优惠券或解释等哪种方式介入。
4. 什么情况下应该保持沉默或延迟推荐。
5. 什么情况下应将用户转交给深度购物 Agent。

建议使用以下英文定位：

> VideoShopEnv is a video-native, dynamic and risk-aware commerce environment for training agents to decide when and how to intervene in content feeds.

建议使用以下中文定位：

> VideoShopEnv 是面向短视频内容电商的 Agent 训练与评测环境，研究 Agent 在动态商业约束下何时介入、推荐什么以及如何兼顾转化和用户体验。

## 2. 与 ShopSimulator 的边界

| 维度 | ShopSimulator | VideoShopAgent |
| --- | --- | --- |
| 用户入口 | 用户主动提出购物目标 | 用户正在消费短视频内容 |
| 意图类型 | 显式或待澄清的购买需求 | 潜在、变化且可能不存在的购买意图 |
| 核心决策 | 搜索、比较并购买正确商品 | 是否介入、何时介入、以何种方式介入 |
| 主要动作 | 询问、搜索、点击、查看、购买 | 等待、商品卡、优惠券、替代品、解释、转交 |
| 世界状态 | 商品属性、规格、价格、用户画像 | 视频时间轴、会话行为、疲劳度、库存、券预算、风险 |
| 主要奖励 | 商品约束满足度和最终成功率 | 净成交、留存、打扰、退款、成本和真实性 |
| 技术协议 | 文本动作和网页式交互 | 类型化工具调用和可验证动作 |

VideoShopAgent 应将 ShopSimulator 视为：

- 搜索购物能力的外部基线。
- 高购买意图阶段可插拔的深度购物子环境。
- 长轨迹、个性化和强化学习实验的参考对象。

不建议直接复制 ShopSimulator 的代码或数据。其公开仓库当前未明确提供 LICENSE，应在获得授权前保持实现独立。

## 3. 当前基础

项目目前已经具备：

- 短视频、用户、会话、商品和商业上下文状态。
- `retrieve_candidates`、`rank_products`、`get_coupon`、`find_substitute`、`explain_recommendation` 等工具。
- `delay_recommendation`、`show_product_card`、`show_coupon`、`switch_to_substitute`、`show_explanation` 等动作。
- 原生 function/tool calling 适配。
- 工具参数和最终动作的严格校验。
- 逐轮原始调用轨迹、provider 元数据、奖励和状态变化记录。
- 确定性 seed、失败重试和中断续跑。
- SFT、动作监督和 Gold 轨迹转换能力。
- 点击、加购、净购买、优惠券真实性、打扰、退款和风险奖励。

这些能力已经构成一个良好的 harness，但还不足以构成具有公信力的 benchmark。

## 4. 当前主要问题

### 4.1 任务存在答案泄漏

当前合成任务会直接告诉 Agent “使用优惠券”“等待推荐”或“寻找替代品”。这测量的是指令遵循能力，而不是环境理解和决策能力。

### 4.2 场景规模不等于有效多样性

增加商品数和轨迹数不会自动增加任务难度。当前场景主要由少量模板轮换生成，容易产生同构轨迹和动作分布偏斜。

### 4.3 视频仍是结构化文本状态

当前视频主要由 caption、category、objects、styles 等字段表达，尚未真正评测视频理解、时序定位、ASR/OCR grounding 和商品出现时机。

### 4.4 用户模拟器过于简单

当前点击、加购、购买和退款主要由固定概率公式驱动。模型可能学习模拟器公式，而不是具备可迁移性的用户决策规律。

### 4.5 商业状态缺少真实动态

优惠券、库存、价格和活动预算虽然已进入状态，但缺少随时间、用户行为和多 Agent 竞争而变化的世界动力学。

### 4.6 缺少公认的评测协议

目前还缺少冻结测试集、隐藏 Gold、强弱基线、统计置信区间、分布外测试和可复现实验报告。

## 5. 目标架构

```text
Video / Feed World
  - clip, frame, ASR, OCR, product appearance timeline
  - feed transition and creator/content context
            |
            v
Latent User World
  - short-term intent
  - long-term preference
  - ad fatigue and trust
  - cross-session memory
            |
            v
Dynamic Commerce World
  - catalog and retrieval
  - inventory and price
  - coupon budget and expiry
  - campaign pacing
  - return and fulfillment risk
            |
            v
Agent Runtime
  - observe
  - call tools
  - validate evidence
  - choose intervention or silence
  - optionally hand off to shopping agent
            |
            v
Evaluator and Data Engine
  - deterministic replay
  - step and episode rewards
  - counterfactual evaluation
  - SFT / DPO / RL export
  - benchmark report
```

## 6. 分阶段路线图

## Phase 0：冻结现状和建立基线

目标：建立可重复的项目基准，避免后续优化无法归因。

优先级：P0  
建议周期：1 周

任务：

- 固定当前代码版本、配置、seed 和输出 schema。
- 建立 random、rule-based、small LLM、strong LLM 四档基线。
- 报告每类场景的动作分布、成功率、净购买率、退款率和违规率。
- 对 reward 各分量分别记录，避免只观察总分。
- 为数据集和配置生成版本号及内容 hash。
- 增加多 seed 评测和 bootstrap 置信区间。

验收标准：

- 同一版本和 seed 可逐步重放并得到一致结果。
- benchmark 报告能定位到场景、动作、模型和失败类别。
- 基线结果由一条命令完整复现。

## Phase 1：Benchmark Ready

目标：把环境从显式规则测试升级为隐藏决策评测。

优先级：P0  
建议周期：2 至 3 周

任务：

- 将场景 objective 改写为自然用户和内容上下文，不出现目标动作名称。
- 将 `expected_behaviors` 留在 evaluator，不暴露给 Agent。
- 建立因子化场景生成器，独立采样用户、视频、商品、库存、优惠券、风险和时间。
- 引入难负例：高评分错类商品、看似可用但已过期的券、低价高风险商品、相似替代品。
- 建立 train/dev/test 三套冻结 split。
- 按用户、商品、类目和时间做隔离，防止模板及实体泄漏。
- 将场景规模扩展到至少 5,000 条训练任务和 1,000 条隐藏评测任务。
- 加入无可推荐商品、保持沉默才正确的任务。

验收标准：

- instruction 中不存在 `show_coupon`、`delay_recommendation` 等答案词泄漏。
- rule-based policy 不再接近满分。
- strong LLM 显著优于 random，但仍保留有解释价值的失败空间。
- 测试集商品和用户实体不出现在训练集。

## Phase 2：Video Native

目标：让“VideoShop”中的视频成为必要信息，而不是装饰字段。

优先级：P0  
建议周期：3 至 5 周

任务：

- 定义 `VideoTimeline`：关键帧、ASR、OCR、镜头、商品出现区间和主播话术。
- 支持真实短片、预提取视觉特征和纯文本降级模式。
- 新增时间相关动作参数，例如 `show_product_card(product_id, timestamp)`。
- 评测商品 grounding：推荐商品是否真的出现在视频中。
- 评测时机质量：介入是否早于商品出现、打断内容高潮或错过购买窗口。
- 构建可公开分发的小型视频 benchmark，并明确媒体授权。
- 增加视频输入模型和文本输入模型的对照实验。

验收标准：

- 删除视觉或时间轴信号后，模型性能有可测量下降。
- 环境可以指出推荐所依据的帧、字幕或 OCR 证据。
- 能独立报告商品 grounding 和 intervention timing 指标。

## Phase 3：动态用户与商业世界

目标：从单次概率采样升级为具有状态和延迟反馈的模拟世界。

优先级：P1  
建议周期：4 至 6 周

任务：

- 将用户的兴趣、信任、疲劳和价格敏感度设计为潜变量。
- 使用户状态根据曝光、点击、跳过、虚假优惠和购买结果持续更新。
- 加入跨视频和跨会话状态。
- 模拟优惠券预算消耗、库存变化、价格变化和活动 pacing。
- 加入发货、退款和复购等延迟事件。
- 支持多个策略共享同一商业资源，以评测局部最优和预算争夺。
- 使用真实脱敏统计、公开数据或受约束的 LLM shopper 校准转移概率。
- 对用户模拟器做独立评测，避免 Agent 奖励掩盖模拟器缺陷。

验收标准：

- 短期购买最优策略不必然等于长期价值最优策略。
- 虚假或过度促销会导致后续信任和留存下降。
- 不同用户群体表现出稳定且可解释的行为差异。
- 动力学参数可配置、可校准且可进行敏感性分析。

## Phase 4：多目标 RL 与训练基础设施

目标：支持工业级策略训练，而不只生成少量 SFT 轨迹。

优先级：P1  
建议周期：4 至 6 周

任务：

- 将 reward 拆分为独立向量：净 GMV、留存、打扰、退款、券成本、库存目标、真实性。
- 支持 scalarized reward、约束优化和 Pareto 分析。
- 明确区分环境 reward、规则违规和 LLM judge 指标。
- 加入 trajectory replay buffer、批量并行 rollout 和断点续跑。
- 输出适配 veRL、TRL 或其他训练框架的数据格式。
- 加入 SFT、DPO、在线 RL 和离线 RL 的统一实验配置。
- 记录训练和评测的数据 lineage、模型版本和 prompt 版本。

验收标准：

- 至少完成 SFT 与一种 RL 方法的端到端实验。
- RL 策略在隐藏测试集上优于对应 SFT 策略，而非仅提高训练 reward。
- 每个策略均能展示成本、用户体验和转化之间的权衡曲线。

## Phase 5：Feed-to-Shop 组合环境

目标：覆盖从内容发现到深度购物和成交的完整链路。

优先级：P2  
建议周期：3 至 5 周

新增动作：

```text
ask_clarifying_question(question)
handoff_to_shopping_agent(context)
resume_content_feed()
```

任务：

- 定义通用 `ShoppingSubEnv` 协议。
- 实现独立的搜索购物 mock 子环境。
- 在授权允许时实现 ShopSimulator adapter，而不复制其内部代码。
- 将视频上下文、用户意图和候选商品作为 handoff context。
- 评测是否在正确时机转交、转交信息是否完整以及总链路是否成功。
- 支持从搜索失败返回内容流继续探索。

验收标准：

- Agent 能在低意图时保持内容体验，在高意图时进入深度购物流程。
- 组合任务的奖励可以归因到意图判断、转交、搜索和最终购买各阶段。
- 子环境可替换，不与单个外部项目强耦合。

## 7. 数据建设规划

建议将数据分为四层：

### L1：程序化合成数据

- 用于协议测试、边界条件和大规模预训练。
- 必须提高因子组合数量，减少固定模板。
- 应保存生成参数和因果变量，便于反事实分析。

### L2：LLM 增强场景

- 使用 LLM 生成自然语言、模糊需求和用户回复。
- 结构约束和真实状态由程序控制，LLM 不负责决定事实。
- 所有 LLM 生成内容经过一致性验证和去重。

### L3：人工校验 Gold 数据

- 重点覆盖高风险、模糊和策略分歧场景。
- 记录允许多个合理动作的情况。
- 不要求保存私有思维链，保存简洁决策依据、证据和审核标签。

### L4：真实或半真实回放数据

- 使用脱敏统计和经过授权的行为数据校准用户模拟器。
- 保留 propensity、曝光机制和日志缺失信息。
- 不能直接把历史策略结果当作新策略的无偏标签。

## 8. 评测体系

### 8.1 核心结果指标

- Gross Purchase Rate
- Net Purchase Rate
- Return / Refund Rate
- Net GMV
- Coupon Cost per Net Purchase
- Session Retention
- Interruption Rate
- Grounded Recommendation Rate

### 8.2 Agent 能力指标

- Tool Call Validity
- Final Action Validity
- Coupon Truthfulness
- Category Alignment
- Product Grounding
- Intervention Timing
- Substitute Quality
- Evidence Sufficiency
- Handoff Precision / Recall

### 8.3 Benchmark 质量指标

- 动作类别分布。
- 场景难度分布。
- 模板和语义重复率。
- 模型间区分度。
- 多 seed 方差。
- reward hacking 检出率。
- train/test 泄漏率。
- 用户模拟器校准误差。

## 9. Harness 优化清单

### P0

- 隐藏 Gold 和 expected behavior。
- 任务 schema 与 observation schema 版本化。
- 环境转移和 reward 分量逐步记录。
- 数据集 hash、配置 hash、prompt hash 和模型标识。
- 多 seed benchmark 和置信区间。
- failure taxonomy 自动归类。

### P1

- 并行 vectorized environments。
- trajectory replay 和逐步确定性断言。
- 批量 provider 调用、限流和 token 成本统计。
- counterfactual action evaluation。
- LLM judge 只用于结构化规则无法覆盖的软指标。
- Web dashboard 展示轨迹、状态变化和失败证据。

### P2

- 分布式 rollout worker。
- 训练框架 adapter。
- 多 Agent 或多策略共享商业状态。
- 在线 shadow evaluation 接口。

## 10. 工程结构建议

```text
src/videoshop/
  envs/
    video_feed_env.py
    shopping_subenv.py
    composite_env.py
  worlds/
    user_world.py
    video_world.py
    commerce_world.py
  tools/
    retrieval.py
    coupon.py
    substitution.py
    explanation.py
    handoff.py
  rewards/
    components.py
    scalarization.py
    constraints.py
  datasets/
    generators/
    validators/
    splits/
  evaluation/
    metrics.py
    failure_taxonomy.py
    reports.py
  training/
    sft.py
    preference.py
    rollout.py
    adapters/
```

在近期版本中不必立刻重构成以上目录。只有当 Phase 1 的接口稳定后，再逐步迁移，避免为了目录整洁打断当前实验。

## 11. 建议发布节奏

### v0.2：Benchmark Ready

- 隐藏任务目标。
- 因子化场景生成。
- 冻结 split。
- 四档 baseline。
- 统一 benchmark report。

### v0.3：Video Native

- 视频时间轴 schema。
- 关键帧、ASR 和 OCR grounding。
- 时机相关动作与指标。
- 首个公开视频评测集。

### v0.4：Dynamic Commerce

- 动态库存、优惠券预算和价格。
- 跨会话用户状态。
- 延迟退款与长期奖励。
- 多目标策略评测。

### v0.5：Feed-to-Shop

- ShoppingSubEnv 接口。
- `handoff_to_shopping_agent`。
- 搜索购物 adapter。
- 端到端内容到成交 benchmark。

### v1.0：Training Platform

- 大规模并行 rollout。
- SFT + DPO + RL 基线。
- 稳定数据和环境版本。
- 完整论文级实验与复现文档。

## 12. 近期两周执行清单

第一周：

- 新增 `ScenarioSpec`，区分 Agent 可见信息和 evaluator 私有信息。
- 移除 synthetic objective 中的动作提示。
- 增加隐藏 Gold、场景难度和场景因子字段。
- 生成按实体隔离的 train/dev/test split。
- 增加 action distribution 和 per-scenario metrics。
- 冻结 v0.1 baseline 报告。

第二周：

- 实现因子化场景生成器 v1。
- 增加至少 10 类难负例和 5 类 no-action 场景。
- 扩展到 5,000 条训练任务和 1,000 条测试任务。
- 跑 random、rule-based、Qwen 和 DeepSeek 对比。
- 对失败轨迹自动归因并抽样人工审核。
- 发布 v0.2 benchmark schema 草案。

## 13. Go / No-Go 标准

在进入视频和 RL 大规模投入前，Phase 1 应满足：

- 测试任务无显式答案泄漏。
- 数据 split 无实体级泄漏。
- 至少四档策略表现有稳定梯度。
- 强模型仍有足够失败样本可供优化。
- reward 与实际成功指标具有正相关性。
- 增加合成数据规模不会显著提高重复率。
- 单条失败轨迹可以被完整重放和解释。

如果这些条件不满足，应继续修正任务与评测，不应急于扩大轨迹规模或开始昂贵 RL 训练。

## 14. 最终竞争壁垒

VideoShopAgent 的壁垒不应是商品数量，而应由以下能力共同构成：

1. 视频时间轴上的介入决策。
2. 潜在购买意图和用户长期状态建模。
3. 优惠券、库存、预算和退款组成的动态商业世界。
4. 可验证的 function calling 和证据约束。
5. 转化、用户体验、风险与成本的多目标优化。
6. 从内容发现到深度购物的组合式 Agent 环境。
7. 可重放、可审计并能直接用于 SFT、DPO 和 RL 的轨迹基础设施。

项目近期最重要的工作不是继续无上限生成轨迹，而是先让每条任务真正测量需要研究的能力。完成这一点后，扩大环境规模和训练规模才会产生有效收益。
