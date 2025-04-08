# Exploring Data Scaling Trends and Effects in Reinforcement Learning from Human Feedback

摘要：In this paper, we address this gap by systematically exploring data-driven bottlenecks that currently hinder RLHF performance scaling, focusing specifically on the challenges posed by reward hacking and decreasing response diversity. (探索RLHF中的数据驱动performance scaling，聚焦于reward hacking和回复多样性的挑战)，

针对reward hacking， 提出了一个混合奖励系统结合reasoning task verifiers 和 generative reward model，这种方法不仅表现出更强的抵御奖励操控的能力，还能够依据明确定义的真实解决方案，对回答进行准确评估。

为了确保回答的多样性并提高学习效果，我们提出了一种名为预近端策略优化（Pre-PPO）的新型提示选择方法，该方法能明确识别出本质上**具有挑战性、因而不太容易受到奖励操控**的训练提示。

此外，我们发现，**在基于人类反馈的强化学习（RLHF）训练的早期阶段，优先处理数学和编码任务能显著提升性能**，因为这些任务自然地对细粒度的回答差异进行了编码，并且具有明确定义的真实答案。通过在两种模型规模上进行的全面实验，我们验证了所提出方法的有效性和可扩展性。

结果表明，相对RTV对奖励hacking的抵御能力最强，其次是具有真实答案的生成式奖励模型（GenRM），最后是依赖于监督微调（SFT）的 “N 选优” 回答的生成式奖励模型（GenRM）。此外，我们提出的策略使模型能够快速捕捉特定任务的细微差异，从而大幅提升基于人类反馈的强化学习（RLHF）的整体性能。

**这项工作强调了精心构建数据的重要性，并提供了切实可行的方法，以克服基于人类反馈的强化学习（RLHF）中关键的性能障碍。**

**主要学习回答之间粗粒度差异的模型往往会迅速丧失回答的多样性，从而忽略了有价值的细粒度差异。** 为了克服这一局限，我们引入了一种创新的预近端策略优化（Pre-PPO）提示选择方法。该方法明确针对那些对模型而言具有更大学习挑战的提示，从而实现更稳健、更有效的数据扩展。后续分析表明，这些经过策略性选择的提示包含了丰富的细粒度回答差异。

**RLHF performance scale analysis**：

* RLHF exhibits superior generalization compared to Supervised Fine-Tuning (SFT) on novel inputs, especially as the distribution shift between training and testing data increases.
* RLHF significantly reduces output diversity compared to SFT across various metrics, suggesting a fundamental trade-off between generalization and diversity in current LLM fine-tuning approaches.

---

新收集提示的奖励分析。我们通过分析新收集提示的奖励分数，探究了为什么这些新提示未能提升基于人类反馈的强化学习（RLHF）的性能。如图3所示，在0到1的评分范围内，大约90%的这些提示获得了高于0.5的奖励分数。在这种评分分布中，0.5分表示模型的输出与参考相当，而高于0.5分则表明性能更优。

我们的生成式奖励模型（GenRM）经过训练，在推理任务中会将模型的回答与真实答案进行比较，在其他任务中则与监督微调（SFT）的 “N选优” 回答进行比较。因此，高于0.5的分数意味着模型生成的输出被判定为优于这些假定的最优回答。然而，经过仔细的人工检查，我们发现这些高分数输出中有很大一部分表现出了奖励操控行为，并且在质量上比最初精心挑选的回答更差。此外，我们观察到奖励分数的高低与奖励操控情况的严重程度和出现频率之间存在直接关联。奖励分数越高，奖励操控问题就越严重且出现得越频繁。这一发现揭示了我们当前奖励模型的一个关键局限性，并强调了需要更强大的评估指标，以便能够有效区分真正的性能提升和奖励操控的情况。

为基于近端策略优化（PPO）的强化学习训练选择奖励模型分数较低的提示。鉴于上述观察结果，我们设计了一种名为预近端策略优化（Pre-PPO）的选择算法，该算法能明确识别出奖励模型分数较低的提示，以供在初始的近端策略优化（PPO）实验中使用。这些低分数提示对模型来说更具学习挑战性，同时也更不容易受到奖励操控的影响。
