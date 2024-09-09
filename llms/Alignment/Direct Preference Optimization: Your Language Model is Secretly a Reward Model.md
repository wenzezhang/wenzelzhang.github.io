# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

摘要：RLHF，复杂且不稳定，新的参数化奖励模型，在本文中，我们引入了 RLHF 中奖励模型的新参数化，可以以封闭形式提取相应的最优策略，从而让我们仅用简单的分类损失即可解决标准 RLHF 问题。在训练过程中也不需要从LM中进行采样。我们的实验表明，DPO 可以对 LM 进行微调，使其与人类偏好保持一致，效果与现有方法一样好甚至更好。值得注意的是，使用 DPO 进行微调在控制代际情绪的能力上超过了基于 PPO 的 RLHF，并且在总结和单轮对话中匹配或提高了响应质量，同时实施和训练起来要简单得多。

直观地看，DPO 更新增加了优选响应与非优选响应的相对对数概率，**但它包含一个动态的、每个示例的重要性权重**，可防止我们发现在简单的概率比目标下发生的模型退化。

我们的主要贡献是直接偏好优化 (DPO)，这是一种简单的非强化学习算法，用于根据偏好训练语言模型。我们的实验表明，对于使用最多 6B 个参数的语言模型进行情绪调节、总结和对话等任务中的偏好学习，DPO 至少与现有方法（包括基于 PPO 的 RLHF）一样有效。

**As we will describe next in detail, our key insight is to leverage an analytical mapping from reward functions to optimal policies, which enables us to transform a loss function over reward functions into a loss function over policies.**

潜在reward model:

Bradley-Terry:

Plackett-Luce ranking models: 适用于具有多个候选的情况

根据PPO目标得到最优策略p = f(reward)，变量替换将奖励函数表示为最优策略的函数 r = f(p)，优化奖励函数(Bradley-Terry model)其实就是约等于优化最优策略。max(r) = max(f(p)) = max(p);

* 根据PPO目标得到最优策略
