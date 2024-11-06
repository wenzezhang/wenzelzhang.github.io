# Accelerating Large Language Model Pretraining via LFR Pedagogy: Learn, Focus, and Review

摘要：语言模型的预训练依赖于从网络数据中随机选择数据块进行自回归的预测学习。但这种做法是低效的，作者从人类学习的；

我们从人类学习技术（例如**间隔重复**）中汲取灵感，假设LLM的随机数据采样会导致高训练成本和低质量模型，而这些模型往往会忘记数据。为了有效地将网络规模的信息提交到长期记忆中，我们提出了 LFR（**学习、聚焦和回顾**）教学法，这是一种新的动态训练范式，它基于模型的学习，**以系统间隔聚焦并重复审查复杂的数据块步伐和进展。**

**LFR记录了不同数据块的模型困惑度，并经常重新访问困惑度较高且更容易被遗忘的块。**

记录loss较高的训练数据，更容易遗忘的数据；

使用LFR从网络数据中训练了124M到1.5B的GPT-2模型，在语言模型、QA、翻译以及problem solving等领域上与open AI的baseline模型相比都取得了更低的困惑度和更高的精度，**并且有20倍的训练加速**

**核心问题：**significantly reduce training costs while retaining or improving downstream task performance.

对于信息量较大的数据，模型应该多学。

核心贡献：

* Profile LLM pretraining to observe multiple descent behaviour for 25% of the training tokens that are forgotten multiple times.（观察发现25%的训练token被多次遗忘）
* 开发学习焦点复习（LFR）训练方案，利用困惑度来衡量LLM的学习进度，重点关注复杂的数据块，同时定期检查所有数据块以防止遗忘。
* 在AMD GPU上使用OpenWebText数据集进行预训练并且在6个下游任务上进行评估。
* 更低的困惑度，更低的精度，同时20倍以上的训练提速。
* 观察发现LLM在长期记忆中保持事实和指令信息之前首先学习对话和anecdotal数据；
* 证明文本重要性随训练时间和模型大小的变化而变化，从而推动了对 LFR 等动态数据选择方法的需求。

---

现有工作：primarily focused on using distance metrics and clustering techniques。

缺点：

1.First, they require pretrained models for calculating the distance metric on embeddings.

2.clustering-based methods do not scale to encompass the diversity of data in web-scale datasets.

3.they are slow and can take as long as 2-3 hours to generate 6GB subsets of the OpenWebText dataset (38GB)

---

分析：

记录一个训练block的PPL，训练8个epoch那么一个样本就有8个PPL；

* learned：该样本的PPL持续降低
* unlearned: PPL单调上涨
* Forgotten: PPL先降后涨，并且周期重复多次

observe that 25% of data blocks are forgotten at least once during training and that of the data blocks that are forgotten, 82% are forgotten multiple times during training, i.e., they display multiple descent behavior

---

方法：

**间隔重复**是一种基于证据的学习方法，被证明可以提高人类的信息保留和学习速度；

在这种技术中，具有挑战性的信息会被更频繁地、定期地复习，而较容易的信息则很少会呈现给学习者。我们通过以下三个步骤的组合来预训练我们的模型：

* learn:初始，使用整个数据集随机选择data block for p1 epoch,在这P1个epoch中记录所有data block 的困惑度。
* Focus: 丢弃s%的最低困惑度数据，在余下的数据中随机选择block进行训练P2个epoch；
* Review：接下来，我们重新引入所有删除的数据块，并在整个语料库上训练模型 p3 epoch（可重新配置）。

实验配置：

1. Phase 1: Learn for 1 epoch (p1 = 1).
2. Phase 2: Focus on 50% of the data for 1 epoch (s1 = 50, p2 = 1).
3. Phase 3: Review the entire dataset for another epoch (p3 = 1).
4. Phase 4: Focus on 30% of the data for 5 epochs (reps = 2, s2 = 70, p4 = 1).

---
