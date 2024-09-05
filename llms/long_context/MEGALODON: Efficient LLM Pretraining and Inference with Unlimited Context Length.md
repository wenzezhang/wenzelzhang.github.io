# MEGALODON: Efficient LLM Pretraining and Inference with Unlimited Context Length
这篇论文是主要是由meta和cmu在2024年4月发表的，要解决的问题是transformer结构的**二次复杂性**以及**弱的长度外推能力**（weak length extrapolation），虽然现在已经有一些解决该问题的方法如：

* linear attention：一类降低传统注意力的二次复杂度的方法，侧重于通过一些核技巧来实现注意力计算的线性化：

  https://kexue.fm/archives/7546

* state space（如前阵子比较火的SSM模型，mamba模型）

但是这些方法在预训练效率和下游任务的效果上都低于原始的结构。这篇论文提出了一个适用于无限上下文长度并且能够有效训练的新结构；该结构主干是MEGA（exponential moving average with gated attention），然后在此之上提出了许多提升能力和稳定性的组件比如：**complex exponential moving average**，**timestep normalization layer**, **normalized attention mechanism** 以及 **pre-norm with two-hop residual configuration.**

<img src="/Users/pidan/Library/Application Support/typora-user-images/image-20240804161428656.png" alt="image-20240804161428656" style="zoom:50%;" />

* complex exponential moving average：将MEGA中的多维指数平均扩展到复数领域。
* timestep normalization：扩展group normalization layer（将channel分为许多组（group），对每一组做归一化），使其能够在序列维度进行归一化。

为了讲清楚这篇论文所用的方法，先来回顾一下MEGA结构：

使用$ X = {x_1, ... x_n} \in R^{n * d} $  和 $ Y = {y_1, ... y_n} \in R ^{n*d}$ 表示长度为n的输入和输出，首先使用一个$\beta \in R ^ {d *h}$的矩阵，将输入维度由d扩展到h,然后对于h维做EMA：

u (j) t = βjxt,j 

h (j) t = αj ⊙ u (j) t + (1 − αj ⊙ δj ) ⊙ h (j) t−1 (1) 

yt,j = η T j h (j) t

***

首先为了减少二次的复杂度，MEGA简单的将Q，K，V序列划分为长度为c的不同的chunk,每个chunk都应用Attention，复杂度$O(kc^{2}) = O(nc)$;EMA能够帮助捕获每个token附近的局部上下文信息，从而缓解在做chunk-wise-attention过程中位于**chunk边缘的token**上下文信息缺失的问题。该方法的主要问题有：

* 效果上低于full- attention主要是由于MEGA层EMA的表达能力不足；
* 对于不同的数据类型和任务MEGA中使用不同的归一化模式（p re, post）和注意力函数；
* 缺乏在大规模数据集上pre-train的实验；

方法：

* 将EMA扩展到复数领域增强EMA的表达能力；

主要是在h的计算上做了改动：
h_t^{j} = \alpha_j (cos \theta_j + i sin \theta _j) \dot u_t^{j} + ( 1- \alpha_j \dot δj) * (cos \theta_j + isin \theta_j) \dot h_{t-1}^{j}
y_{t,j} = Re(η^T_jh^(j)_t)

* 时间上的norm:
Layer normalization在处理空间维度（如时间步或序列维度）时，确实无法直接减少内部协变量偏移。Group Normalization 在时间维度和分组后的特征维度上进行norm；
通过计算累计mean和variance扩展Group Normalization；
* 注意力Norm的选择：
![img.png](img.png)
* Pre-Norm with Two-hop Residual：
