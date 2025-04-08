# Gemma 3 Technical Report

我们推出了Gemma 3，这是Gemma系列轻量级开源模型中的一个多模态新成员，其参数规模在10亿到270亿之间。这个版本引入了视觉理解能力，覆盖了更广泛的语言范围，并且具备更长的上下文处理能力——至少可处理128K个标记。我们还对模型架构进行了改进，以减少在处理长上下文时容易激增的键值（KV）缓存内存。这是通过**增加局部注意力层与全局注意力层的比例**，并保持局部注意力的跨度较短来实现的。

**Gemma 3系列模型采用了蒸馏方法进行训练，无论是预训练版本还是指令微调版本，其性能都优于Gemma 2。**特别是，我们全新的**训练后优化方案**显著提升了模型在数学运算、对话、指令跟随以及多语言处理方面的能力，使得Gemma 3-4B-IT的性能可与Gemma 2-27B-IT相媲美，而Gemma 3-27B-IT在各项基准测试中的表现与Gemini-1.5-Pro相当。我们将把所有这些模型向社区开放。

多模态：most Gemma 3 models are compatible with a tailored version of the SigLIP vision encoder，The language models treat images as a sequence of soft tokens encoded by SigLIP.语言模型将图像视为由SigLIP编码的一系列软标记。我们通过将视觉嵌入压缩为固定大小的256维向量，降低了图像处理的推理成本。该编码器在固定分辨率下工作，并且我们从LLaVA（刘等人，2024年）中获得灵感，采用平移扫描（P&S）方法来实现灵活的分辨率。

LLaVA论文中通过LLaVA - UHD模型实现灵活分辨率，主要包括以下几个关键方法：

- \*\*图像模块化策略\*\*：将原始分辨率的图像划分为较小的可变大小切片。确定图像切分规则时，给定图像分辨率为\\((W\_I, H\_I)\\)，ViT预训练使用的分辨率为\\((W\_v, H\_v)\\)，通过定义一个得分函数来衡量切分与标准ViT的偏差，选择最合适的切分行列，以实现对原始分辨率图像的完全适应，避免填充或形状扭曲调整。采用任意纵横比切片编码，按照划分策略给出的纵横比对图像切片进行编码，而非使用固定分辨率。先按比例调整原始图像大小，使其符合划分策略的纵横比，让resize后的图像patch数量和ViT训练时一样，再通过二维插值对重塑后的二维位置嵌入进行插值，以适应划分策略给定的切片分辨率，用于视觉编码。

- \*\*压缩模块\*\*：使用共享的感知器重采样层对每个图像切片的视觉标记进行压缩。视觉编码器输出的图像标记通过跨注意力机制，使用一组查询向量被重采样到更少的数量，如在实验中从576减少到64。这样无论图像分辨率如何，都能保持固定且可承受数量的视觉标记，适合高分辨率图像理解，有效降低了LLM处理视觉标记的计算成本。

- \*\*空间模式\*\*：设计了一种空间模式来组织切片标记，以告知LLM图像切片在图像中的位置。通过两个特殊标记来传达图像切片的相对位置，这种简单的模式可以有效地传达动态分区，让LLM能够理解图像切片的空间组织，从而更好地处理不同分辨率和纵横比的图像。

上下文长度扩展到128K而不损失精度，local layer 1024个token，每1个global layer对应5个local layer。

---

模型结构：

* GQA+(post,pre, RMSnorm) we replace the soft-capping of Gemma 2 with QK-norm.
* a pattern of 5 local layers for every global layer, starting with a local layer as the first layer of the model.
* Gemma 3系列模型支持128K个标记的上下文长度，但参数为10亿的模型除外，该模型的上下文长度为32K。我们将全局自注意力层中旋转位置嵌入（RoPE）的基础频率从1万提高到了100万，同时将局部层的频率保持在1万。我们采用了与陈等人（2023年）提出的位置插值法类似的过程，来扩展全局自注意力层的作用范围。

---

pretrain:

a similar recipe as in Gemma 2 for pre-training with knowledge distillation.

**Distillation**. We sample 256 logits per token, weighted by teacher probabilities. The student learns the teacher’s distribution within these samples via cross-entropy loss. The teacher’s target distribution is set to zero probability for nonsampled logits, and renormalized.

通过Quantization Aware Training：构建量化版本的模型（训练5000步）

For the vision encoder, we pre-compute the embeddings for each image and directly train with the embeddings, adding no cost to the training of the language models.

---

Small versus large teacher：

We train a student with 2 teachers of different sizes, one large and one small, for different training horizons.
