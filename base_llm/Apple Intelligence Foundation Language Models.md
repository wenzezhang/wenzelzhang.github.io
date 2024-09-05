# Apple Intelligence Foundation Language Models
苹果7月份发布的大语言模型训练技术报告，里面介绍了基座模型结构、训练数据处理、训练过程、预训练，后训练对齐等技术细节；
模型体量较小3B左右的模型；
在苹果的技术日上提出了Apple Intelligence，里面包含了一系列的生成模型，支持写作和润色文本，总结邮件和信息，在聊天中创建图片（表情包），
还可以直接对话操作in-app的动作从而简化APP的操作。这篇论文主要是介绍了端侧的3B模型和云端的AFM服务。
* Empower users with intelligent tools
* Represent our users
* Design with care
* Protect privacy
如何构建高效且强大的基座模型？
如何训练这些模型？
如何训练adapter用来满足特征的用户需求？
如何评价模型的效果？
首先是模型结构的设计：
* 共享的输入输出编码矩阵，减少参数量；
* Pre-Normalization + RMSNorm 提升训练稳定性；
* Query/key normalization 提升训练稳定性；
* Grouped-query attention 使用8 key-value heads, 减少KV cache；
* SwiGLU activation
* RoPE 基础频率为500K
***
预训练：
**每一步都注重效率和数据质量，以便通过高效、低延迟的模型进行预训练，以获得高质量的端到端用户体验；**
数据：
多样和高质量的数据混合
**我们发现数据质量比数量更重要**，是下游模型性能的关键决定因素。下面我们提供
有关数据混合的关键组成部分的更多详细信息。
* 网页
使用Applebot爬取信息。另外，我们采取措施排除包含以下内容的页面：
亵渎并应用过滤器删除某些类别的个人身份信息 (PII)。然后，剩余的文档由执行质量过滤和纯文本提取的管道处理：
* Body 抽取使用 Safari’s reader mode and the Boilerpipe 算法；
* 使用启发式和基于模型的分类器进行安全和脏话过滤。
* 使用局部敏感的 n-gram 哈希进行全局模糊去重。
* 使用启发式和基于模型的分类器进行广泛的质量过滤
* 针对 811 个常见的预训练基准进行净化，根据与任何基准数据集的 4-13 gram 碰撞来过滤整个文档，除非给定 n-gram 的碰撞计数达到 1000 的“通用”阈值。
使用公开数据集，高质量长上下文的数据；
代码数据；
数学数据：Math QA 数据集 3B token + 14B token
Public datasets：去除个人信息；
vocabulary size is 100k and
49k tokens for AFM-server and AFM-on-device；
模型训练：
* 