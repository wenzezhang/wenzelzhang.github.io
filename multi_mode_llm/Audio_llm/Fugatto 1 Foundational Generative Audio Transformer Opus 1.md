# Fugatto 1 Foundational Generative Audio Transformer Opus 1

Fugatto 是一种多功能音频合成和转换模型，能够遵循带有可选音频输入的自由格式文本指令。虽然在简单的下一个标记预测目标上使用文本进行训练的大型语言模型 (LLM) 可以学习直接从数据中推断指令，但仅在音频数据上训练的模型缺乏这种能力。这是因为音频数据本身并不包含用于生成它的指令。为了克服这一挑战，**我们引入了一种专门的数据集生成方法，该方法针对生成各种音频生成和转换任务进行了优化，确保数据揭示音频和语言之间有意义的关系。**另一个挑战在于仅使用数据来实现组合能力，例如组合、插入指令或否定指令。为了解决这个问题，我们提出了 ComposableART，这是一种推理时间技术，可将无分类器指导扩展到组合指导。它可以实现指令的无缝且灵活的组合，从而在训练分发之外实现高度可定制的音频输出。我们对一系列不同任务的评估表明，Fugatto 的性能与专用模型相比具有竞争力，而 ComposableART 则增强了其声音调色板和对合成的控制。最值得注意的是，我们强调了我们的框架合成紧急声音的能力——超越传统音频的声音现象——释放新的创意可能性。

ComposableART:通过潜在空间操作（包括来自不同模型）来合成指令的推理方法，称为可合成音频表示变换;

multifaceted data and instruction generation strategy:

* 首先使用LLMs来生成和增强指令以及标题（providing Fugatto with instructions that are closer to free-form instructions）。
* 开发提供绝对或者相对的指令（比如：合成一个开心的声音，提升这个声音的开心程度）
* 使用音频理解模型为音频片段生成描述和标题。
* 我们转换现有数据集以发现新的关系，从而无需额外的原始数据即可创建全新的任务。
* 我们使用音频处理工具在文本、音频及其相应的转换之间创建新的连接。

尽管数据很重要：compose, interpolate between, and negate instructions is generally difficult to obtain through data alone. 提出了ComposableART。

* I – Generating Free-Form Instructions with LLMs

  * 让LLM去生成一个指令生成器（python 函数）：不同长度，不同角色（standard, young-crowd, thirty-somethings, professional），任务指定（音频描述，语言或者其他等），每个角色都有一组动词、副词、连接器和其他资产来创建指令。
* II – Generating Absolute and Relative Instructions

  * 创建此类数据集后，我们可以选择一个样本并为其创建绝对指令，或者选择两个样本并创建相对指令。与绝对指令类似，我们使用由 LLM 生成的 python 方法生成相对指令，该方法根据任务、要修改的属性以及应如何修改它创建一条指令。
* III – Generating Audio Descriptions with Audio Understanding Models

  * For speech data, we implemented a prompt generation pipeline that automates the creation of natural language descriptions for voices. The pipeline converts speech attributes predicted by models – such as “gender”, emotion, and speech quality – into detailed natural language descriptions using LLM-generated templates. These templates describe voices in various ways based on the speaker attributes, enhancing diversity by generating descriptions in multiple formats.
* IV – Creating New Tasks and Datasets by Transmuting Datasets

  * 我们利用数据集中样本之间的隐式关系来启用新任务。一般来说，我们寻找一个因素保持不变而其他因素发生变化的数据集。例如，我们利用情感语音合成数据集以及同一说话者对同一文本的不同演绎（Livingstone & Russo，2018）来定义语音转换任务。类似地，我们利用具有相同音符的不同演绎的乐器合成数据集（Engel 等人，2017）来定义乐器转换任务。我们还利用提供声音混合各个部分的数据集（Rafii et al., 2017）来支持源分离等任务，以及根据音频上下文和字幕（可能是合成的）生成音频。
* Creating New Tasks and Datasets by Leveraging Audio Processing Tools:

  * 我们使用 Praat (Boersma & Van Heuven, 2001) 和 Pedalboard (Spotify, 2024) 来操纵多个语音和音频因素，从而创建语音和音频的合成配对数据。对于每个因素，我们应用受控修改，使我们能够生成具有特定更改的语音和音频样本。通过这种策略我们可以创建语音和音频对来描述语音和音频转换任务，例如改变一个或多个语音因素，对于每个因素，我们确定一个实际的调整范围并定义对应于不同变化程度的增量，例如分别为“轻微”、“中等”和“显着”。
