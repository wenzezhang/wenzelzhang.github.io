# Scaling Speech-Text Pre-training with Synthetic Interleaved Data

摘要：语音语言模型 （SpeechLM） 接受语音输入并生成语音输出，与基于文本的大型语言模型 （LLM） 相比，可实现更自然的人机交互。开发 SpeechLM 的传统方法受到无监督语音数据和并行语音文本数据可用性有限的限制，这些数据的丰富性明显低于文本预训练数据，从而限制了它们作为 LLM 的可扩展性。我们提出了一种新的方法来扩展语音文本预训练，方法是利用来自文本语料库的大规模合成交错数据，**无需并行语音-文本数据集。**我们的方法通过从现有文本语料库中采样文本跨度并使用文本到标记模型合成相应的语音跨度来有效地**构建语音-文本交错数据，而无需生成实际语音。**我们还通过将向量量化瓶颈合并到编码器中，采用了**从自动语音识别 （ASR） 模型派生的监督语音分词器**。这种监督训练方法即使在较低的采样率（例如 12.5Hz）下也能产生具有很强语义保留的离散语音标记，同时仍保持语音重建质量。从预训练语言模型开始，将我们的预训练扩展到 1 万亿个标记（具有 600B 合成交错语音文本数据），我们在语音语言建模和口语问答方面实现了最先进的性能，将口语问题任务的性能从之前的 SOTA （Moshi） 提高到 31%。我们进一步证明，通过**使用语音对话数据微调预训练模型，**我们可以开发一个端到端的语音聊天机器人，在对话能力和语音质量方面实现与现有基线相当的有竞争力的性能，甚至仅在语音领域运行。

---

a key challenge remains: the scarcity of speech data compared to text data.

提出了一个新的可以scaling speech-text预训练的方法，从文本语料中合成语音。将语料中的文本片段使用**text-to-token**模型转换成speech- token。

* 使用ASR数据以监督的方式训练一个tokenizer，采样率12.5HZ
* 使用TTS数据来训练一个text-to-token model，用来生成文本语音交错数据集

贡献：

* 提出了一个新的合成文本-语音交错数据的方法
* 设计了一个Speech LM的结构，使用12.5HZ采样率的codebook,加上一个flow-matching的解码器。
* scaling 预训练至1T token。
* 在对话数据上微调，搭建了一个端到端的对话模型。

---

方法：

* a supervised speech tokenizer
* a technique for synthesizing interleaved speech-text data
* a two-stage training process to extend pre-trained language models to the speech domain

**Supervised Speech Tokenizer：**

基于训练好的ASR模型在encoder的中间层上添加额外的pooling层和VQ层。

pooling 层：1D的平均pooling。

random restart trick，commitment loss，EMA更新

adapt the Whisper architecture to support streaming inference，将encoder层之前的卷积层替换成causal convolution layer，

replace the bidirectional attention in the encoder with block causal attention。

**Speech Decoder**：follow the decoder architecture of CosyVoice.

---

SYNTHESIZE INTERLEAVED SPEECH-TEXT DATA:

动机：that training on interleaved speech-text data encourages the model to learn an alignment between speech and text, facilitating the transfer of text-based knowledge to speech representations.

训练一个Text-to-Token Model：

To prepare the training data, we first tokenize speech from text-to-speech datasets into discrete speech tokens. The text-to-token model is then trained to predict these speech token sequences based on the input text.

交错数据构建（Interleaved Data Construction）：

Span lengths are drawn continuously from a Poisson distribution (λ = 10) until the total length of selected spans reaches the predefined ratio η of the original text length. Next, text spans corresponding to the drawn lengths are randomly selected from the document. These spans are converted into speech tokens using the text-to-token model, producing an interleaved speech-text sequence.η to 0.3 for optimal performance.
