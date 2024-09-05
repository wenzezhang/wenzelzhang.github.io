# WAVTOKENIZER: AN EFFICIENT ACOUSTIC DISCRETE CODEC TOKENIZER FOR AUDIO LANGUAGE MODELING

codec:将高维度的自然信号转化为低维度的离散信号，与之前的模型相比：

extreme compression：24K采样率1s钟大概是40到75个tokens.

improved subjective quality: state-of-the-art reconstruction quality （UTMOS score）

主要改进：设计**更广泛的 VQ 空间**、**扩展的上下文窗口和改进的注意力网络**，以及引入**强大的多尺度鉴别器和逆傅里叶变换结构**。我们在语音、音频和音乐领域进行了广泛的重建实验。

---

大部分codec model的结构：encoder, Residual Vector Quantization, decoder.其中编码器在时间领域上下采样声音信号来获取压缩的音频帧。然后，每个压缩音频帧由一系列量化器量化，**每个量化器对前一个量化器的残差进行操作。**量化器的个数决定了整体比率。另一方面，解码器在时域中执行上采样，以根据量化器输出重建音频信号。

现代codec model的目标：

* 对不同音频信号的统一建模：speech, music, and audio.
* 重建质量：human-level reconstruction quality
* 压缩：the number of quantizers and the temporal dimension of the codec.
* 端到端结构：语义token和声学token。
* Rich Semantic Information： Exploring more elegant ways to integrate semantic information directly into the codec remains an open question

本文：a multi-scale discriminator，an inverse Fourier transform upsampling structure from the vocoder to the decoder

为了将多个量化器压缩到一个量化器，我们发现，简单地扩展 VQ 空间，再加上**最近的K-means聚类初始化和随机唤醒策略**，可以显着压缩音频表示，同时保持高码本利用率。此外，扩展语音建模的上下文窗口并将注意力网络纳入解码器不仅可以平衡重建质量和压缩，还可以丰富模型的语义信息(aligning the speech space with the textual vocabulary, suggesting its potential as a latent form of a unique language.)

---

we **used k-means ´ clustering to initialize the codebook vectors**. We adjusted the number of cluster centers to 200 to align with the larger codebook space.

每个输入的选定代码使用衰减为 0.99 的指数移动平均值进行更新，并且多个批次中未分配的代码将替换为从当前批次中随机采样的输入向量。这种强制激活策略
（Dhariwal et al., 2020）有助于确保大码本空间的有效利用。

**这表明有可能将语音与广泛的自然语言词汇结合起来，通过标记器将其强制映射为一种独特的语言。我们将在后续的实验中验证这一点。**

我们在所有深度保持一致的特征分辨率，通过傅里叶逆变换实现波形上采样

在解码过程中，X使用Short-Time Fourier Transformer来表示：
 Incorporating an attention network module into the decoder can enhance information reconstruction and semantic modeling.
expanding contextual modeling windows to three seconds for WavTokenizer with attention modules will further improve codec reconstruction during training.
 adding the attention module before the ConvNext module appears to be the optimal solution.

we use a multi-scale discriminator
(MSD) and a complex STFT discriminator (Zeghidour et al., 2021) at multiple time-scales (Defossez ´
et al., 2022). We adopt a hinge loss formulation instead of the least squares GAN objective, as
suggested by (Zeghidour et al., 2021).

the WavTokenizer model consists of four components: quantizer
loss, mel-spectrum reconstruction loss, adversarial loss, and feature matching loss.
***
实验：
训练数据：8W小时
LibriTTS，VCTK， CommonVoice， LibriLight
AudioSet
Jamendo， MusicDB
WavTokenizer-medium： 5000小时
WavTokenizer-small: 585小时
n. During the training phase, we uniformly truncated excessively
long segments in the training data to a fixed length of 10 seconds and subsequently performed a random crop of the waveform to obtain audio snippets of 3-second duration for feeding WavTokenizer.
