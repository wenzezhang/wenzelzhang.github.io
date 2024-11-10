# SPIRIT LM: Interleaved Spoken and Written Language Model

摘要：语音和文本序列连接为单个标记流，并使用小型自动管理的语音文本并行语料库通过单词级交错方法进行训练。基础版本使用speech phonetic units（HuBERT）, 高级版本(EXPRESSIVE)额外使用pitch和style units；

SPIRIT LM can learn new tasks in a few-shot fashion across modalities (i.e. ASR, TTS, Speech Classification).

**联合文本大模型的生成能力和知识以及语音的表达能力。**

使用expressive tokens扩展语音token，捕获pitch和style。

**Pitch Tokens：**使用VQ-VAE trained on the F0 of the input speech.For training the pitch quantizer, the F0 is extracted using pyaapt8 . However, for the language model training, we extract F0 using FCPE9 , a fast pitch estimator using Transformer, for inference speed. 12.5 pitch tokens per second.

**Style Tokens**：extract speechprop features from Duquenne et al. (2023), which capture speech input’s expressive style. The features were pooled with average pooling over input segments of 1 second, **making one feature every one second.**

Expressive Speech Tokenization：

mix the 3 types of tokens

解码器：1-hot speaker embedding from Expresso’s voices

字级别的对齐：

performing an alignment at the word level using the aligner tool from Pratap et al. (2023).
