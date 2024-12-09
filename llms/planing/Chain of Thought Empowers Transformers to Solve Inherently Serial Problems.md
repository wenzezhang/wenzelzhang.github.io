# Chain of Thought Empowers Transformers to Solve Inherently Serial Problems

摘要：这项工作通过表现力的视角，从理论上理解了 CoT 对于纯解码器 Transformer 的强大功能，从概念上讲，CoT 使模型能够执行固有的串行计算，而这是 Transformer 所缺乏的，尤其是在深度较低时。

This paper aims to study why the form of CoT improves the reasoning capability of LLMs. 本文的假设是COT允许transformer模型执行更多的串行计算（serial computations）。使用表达能力的长度来形式化和分析这个假设。

使用circuit complexity来讨论transformer的能力。transform更适合并行计算，但是串行计算受限与网络的深度。

we prove that a constant-precision transformer with T intermediate steps and embedding dimension logarithmic in the sequence length can express any functions computable by a circuit of size T.
