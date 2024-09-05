# 长上下文问题相关论文和解决思路
## 问题定义：
* 现有的LLM多是基于transform结构而transform结构中SelfAtn计算复杂度与上下文的长度成二次方，
导致在处理长上下文需要大量显存和计算时间。
* LLM在推理时候需要用的kv-cache，长上下文也会导致需要的kv-cache数目快速增长，导致内存不足；
* transform 会有loss in the middle的问题，模型会更关注与开头和结尾部分的信息， 更容易忽略中间部分的信息；
