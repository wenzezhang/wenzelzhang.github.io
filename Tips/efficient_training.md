* [通过打包 Flash Attention 来提升 Hugging Face 训练效率](https://mp.weixin.qq.com/s/viC6k9B5xiSiOctdnRf87Q)
  提供了一个更加高效的新的数据整理器 `DataCollatorWithFlattenin`可以无缝地将序列连接成一个单一的张量，同时在 Flash Attention 2 计算过程中考虑到序列边界。**训练时候的吞吐量能够提高两倍**
