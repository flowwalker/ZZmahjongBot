# Record

加回 BiLSTM + SelfAttention 的小幅提升版本，相当于缩小参数量、提升特征数为155重训v6。

实验的结果是略有提升。

## 架构

```
CoordConv(155+2) → Stem(155→256, Conv-GN-Mish)
  Stage1: 8× SEBlock(256)       (Conv-GN-Mish-Conv-GN → SE → +skip → Mish)
  Trans:  256→128 (Conv-GN-Mish)
  Stage2: 8× SEBlock(128)
  BiLSTM(128→128, bidirectional) → 256
  SafeSelfAttention(256, 8 heads) → +residual → LayerNorm → Mish
  Flatten(36×256=9216) → FC(1024) → FC(235)
```

