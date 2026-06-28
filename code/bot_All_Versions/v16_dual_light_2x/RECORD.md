# Record

鉴于cnnSE的2x行之有效，首次创新性地引入双路并行架构（轻量化版）。

- 回到 CNN+BiLSTM 双路 + Transformer （多加一个全连接）融合。
- 轻量化的原因是：LSTM太慢了！不得不换小型来换速度

可惜的是因为参数量太小，acc呈现完美抛物线趋势：大致是82,83,85,86.x,83,80，只能选择中间86版本，然而似乎本地测试还是不太如cnnSE版本稳定，水平评测应当是相当的

## 架构：Dual Path Light（创新）

```
CoordConv(160+2→64) → 9×ResBlock(64) [共享CNN, 无金字塔]
  ├→ 上路: BiLSTM(64→128, bidir) → proj(256→64)
  └→ 下路: CNN Branch(32ch, zero-init, 9×ResBlock)
      ↓
Concat(64+32=96) → PosEmbed → 4×TransformerBlock(96, 4h) → Flatten(3456) → Head(512→235)
```

轻量化设计：
- 共享 CNN 全 64ch 平直（无 256→128→64 金字塔）
- 下路 CNN 32ch，权重零初始化 → 初始退化为 Mish(residual)

## 训练设计

- batch 内变换一致：LazyAugSampler 按 transform 分组，同 batch 4096 样本用同一变换，梯度不互相抵消
- 跨 epoch 一致：不 reshuffle transforms

