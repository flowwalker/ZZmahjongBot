# 模型架构演进

> 11 个版本，CNN → CNN+SE → BiLSTM+SA → Dual Tower → Gated Fusion → Multi-branch → 回归 CNN-SE → Dual Light

经过多种架构的不严谨测试，最终得到的观点如下：

- cnnSE版本的训练速度快于lstm+attn版本，在数据增强后时间效益尤为明显
- lstm+attn版本的训练结果会优于cnn版本，优势主要在于能够注意并擅长大胡牌，且更不容易点炮，然而数据增强后训练时间真的是太长了
- 不做点小小的"注意力"，pureCnn是没有前景的
- ⚠️最终提交的bot的是兼顾时间和效果：cnnSE(160ch)+2x数据增强

---

以下 是各个版本的model架构具体说明：

## 一、纯 CNN 系列

### v1: 3层CNN (6ch)

```
CNNModel: Conv(6→64) → Conv(64→64) → Conv(64→32) → FC → 双头(Policy+Value)
输入: 6ch 特征, ~0.7M 参数
```

最简模型。三层卷积 + 全连接双头（策略+价值），无残差连接、无归一化层、无SE模块。训练实验的起点。

---

## 二、CNN + SE 系列

### v2: CNN-SE 4Block (16ch)

```
CoordConv → 4×ResBlock-SE(64ch) → reduce Conv(64→32) → 双头
Mish激活, GroupNorm(4)
输入: 16ch
```

**创新 — Squeeze-Excitation**: 首次引入 SE 模块，通过全局平均池化→FC→Sigmoid 学习通道间权重，让模型自适应关注重要特征通道。使用 CoordConv 替代普通 Conv2d 以保留空间位置信息。

### v3: CNN-SE Large (16ch)

```
CoordConv → 16×ResBlock-SE(256ch) → reduce(256→128) → 双头
输入: 16ch, ~29M 参数
```

相比 v2：参数量扩张约 3×（16个 SE-ResBlock，256ch 隐层），属于大模型路线探索。输入仍为 16ch。

实际效果并不更好。

### v8: CNN-SE Pyramid (160ch)

```
CoordConv → 3×ResBlock(256) → Trans(256→128) → 3×ResBlock(128) → Trans(128→64) 
→ 3×ResBlock(64) → Flatten(2304) → 双头
输入: 160ch, ~10.4M 参数
```

**创新 — 金字塔多尺度**: 采用三级金字塔结构（256ch→128ch→64ch），每级3个ResBlock，通过 Transition 层下采样。多尺度特征融合让模型同时捕捉全局和局部模式。

此架构成为最终主力，后续多个训练配置（预加载2x/12x/72x/288x）均基于此模型。是 CNN+SE 路线的收敛版本。

最主要的目的是：

- 省时间
- 其次说服自己，猜测他应当能够自己识别关键信息

---

## 三、BiLSTM + Self-Attention 系列

### v4: BiLSTM-SA (148ch)

```
CoordConv → 16×ResBlock(256) → Trans(256→128) → BiLSTM + SelfAttention(256,8头) → 双头
输入: 148ch
```

**创新 — 序列建模**: 首次将 BiLSTM + Self-Attention 引入，用双向 LSTM 捕捉弃牌序列的时序模式（112ch 弃牌历史），Self-Attention 层对 LSTM 输出做跨时间步注意力加权。

相比 CNN-SE：从空间卷积转向时序建模。SafeSelfAttention 机制防止数值溢出。

- 这个实测效果是非常不错的，奈何时间。

---

## 四、Dual Tower / Dual Path 系列

### v5: Dual Tower (224ch)（实验失败，发散，v头太难训）

```
CoordConv → Stem(→256) → 策略塔(12×SEResBlock(256)) // 价值塔(8×SEResBlock(256))
→ 融合输出
输入: 224ch
```

策略塔和价值塔各自拥有独立的残差块栈，共享 Stem 主干。让策略学习和价值估计各自使用不同的特征表示。

### v9: Dual Light (160ch)

```
共享CNN(160ch→64ch全通道) ┬→ BiLSTM分支(64ch) ┐
                          └→ CNN分支(32ch)    ─┤→ Transformer融合 → 策略头
输入: 160ch，无Value Head
```

**核心创新 — 双路异构融合**: 一路 BiLSTM 捕捉弃牌时序模式，一路 CNN 捕捉空间/通道模式，通过 Transformer Encoder 层融合两路特征。

相比 v5：大幅轻量化——共享CNN全压缩到 64ch，BiLSTM分支 64ch、CNN分支 32ch，Value Head 返回 dummy zero，策略头大幅缩减(4608→512→235)。

- 目的是为了让lstm尽量省时间用于数据增强，因为尝试训过正常版本的1个epoch，非常之久。
- 但是结果是：并没有显著优于cnnSE的数据增强，水平相当且似乎还更不稳定一点，但感觉实战或许会更强
- 是未来的目标理想架构（加大参数容量后）

### v10: Dual ZeroCNN (160ch)（未做实验）

```
金字塔 CNN(256→128→64, 各3×ResBlock) ┬→ BiLSTM(64→128→bidir→64) ┐
                                      └→ CNN分支(9×ResBlock64→32) ─┤→ Transformer → 双头
+ _zero_branch_weights()—— CNN分支权重置零
输入: 160ch, ~16.9M 参数
```

相比 v9：架构完全不同——采用金字塔 CNN 主干（三级 256→128→64），保留完整 Value Head，参数容量大 5 倍。**消融实验**——通过置零 CNN 分支权重，验证 CNN 分支对 Dual 架构的贡献度。

---

## 五、门控融合系列

### v6: Gated Fusion (224ch)（未做实验，最终以cross attention替代）

```
CoordConv → DeepStem(128→192→256→320) → 12×ResBlock(320) + BiLSTM(320→640)+SA
→ 门控融合(可学习alpha) + 辅助任务头
输入: 224ch
```

---

## 六、多分支交叉注意力系列

### v7: Multi-branch Cross-Attention (224ch)（完全失败，完全不收敛）

```
DeepStem(128→192) → 6×ResBlock(192) 空间分支 // BiLSTM(192,128)+SA(256,4头) 时间分支
→ CrossAttentionFusion(192,4头) → 双头
输入: 224ch
```

**核心创新 — 交叉注意力融合**: 空间分支（CNN）和时间分支（BiLSTM+SA）通过 Cross-Attention 互相关信息，让两个分支彼此"查询"对方学到的特征。与 v6 的门控融合策略不同（注意力 vs 门控权重）。

---

## 七、其他架构探索

### v11: CNN+Transformer (160ch)（未做实验）

```
Pyramid CNN(256→128→64, 3×ResBlock) + PositionEmbedding → 1×TransformerBlock(64,4头) → 双头
输入: 160ch
```

混合架构——CNN 金字塔提取多尺度空间特征 + Transformer 层做全局序列建模。与 Dual Path 中 Transformer 仅做融合不同，这里是端到端的 CNN→Transformer pipeline。

