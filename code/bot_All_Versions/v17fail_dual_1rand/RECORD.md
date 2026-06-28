# Record

**失败版本**。acc 卡在 ~83%，上不去。

## 失败原因

分析：尽管见得多（288 种变换），但 batch 内/跨 batch 不一致导致学习不稳定：

- **per_sample_random**：每个样本每次 `__getitem__` 从 288 变换中随机取 1 个
- 同 batch 内 4096 个样本各自对应不同变换 → 梯度方向互相冲突
- 每个 epoch 同一原始样本看到不同变换 → 模型无法学到稳定的"样本→动作"映射
- 对比 v16：batch 内变换一致（同一变换下 4096 个样本梯度同向叠加），跨 epoch 也一致

## 架构

同 dual_path：共享 CNN 金字塔（256→128→64，9×SEResBlock）→ BiLSTM // CNN Branch → concat[64+64=128] → PosEmbed → SelfAttn → FFN → Head(4608→235)，含 Value Head。

略有调整：

| | v16 (dual_light) | v17 (dual_1rand) |
|---|---|---|
| 共享 CNN | 64ch 平直 | 256→128→64 金字塔 |
| 增强策略 | 固定 2 变换，batch 内一致 | 每样本随机 288 取 1 |
| 参数量 | 14MB | 64MB |
| 结果 | acc ~86.x% | acc ~83% |



