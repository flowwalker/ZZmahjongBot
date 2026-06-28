# Record

用内存预加载管线彻底解决了 v13 的 IO288x 灾难。

这也是在众多探索后，时间和金钱范围内实现的最优版本。

于是接下来的优化方向当然如下：

- 探索rand增强尽量见到288种
- 探索lstm+attn的2x版本->探索lstm+attn lite的2x版本
- 探索cnnSE的12x版本
- 探索上传后的自增强以弥补少数的损失

遗憾的是接下来，要么失败，要么边际效益递减。

## 重写数据管线：

- **v13 的 IO288x 灾难**：`preprocess_mp.py` 将 288× 增强物理落盘，存储爆炸 + I/O 瓶颈，连一个 checkpoint 都没跑出来
- **故采取方案**：原始数据一次性加载到内存（~33GB），`__getitem__` 中延迟执行增强变换

这是数据管线的决定性突破，后续所有 mem 系列版本（12x/72x/288x）都基于这套管线。

## 创新：内存预加载管线

### MemPreloadDataset (`dataset_mem.py`)
- `__init__`：一次性 `np.load` 所有 npz → 3 个连续大数组 (obs/mask/act)，~33GB int8
- `__getitem__`：纯内存索引 + int8 解码 + 延迟变换 — 零磁盘 I/O
- 虚拟长度 = n_aug × 原始样本数
- 加载时间 ~1-2 分钟 (SSD)

### LazyAugSampler
- 只 shuffle 原始索引（~5.87M），避免 `randperm(N*288)` 内存爆炸
- 对 n_aug 个变换逐一 shuffle 原始索引
- 每个 batch 包含 batch_size 个不同原始样本（同一变换下）
- 峰值内存 ~47MB vs 全量 randperm 的 6-54GB

### 分级增强 (`augment.py` 新增 `get_transforms_by_level`)
- 2×：数字对称 (1↔9, 2↔8, …)
- 12×：+ 花色全排列 (W/T/B)
- 72×：+ 箭牌全排列 (J1/J2/J3)（创新）
- 288×：+ 风位轮转 → 全量增强（创新）

默认 `N_AUG=2`（仅数字对称），对应目录名 "2x"。切换级别只需改 `sl_pretrain_mem.py` 中的 `N_AUG` 变量。

