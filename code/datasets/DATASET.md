# 数据集加载演进

## 核心数据集

### v1: 二分查表

```
MahjongGBDataset
__init__: 读 count.json → 遍历全部 npz → 预加载到 self.cache (list-of-arrays)
__getitem__: bisect_right 查 match_id → 索引 match 内 offset → 纯内存 list 索引
```

最早的原型。`__init__` 一次性将所有 npz 文件加载进内存（以 match 为单位的 list-of-arrays），`__getitem__` 不再做磁盘 I/O。数据未做扁平化，每个 match 独立一个 array，内存布局不连续。

**使用**: bot v1_starter, v2_baseline

---

### v2: 连续拼接

相比 v1 的改进：

```
MahjongDataset
__init__: 遍历所有 npz → np.concatenate → 连续内存块 (all_obs/mask/act)
__getitem__: O(1) 直接切片 all_obs[idx], all_mask[idx], all_act[idx]
```

**核心变化**: 将离散的 match 级数组用 `np.concatenate` 拼接为三个连续大数组，`__getitem__` 退化为纯内存索引。消除了 v1 的 `bisect_right` 查表和逐样本 `np.load`。

原因是: v1 的每样本一次 `np.load` + `bisect_right` 在 DataLoader 多进程下磁盘争抢严重。

**使用**: bot v3-v6

---

### 🔥v3: 内存预加载 + 固定增强（最终确定的最终方案）

相比 v2 的改进：

```
MemPreloadDataset
__init__: 预分配 int8 数组 → 逐 match 填入 → 加载 level 对应的固定变换表
__getitem__: int8→float32 解码 + apply_transform → 增强样本

LazyAugSampler
按 n_aug 组 shuffle 原始索引 → 同 batch 样本来自同一变换
```

🔥创新**:
1. **内存预分配**: 一次性 `np.empty` 分配全量 int8 数组，逐 match 填入，避免 `np.concatenate` 的内存碎片
2. **延迟增强**: 变换在 `__getitem__` 时即时应用，不占额外磁盘/内存
3. **LazyAugSampler**: 对 n_aug 个变换逐一 shuffle 原始索引，峰值内存 ~47MB vs 全量 randperm 的 6-54GB

**动机**: v2 只是连续拼接，无增强能力。v3 引入内存预加载 + 延迟增强管线。

**使用**: bot v14-v16, v21-v22, v24

---

### v4: 每样本随机变换（大大的失败，要是batch里每个5x直接发散）

相比 v3 的改进：

```
per_sample_random=True
n_aug 强制为 1，每次 __getitem__ 从 288 随机取 1 个变换
+ reshuffle_transforms(): 每 epoch 重采样变换种子
```

**变化**: 从固定变换表（每个原始样本产出 n_aug 个确定增强版本）改为每个 epoch 每样本随机选一个变换。训练多样性提升，验证集仅用恒等变换保证跨 epoch 可比。

**使用**: bot v17fail

---

### v5: 分层采样（实际效果好像不咋样）

相比 v4 的改进：

```
stratified_random=True
n_aug=2，每样本确定性生成 2 个仅在花色排列维度不同的变换:
  ds(1/2) × sp(2/6) × hp(1/6) × wp(1/4)
+ _epoch_seed 每 epoch 递增，保证变换组合不同
```

**核心变化**: 从完全随机改为四维度分层——数字对称(ds)、花色排列(sp)、箭牌排列(hp)、风位旋转(wp)各占一个固定选择，唯独花色排列取 2 个不同值。保证每个 batch 的 2 个增强版本在最重要的花色维度上不同。

v4 的完全随机可能导致某些维度重复采样。分层采样保证四因子均匀覆盖，同时 n_aug=2 控制训练成本。

总之好像不太行
