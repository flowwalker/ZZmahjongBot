# SL训练策略演进

## v1: 基础Loader

```
SLDataset → np.load逐文件读取 → DataLoader → 训练循环
数据按比例切分 train/val (begin/end参数)
```

最简实现。每个 `__getitem__` 调用 `np.load` 从磁盘读取单个 `.npz` 文件，然后索引到具体样本。

**性能瓶颈**: 每个样本一次磁盘 I/O + npz 解压，DataLoader 多进程下磁盘争抢严重。

---

## v2: int8 压缩Loader

相比 v1 的改进：

```
数据以 int8 存储 → np.load → 训练时转 float32
```

**核心变化**: 特征和 mask 以 `int8` 格式存储（而非 float32），磁盘占用减少 75%。训练时即时转换为 float32。

苦难是： 大规模数据集（数百万样本）的磁盘空间和 I/O 带宽压力。

---

## 🔥v3: 内存预加载

配合最后一个版本的dataset的良药。速度提升不止一点。

相比 v2 的改进：

```
MemPreloadDataset: __init__时一次性np.empty预分配 → 逐文件填充 → 连续内存块
__getitem__: O(1) 直接切片，零I/O
+ LazyAugSampler: 分组shuffle + 即时增强
```

**核心创新**:
1. **全量内存预加载**: 初始化时将全部 npz 拼接为连续 NumPy 数组，`__getitem__` 退化为纯内存切片
2. **LazyAugSampler**: 按 n_aug 个变换逐一 shuffle 原始索引（同 batch 样本使用同一变换），增强在取样时即时应用

**动机**: 消除逐样本的磁盘 I/O 瓶颈。内存预加载后，数据读取延迟从 ms 级降到 μs 级。代价是需要足够大的 RAM。

---


