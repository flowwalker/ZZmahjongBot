# Record

**失败版本**。v14 的简单扩展：`N_AUG` 从 2 提升到 12（数字对称 + 花色全排列 W/T/B）。

失败原因：学习率调节不当导致过拟合；若降低 lr 则训练时间翻 12 倍，没有足够的计算时间来搜索合适的学习率。

## 与 v14 差异

仅 `sl_pretrain_mem.py` 两处改动：
- `N_AUG = 2` → `N_AUG = 12`
- `no_improve = 0` → `no_improve = 0 if start_epoch == 0 else sum(...)` — 断点续传时 patience 计数修正

其余完全相同：agent/feature/model/augment/dataset_mem 哈希与 v14 一致。

## 架构

同 v14：SE Pyramid CNN（256→128→64，9×SEResBlock），160ch 输入，内存预加载管线。

## 权重

| 文件 | 说明 |
|------|------|
| model_mem_12x_ep1.pt | epoch 1，raw state_dict，90 keys |
| model_mem_12x_ep3.pt | epoch 3，raw state_dict，90 keys |

来源：`tmp/experiment/model_mem_12x_ep*.pt`

## 无 Botzone 部署

此版本未生成 `__main__.py` 和 `bot.zip`。
