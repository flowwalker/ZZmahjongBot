# code/ — 国标麻将 AI 实验代码库

## 目录结构

```
code/
├── bot_All_Versions/    ★ 24 个 bot 版本的原始代码（每个含 feature/model/agent 等）
├── features/            特征工程演进 (6 .py)
├── models/              模型架构演进 (11 .py)
├── preprocess_s/        数据预处理演进 (4 .py)
├── datasets/            数据集加载演进 (5 .py)
├── sl_pretrains/        SL 训练脚本演进 (3 .py)
├── augments/            数据增强模块演进 (4 .py)
├── mains/               Botzone 部署脚本演进 (3 .py)
├── rl_failure/          RL 强化学习失败尝试 (14 .py)
├── tools/               ★ 实用工具 (5 .py)
└── README.md            本文件
```

---

## 各目录详情

### bot_All_Versions/ — 24 个 bot 版本

按时间顺序的完整 bot 代码，每个目录包含该版本的 `feature.py`, `model.py`, `agent.py` 及相关的训练/预处理脚本。

| v# | 目录名 | 架构 | 说明 |
|----|--------|------|------|
| 1 | v1_starter | 3层CNN, 6ch | 起点 |
| 2 | v2_baseline | 3层CNN, 6ch | baseline |
| 3 | v3_cnn_Res_Mish_SE_standard | CNN-SE, 16ch | 引入 SE |
| 4 | v4_cnn_Res_Mish_SE_Large | CNN-SE Large, 16ch | 大模型探索 |
| 5 | v5_standard_nn_FlatMC_16ch | CNN, 16ch | FlatMC |
| 6 | v6_to_148ch_add_lstm_attn | BiLSTM-SA, 148ch | 引入 LSTM |
| 7 | v7fail_to_224ch_adjust_lstm_attn | BiLSTM-SA, 224ch | 失败 |
| 8 | v8fail_dual_tower | Dual Tower, 224ch | 失败 |
| 9 | v9fail_gated_fusion | Gated Fusion, 224ch | 失败 |
| 10 | v10fail_cross_attn | Multi-branch Cross-Attn, 224ch | 失败 |
| 11 | v11_to155ch_fromv6_return_pure_cnn | Pure CNN, 155ch | 回归 CNN |
| 12 | v12_to155ch | Pure CNN, 155ch | |
| 13 | v13fail_to160ch_return_cnnSE_IO288x | CNN-SE, 160ch | 288x I/O |
| 14 | v14best_cnnSE_preload2x | CNN-SE, 160ch | ✅ 2x 预加载 |
| 15 | v15fail_cnnSE_preload12x | CNN-SE, 160ch | 12x |
| 16 | v16_dual_light_2x | Dual Light, 160ch | 轻量双路 |
| 17 | v17fail_dual_1rand | Dual, 160ch | 1rand |
| 18 | v18fail_dual_5x_stratified_zerocnn | Dual ZeroCNN, 160ch | 消融 |
| 19 | v19best_cnnSE_preload2x_3vote | CNN-SE, 160ch | ✅ 3模型投票 |
| 20 | v20best_cnnSE_preload2x_3vote50x | CNN-SE, 160ch | ✅ 投票+TTA |
| 21 | v21_no_time_cnnSE_preload72x | CNN-SE, 160ch | 72x |
| 22 | v22_no_time_cnnSE_preload288x | CNN-SE, 160ch | 288x |
| 23 | v23f_no_time_cnn_transformer | CNN+Transformer, 160ch | 混合架构 |
| 24 | v24_dual_full_2x | Dual Full, 160ch | 双路完整版 |

### features/ — 特征工程 (6 版)

`6ch → 16ch → 148ch → 224ch(失败) → 155ch → 160ch` 的通道膨胀与收敛历程。详见 `FEATURE.md`。

### models/ — 模型架构 (11 版)

`CNN → CNN+SE → BiLSTM+SA → Dual Tower → Gated Fusion → Multi-branch Cross-Attn → CNN-SE(Pyramid) → Dual Light → Dual ZeroCNN → CNN+Transformer`。详见 `MODEL.md`。

### preprocess_s/ — 数据预处理 (4 版)

`单线程 → 多进程(虚拟切分) → 增强内嵌(失败) → raw_only 多进程`。详见 `PREPROCESS.md`。

### datasets/ — 数据集加载 (5 版)

`二分查表(list-of-arrays) → np.concatenate 扁平化 → 内存预加载+延迟增强 → 每样本随机变换 → 四维度分层采样`。详见 `DATASET.md`。

### sl_pretrains/ — SL 训练脚本 (3 版)

`逐文件 np.load I/O → int8 压缩+解码 → 内存预加载 MemPreloadDataset`。详见 `SL_TRAIN.md`。

### augments/ — 数据增强 (4 版)

288-fold 群论结构增强：`基础 288 变换表 → 随机采样 → 分层采样`。详见 `AUGMENT.md`。

### mains/ — Botzone 部署 (3 版)

`单模型 argmax → 3 组同架构权重全局平均投票 + 100× TTA → 投票 + 50× TTA (5s 预算)`。详见 `MAIN.md`。

### rl_failure/ — RL 失败记录 (14 .py + NOTE.md + run.sh)

4 代 PPO 演进 + 5 个失败模式分析。详见 `NOTE.md`。

---

## 工具使用 (tools/)

```
tools/
├── tournament.py         ★ 多 bot 对战锦标赛
├── validate_version.py   验证 bot 版本能否正常预处理+训练
├── export_weights.py     权重导出 → Botzone 格式
├── data_test.txt         测试用数据 (前 5 局)
└── baseline_engine/      游戏引擎 (MahjongGBEnv)
```

### tournament.py — 对战锦标赛

从 `bot_All_Versions` 动态加载任意 bot 对战，自动检测 `.pt` 权重文件。

```bash
cd code/tools

# 列出所有可用 bot
python3 tournament.py --list

# 4 个不同 bot 对战 200 局
python3 tournament.py -n 200 v1_starter v2_baseline v14best_cnnSE_preload2x v16_dual_light_2x

# 同模型四家混战（测试随机性）
python3 tournament.py v14best v14best v14best v14best

# 支持模糊匹配
python3 tournament.py v3 v4 v14 v16
```

输出：每 20 局实时排名 + 最终胜率/场均得分/🥇🥈🥉分布。

（但是，至今有个随机性不足的bug未完全修复，本人大多是开多终端进行尝试的）

### validate_version.py — 版本验证

验证指定 bot 能否完成 预处理 → 数据加载 → 小批次训练 全流程。

```bash
cd code/tools

# 验证单个版本
python3 validate_version.py v1_starter

# 验证所有版本（批量）
python3 -c "
from pathlib import Path
import subprocess, sys
bot_dir = Path('../bot_All_Versions')
for d in sorted(bot_dir.iterdir()):
    if d.is_dir() and list(d.glob('*.py')):
        r = subprocess.run([sys.executable, 'validate_version.py', d.name], capture_output=True, text=True)
        print(f'{d.name}: {\"✅\" if r.returncode==0 else \"❌\"}')"
```

注意：这里仅有human文件夹无feature文件无法通过，因为基于human规则只需上传单文件main即可。

### export_weights.py — 权重导出

将 PyTorch checkpoint 导出为 Botzone 可加载的纯权重文件。

---

## 三大主要的创新点

1. **288-fold 分层数据增强**: 数字对称(2) × 花色排列(6) × 箭牌排列(6) × 风位旋转(4) — 群论结构，分层金字塔 (2×/12×/72×/288×)
2. **Dual Path 双路架构**: BiLSTM(时序) // CNN(空间) + Transformer 融合，完整版 16.86M，轻量版 3.56M
3. **测试时增强 + 多模型投票**: TTA 随机采样 + 5s 预算控制 + 全局平均投票

当然还有更多创新点敬请看paper的前面强调内容以及具体的各文件的探索历程。

## 
