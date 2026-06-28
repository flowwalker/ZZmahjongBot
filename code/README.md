# ZZMahjongBot — 代码使用说明

> 国标麻将 AI，Botzone 积分赛排名 17（1289分），模拟赛排名 11（1410分）。
> 论文见 `paper.pdf`。

这份代码是国标麻将 AI 的完整实验记录，包含从最初 6 通道 CNN 到最终 Dual Path 架构的 24 个 bot 版本，以及特征工程、数据增强、训练管线等各维度的演进。下面是一些可以运行的命令，方便快速上手和检查各模块是否正常工作。

---

## 环境准备

```bash
# 1. Python 3.12+
python3 --version

# 2. 安装依赖
pip install torch numpy

# 3. 安装 PyMahjongGB（国标麻将算番库）
pip install PyMahjongGB

#    Linux 上直接安装即可。Mac Apple Silicon 若编译失败，尝试：
#    CC=/opt/homebrew/opt/llvm/bin/clang \
#    CXX=/opt/homebrew/opt/llvm/bin/clang++ \
#    LDFLAGS="-L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++" \
#    pip install PyMahjongGB

# 4. 确认安装
python3 -c "from MahjongGB import MahjongFanCalculator; print('PyMahjongGB OK')"

# 5. 进入 code/ 目录
pwd         # 当前位置应为 code/
ls tools/   # 应看到 validate_version.py, tournament.py 等
```

---

## 快速检查：单个版本的预处理+训练管线

`tools/validate_version.py` 会对指定版本做一轮小规模的预处理和训练，检查整个管线是否跑通：

```bash
cd tools
python3 validate_version.py v2_baseline
```

正常的话会看到预处理完成、几 batch 训练 loss 下降。整个过程约 30 秒。

文件里面内置了data.txt的小型测试版本data_test.txt，因此可以进行小型验证

对其他版本也可以用同样方式跑：

```bash
python3 validate_version.py v3_cnn_Res_Mish_SE_standard   # 首次引入 SE 注意力
python3 validate_version.py v14best_cnnSE_preload2x       # CNN-SE 金字塔, 160ch
python3 validate_version.py v16_dual_light_2x             # Dual Path 轻量版
```

`human_design_versions` 是纯规则 bot（不含神经网络，按向听数+牌效贪心决策），所以不适用于这个脚本。

---

## 特征工程演进

特征通道从 6ch 逐步演进到 160ch，每个版本在 `features/` 下：

```bash
cd tools
ls ../features/v*.py
```

查看各版本的通道数：

```bash
python3 -c "
import sys
sys.path.insert(0, '../bot_All_Versions/v2_baseline')
sys.path.insert(0, '../features')
for f in ['v1_6ch_minimal', 'v2_16ch_visible', 'v3_148ch_discard_history', 
          'v4_224ch_semantic', 'v5_155ch_aux_state', 'v6_160ch_meld21']:
    mod = __import__(f)
    print(f'{mod.FeatureAgent.OBS_SIZE:4d}ch  {f}')
"
```

输出大致为 6ch → 16ch → 148ch → 224ch → 155ch → 160ch。其中 224ch 版本因 Python 预处理耗时过长被放弃，最终收敛到 160ch。详细说明见 `features/FEATURE.md`。

---

## 模型架构演进

11 个模型版本在 `models/` 下，覆盖了从纯 CNN 到 Dual Path 的完整探索：

```bash
cd tools
ls ../models/v*.py
```

实例化几个代表性模型，跑一次前向传播确认结构正常：

```bash
python3 -c "
import sys, torch
sys.path.insert(0, '../models')
tests = [
    ('v1_cnn_3layer_6ch',       6),
    ('v2_cnn_se_4block_16ch',   16),
    ('v4_bilstm_sa_148ch',      148),
    ('v8_cnn_se_pyramid_160ch', 160),
    ('v9_dual_light_160ch',     160),
]
for name, ch in tests:
    mod = __import__(name)
    Model = None
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
            Model = obj; break
    m = Model(in_channels=ch)
    x = torch.randn(1, ch, 4, 9)
    y = m({'observation': x, 'action_mask': torch.ones(1, 235)})
    params = sum(p.numel() for p in m.parameters())
    print(f'{name:<35}  {params/1e6:5.1f}M params,  forward OK')
"
```

详细说明见 `models/MODEL.md`。

---

## Bot 对战

`tools/tournament.py` 可以从 `bot_All_Versions/` 加载任意 bot 进行麻将对局：

```bash
cd tools

# 列出可用 bot
python3 tournament.py --list

# 四个版本混战 20 局
python3 tournament.py -n 20 v2_baseline v3_cnn_Res_Mish_SE_standard v14best_cnnSE_preload2x v16_dual_light_2x
```

> 注意：当前版本的洗牌随机性不太够，同一副牌在不同对局中的变化有限，结果只能作为大致参考。这个问题在 `tools/NOTE.md` 里有说明。

---

## 论文与代码对应

| 论文段落 | 代码位置 |
|----------|---------|
| §2.1 SE 通道注意力 | `models/v2_cnn_se_4block_16ch.py` |
| §2.2 160ch 特征 | `features/v6_160ch_meld21.py` |
| §2.3 TTA + 多模型投票 | `mains/` |
| §2.4 CNN-SE 金字塔 | `models/v8_cnn_se_pyramid_160ch.py` |
| §2.4 Dual Path | `models/v9_dual_light_160ch.py` |
| §2.5 内存预加载 | `datasets/` |
| §2.5 数据增强 288-fold | `augments/` |
| §3.1 全架构拼接崩溃 | `models/v6_gated_fusion_224ch.py`, `v7_multibranch_cross_attn_224ch.py` |
| §3.2 v 头失败 | `rl_failure/` |
| §3.5 288× 灾难 | `bot_All_Versions/v22_no_time_cnnSE_preload288x/` |
| §5.1 特征演进 | `features/` + `features/FEATURE.md` |
| §5.2 模型演进 | `models/` + `models/MODEL.md` |
| Botzone 最终提交 | `bot_All_Versions/botzone_versions/bot_final_upload/` |

---

## 目录结构

```
code/
├── tools/                         入口脚本
│   ├── validate_version.py        检查单个版本管线是否通
│   ├── tournament.py              Bot 对战
│   ├── export_weights.py          权重导出
│   └── data_test.txt              测试用数据（前 5 局）
│
├── bot_All_Versions/              24 个 Bot 版本
│   ├── v1_starter/                起点：3层CNN, 6ch
│   ├── v2_baseline/               3层CNN, 16ch
│   ├── v3_cnn_Res_Mish_SE_standard/ 引入 SE + CoordConv + Mish
│   ├── v6_to_148ch_add_lstm_attn/   引入 BiLSTM+SA, 148ch
│   ├── v14best_cnnSE_preload2x/    CNN-SE 金字塔, 160ch, 2×增强
│   ├── v16_dual_light_2x/          Dual Path 轻量版, 160ch
│   ├── v19best_cnnSE_preload2x_3vote/ 3模型投票版
│   ├── v20best_cnnSE_preload2x_3vote50x/ 3模型×50 TTA
│   ├── botzone_versions/           实际提交 Botzone 的版本
│   └── ...                        其余版本
│
├── features/                       特征工程演进（6版）
├── models/                         模型架构演进（11版）
├── preprocess_s/                   数据预处理演进（4版）
├── datasets/                       数据集加载演进（5版）
├── augments/                       数据增强（288-fold群论结构）
├── sl_pretrains/                   SL训练脚本演进（3版）
├── mains/                          Botzone部署入口演进（3版）
├── rl_failure/                     RL尝试记录
└── README.md                       本文件
```

---

## 一些说明

- 目录名中的 `fail` 表示训练发散或架构不可训，`no_time` 表示设计完成但因算力/时间未充分训练。这些失败的尝试在论文 §3 中有详细分析。
- 代码中的探索远多于论文能容纳的内容，各子目录下的 `*.md` 文件补充了更多细节，欢迎进一步了解和完善。
