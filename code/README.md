# ZZMahjongBot — 助教验证指南

> 国标麻将 AI，Botzone 积分赛排名 17（1289分），模拟赛排名 11（1410分）。
> 论文见 `paper.pdf`。

本指南按顺序引导验证：**环境 → 单版本 → 批量 → 演进线 → 对战 → 对照论文**。

---

## 环境准备

```bash
# 1. Python 3.12+
python3 --version

# 2. 安装依赖
pip install torch numpy

# 3. 安装 PyMahjongGB（国标麻将算番库，所有 feature 版本依赖）
pip install PyMahjongGB

#    Linux 上直接安装即可。Mac Apple Silicon 若编译失败，使用：
#    CC=/opt/homebrew/opt/llvm/bin/clang \
#    CXX=/opt/homebrew/opt/llvm/bin/clang++ \
#    LDFLAGS="-L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++" \
#    pip install PyMahjongGB

# 4. 验证安装
python3 -c "from MahjongGB import MahjongFanCalculator; print('PyMahjongGB OK')"

# 5. 确认当前在 code/ 下
pwd         # 应该以 archive/code 结尾（或 code/ 结尾）
ls tools/   # 应该看到 validate_version.py, tournament.py 等
```

---

## 步骤 1：验证单个版本（管线完整性）

验证一个 Bot 版本能否完成 **预处理 → 数据加载 → 小批次 SL 训练** 全流程。

```bash
cd tools
python3 validate_version.py v2_baseline
```

**期望输出**：
```
📁 版本: v2_baseline
📋 Step 1: 拷贝文件...
📋 Step 2: 提取前 5 局数据...
📋 Step 3: 运行预处理 (preprocess.py)...
   ✅ 预处理完成
📋 Step 4: 小批次 SL 训练 (10 batches × 64)...
   模型类: CNNModel     OBS_SIZE: 16     ACT_SIZE: 235
   batch 1/4: loss=2.0...   batch 4/4: loss=1.5...   ✅ 训练完成
✅ 版本 'v2_baseline' 验证通过！
```

这证明了：feature 提取 → 数据预处理 → 模型前向/反向传播 → 参数更新 全链路正常。

---

## 步骤 2：批量验证关键版本

以下是论文中重点讨论的 4 个代表性版本，覆盖了架构演进的关键节点：

```bash
cd tools

# 依次验证（每个约 30 秒）
python3 validate_version.py v2_baseline          # CNN 基础版，16ch
python3 validate_version.py v3_cnn_Res_Mish_SE_standard  # 首次引入 SE 注意力
python3 validate_version.py v14best_cnnSE_preload2x      # CNN-SE 金字塔，160ch（最终部署基础）
python3 validate_version.py v16_dual_light_2x            # Dual Path 双路并行（轻量版）
```

或一键批量：
```bash
python3 -c "
import subprocess, sys
for v in ['v2_baseline', 'v3_cnn_Res_Mish_SE_standard', 'v14best_cnnSE_preload2x', 'v16_dual_light_2x']:
    r = subprocess.run([sys.executable, 'validate_version.py', v], capture_output=True, text=True, timeout=120)
    ok = 'PASS' if r.returncode == 0 else 'FAIL'
    print(f'{ok} {v}')
"
```

**验证 24 个全部版本** (约 10 分钟)：
```bash
for d in ../bot_All_Versions/*/; do
    v=$(basename "$d")
    if [ -f "$d/feature.py" ] && [ -f "$d/model.py" ]; then
        python3 validate_version.py "$v" 2>&1 | tail -1
    fi
done
```

> **预期结果**：24 个版本全部通过（除 `human_design_versions`——它是人工规则 bot，无需 feature/model，自然不适用此脚本）。

---

## 步骤 3：特征工程演进验证（论文 §5.1）

特征通道从 6ch → 160ch 的演进，每个版本定义在 `features/` 目录下：

```bash
cd tools
# 列出特征版本
ls ../features/v*.py
```

```bash
# 验证每个特征版本的 OBS_SIZE（需 bot 目录提供 agent.py 基类）
python3 -c "
import sys
sys.path.insert(0, '../bot_All_Versions/v2_baseline')   # 提供 agent.py
sys.path.insert(0, '../features')
for f in ['v1_6ch_minimal', 'v2_16ch_visible', 'v3_148ch_discard_history', 
          'v4_224ch_semantic', 'v5_155ch_aux_state', 'v6_160ch_meld21']:
    mod = __import__(f)
    print(f'{mod.FeatureAgent.OBS_SIZE:4d}ch  {f}')
"
```

**期望输出**：
```
   6ch  v1_6ch_minimal
  16ch  v2_16ch_visible
 148ch  v3_148ch_discard_history
 224ch  v4_224ch_semantic
 155ch  v5_155ch_aux_state
 160ch  v6_160ch_meld21
```

论文中的"6ch→16ch→148ch→224ch(放弃)→155ch→160ch"路线由此可证。

详细说明见 `features/FEATURE.md`。

---

## 步骤 4：模型架构演进验证（论文 §5.2）

11 个模型架构从纯 CNN 到 Dual Path，定义在 `models/` 目录下：

```bash
cd tools
# 列出模型版本
ls ../models/v*.py
```

```bash
# 验证每个模型可实例化、可前向传播
python3 -c "
import sys, torch
sys.path.insert(0, '../models')

# 逐一实例化（代表性的 5 个）
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
    print(f'✅ {name:<35}  {params/1e6:5.1f}M params,  output OK')
"
```

**期望输出**：5 个模型均显示 `✅ ... output OK`，参数量与实际架构匹配。其中 v9_dual_light 约 3.6M 参数，与论文中"Dual Light 3.56M"一致。

详细说明见 `models/MODEL.md`。

---

## 步骤 5：Bot 对战测试

`tools/tournament.py` 从 `bot_All_Versions/` 动态加载任意 Bot 进行麻将对局。

### 5.1 列出可用 Bot

```bash
cd tools
python3 tournament.py --list
```

### 5.2 四 Bot 混战

```bash
# 4 个不同版本各坐一家，打 20 局（约 2 分钟）
python3 tournament.py -n 20 v2_baseline v3_cnn_Res_Mish_SE_standard v14best_cnnSE_preload2x v16_dual_light_2x
```

**期望输出**：每 20 局打印一次实时排名（总分、场均、胜率、🥇🥈🥉分布），最后给出汇总表。

### 5.3 同模型四家混战（检一致性）

```bash
# v14 对自己 ×4，验证模型在相同局面下的决策一致性
python3 tournament.py -n 10 v14best v14best v14best v14best
```

> **已知限制**：当前版本洗牌随机性不足，同一副牌的牌面分布在重复对局中变化有限。结果仅供参考，已在 `tools/NOTE.md` 中说明。

---

## 步骤 6：对照论文

阅读 `paper.pdf` 后，可对照以下映射表在代码中找到对应实现：

| 论文段落 | 代码位置 | 关键文件 |
|----------|---------|---------|
| §2.1 SE 通道注意力 | `models/v2_cnn_se_4block_16ch.py` | SE 模块首次引入 |
| §2.2 160ch 特征 | `features/v6_160ch_meld21.py` | 最终特征版本 |
| §2.3 TTA + 投票 | `mains/` | Botzone 部署入口 |
| §2.4 CNN-SE 金字塔 (v8) | `models/v8_cnn_se_pyramid_160ch.py` | 最终部署模型 |
| §2.4 Dual Path | `models/v9_dual_light_160ch.py` | 双路并行架构 |
| §2.5 内存预加载 | `datasets/` | MemPreloadDataset |
| §2.5 数据增强 288-fold | `augments/` | 群论增强实现 |
| §3.1 全架构拼接崩溃 | `models/v6_gated_fusion_224ch.py`, `v7_multibranch_cross_attn_224ch.py` | 标 "fail" 的版本 |
| §3.2 v 头失败 | `rl_failure/` | RL PPO 尝试记录 |
| §3.5 288× 灾难 | `bot_All_Versions/v22_no_time_cnnSE_preload288x/` | 标 "no_time" |
| §5.1 特征演进 | `features/` + `features/FEATURE.md` | 6 版演进说明 |
| §5.2 模型演进 | `models/` + `models/MODEL.md` | 11 版演进说明 |
| Botzone 最终提交 | `bot_All_Versions/botzone_versions/bot_final_upload/` | 实际提交的 `__main__.py` |

---

## 目录结构速查

```
code/
├── tools/                          ★ 入口：验证和对战脚本
│   ├── validate_version.py         验证单个版本的完整管线
│   ├── tournament.py               Bot 对战锦标赛
│   ├── export_weights.py           权重导出
│   └── data_test.txt               测试用数据（前 5 局）
│
├── bot_All_Versions/               ★ 24 个 Bot 版本（每个含 feature/model/agent/preprocess 等）
│   ├── v1_starter/                 起点：3层CNN, 6ch
│   ├── v2_baseline/                baseline: 3层CNN, 16ch
│   ├── v3_cnn_Res_Mish_SE_standard/ 首次引入 SE + CoordConv + Mish
│   ├── v6_to_148ch_add_lstm_attn/   引入 BiLSTM+SelfAttention, 148ch（质的飞跃）
│   ├── v14best_cnnSE_preload2x/    CNN-SE 金字塔, 160ch, 2×增强（最终部署基础）
│   ├── v16_dual_light_2x/          Dual Path 轻量版, 160ch
│   ├── v19best_cnnSE_preload2x_3vote/ 3模型投票版
│   ├── v20best_cnnSE_preload2x_3vote50x/ 3模型×50 TTA（Botzone最终提交版）
│   ├── botzone_versions/           实际提交 Botzone 的历史版本
│   └── ...                         其余版本（含失败的探索）
│
├── features/                       特征工程演进（6版, 6ch→160ch）
├── models/                         模型架构演进（11版, CNN→Dual）
├── preprocess_s/                   数据预处理演进（4版）
├── datasets/                       数据集加载演进（5版, 含MemPreload）
├── augments/                       数据增强（288-fold群论结构）
├── sl_pretrains/                   SL训练脚本演进（3版）
├── mains/                          Botzone部署入口演进（3版）
├── rl_failure/                     RL强化学习失败记录（诚实存档）
└── README.md                       本文件
```

---

## 可能的问题与回答

**Q: `human_design_versions/` 为什么不通过验证？**
A: 它是纯规则 bot（基于向听数+牌效的贪心策略），不含神经网络，无需 feature/model，只需上传 `human_design.py` 到 Botzone 即可运行。`validate_version.py` 的设计目标是验证神经网络 Bot，不适配规则 bot。

**Q: 部分版本目录名带 "fail" 或 "no_time" 是什么意思？**
A: 诚实标注——`fail`=训练发散或架构不可训，`no_time`=设计完成但因算力/时间未充分训练。论文 §3 对这些失败做了详细分析。
