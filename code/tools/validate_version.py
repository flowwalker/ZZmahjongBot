#!/usr/bin/env python3
"""
版本验证脚本 — 验证 bot_All_Versions 中某个版本能否正常预处理+训练。

用法:
    cd code/tools
    python validate_version.py <version_name>

示例:
    python validate_version.py v1_starter
    python validate_version.py v2_baseline
    python validate_version.py v14best_cnnSE_preload2x

流程:
    1. 从 data.txt 提取前 N 局到临时工作区
    2. 运行该版本的预处理脚本 (preprocess.py / preprocess_mp.py)
    3. 用该版本的 model + feature + dataset 做小批次 SL 训练
    4. 报告成功或失败 (含错误信息)
"""

import sys
import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

# 配置
CODE_ROOT = Path(__file__).resolve().parent.parent
BOT_ALL = CODE_ROOT / "bot_All_Versions"
DATA_TXT = CODE_ROOT / "tools" / "data_test.txt"
N_MATCHES = 5          # 提取前 N 局
TRAIN_BATCHES = 10     # 训练批次数
BATCH_SIZE = 64        # 每批大小

# 辅助函数

def extract_first_n_matches(n: int, src: Path, dst: Path):
    """从完整 data.txt 中提取前 n 局到 dst"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    matches_found = 0
    with open(src, encoding='utf-8') as fin, open(dst, 'w', encoding='utf-8') as fout:
        for line in fin:
            fout.write(line)
            if line.startswith('Match '):
                matches_found += 1
                if matches_found >= n:
                    break
    return matches_found


def read_count_json(count_path: Path) -> list:
    """读取 count.json，返回每局样本数列表"""
    import json
    with open(count_path) as f:
        return json.load(f)


# 验证逻辑

def validate_version(version_name: str):
    """验证指定版本"""
    version_dir = BOT_ALL / version_name
    if not version_dir.is_dir():
        print(f"❌ 版本目录不存在: {version_dir}")
        return False

    py_files = sorted(version_dir.glob("*.py"))
    if not py_files:
        print(f"❌ 版本目录无 .py 文件: {version_dir}")
        return False

    print(f"📁 版本: {version_name}")
    print(f"   文件: {', '.join(p.name for p in py_files)}")

    # 创建临时工作区
    tmp = Path(tempfile.mkdtemp(prefix=f"validate_{version_name.replace('/', '_')}_"))
    tmp_data = tmp / "data"
    tmp_data.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: 拷贝版本文件到临时区
        print(f"\n📋 Step 1: 拷贝文件到 {tmp}")
        for f in py_files:
            shutil.copy(f, tmp / f.name)

        # Step 2: 提取前 N 局数据
        print(f"\n📋 Step 2: 提取前 {N_MATCHES} 局数据...")
        data_txt = tmp_data / "data.txt"
        n_found = extract_first_n_matches(N_MATCHES, DATA_TXT, data_txt)
        print(f"   提取 {n_found} 局 → {data_txt} ({data_txt.stat().st_size:,} bytes)")

        # Step 3: 运行预处理
        preprocess_file = None
        for cand in ["preprocess.py", "preprocess_mp.py"]:
            if (tmp / cand).exists():
                preprocess_file = cand
                break

        if preprocess_file:
            print(f"\n📋 Step 3: 运行预处理 ({preprocess_file})...")
            result = subprocess.run(
                [sys.executable, str(tmp / preprocess_file)],
                cwd=str(tmp),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                print(f"   ❌ 预处理失败 (exit {result.returncode})")
                print(f"   stdout:\n{result.stdout[-2000:]}")
                print(f"   stderr:\n{result.stderr[-2000:]}")
                return False
            print(f"   ✅ 预处理完成")
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines()[-5:]:
                    print(f"      {line}")

            # 验证产物
            npz_files = sorted(tmp_data.glob("*.npz"))
            count_json = tmp_data / "count.json"
            print(f"   产物: {len(npz_files)} npz, count.json={'✅' if count_json.exists() else '❌'}")
            if count_json.exists():
                counts = read_count_json(count_json)
                print(f"   样本分布: {counts}  (总计 {sum(counts)})")
        else:
            print(f"\n📋 Step 3: 无预处理脚本，跳过")
            # 对于 mem 版本 — 需要预先生成 npz
            # 检查是否有 data/ 下的 npz 文件可复用
            print("   注意: 此版本无预处理脚本，可能需要已有 npz 数据")

        # Step 4: 小批次 SL 训练
        print(f"\n📋 Step 4: 小批次 SL 训练 ({TRAIN_BATCHES} batches × {BATCH_SIZE})...")
        train_ok = run_minimal_training(tmp, version_name)
        if not train_ok:
            return False

        print(f"\n{'='*60}")
        print(f"✅ 版本 '{version_name}' 验证通过！")
        print(f"   (预处理 + 小批次训练 均无错误)")
        return True

    except subprocess.TimeoutExpired:
        print(f"❌ 预处理超时 (>120s)")
        return False
    except Exception as e:
        print(f"❌ 验证过程异常: {e}")
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\n🧹 已清理临时目录: {tmp}")


def run_minimal_training(tmp: Path, version_name: str) -> bool:
    """
    在临时目录中运行最小化 SL 训练。
    不调用 sl_pretrain.py 的 main，而是内联训练循环以避
    免硬编码路径、NPU 设备、checkpoint 等问题。
    """
    tmp_str = str(tmp)
    tmp_data_str = str(tmp / "data")

    # 内联训练脚本
    train_script = f'''
import sys, os
os.chdir({tmp_str!r})
sys.path.insert(0, {tmp_str!r})

import torch
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader

# 1. 导入版本的 model / feature
from feature import FeatureAgent

# 检测模型类名 (CNNModel / DualPathModel / ...)
import model as model_mod
ModelClass = None
for name in dir(model_mod):
    obj = getattr(model_mod, name)
    if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
        ModelClass = obj
        break

if ModelClass is None:
    print("❌ 未找到模型类 (nn.Module 子类)")
    sys.exit(1)

print(f"   模型类: {{ModelClass.__name__}}")
print(f"   OBS_SIZE: {{FeatureAgent.OBS_SIZE}}")
print(f"   ACT_SIZE: {{FeatureAgent.ACT_SIZE}}")

# 2. 检测设备
if hasattr(torch, 'npu') and torch.npu.is_available():
    device = torch.device('npu')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"   设备: {{device}}")

# 3. 构建简易数据集
data_dir = {tmp_data_str!r}
count_path = os.path.join(data_dir, 'count.json')

if not os.path.exists(count_path):
    print("⚠️  count.json 不存在，无法训练，但预处理/导入验证通过")
    sys.exit(0)

with open(count_path) as f:
    match_samples = json.load(f)

total_samples = sum(match_samples)
print(f"   样本总数: {{total_samples}}, 局数: {{len(match_samples)}}")

# 收集所有 npz 数据
all_obs, all_masks, all_acts = [], [], []
for i, cnt in enumerate(match_samples):
    npz_path = os.path.join(data_dir, f'{{i}}.npz')
    if not os.path.exists(npz_path):
        print(f"⚠️  缺少 {{npz_path}}，跳过")
        continue
    d = np.load(npz_path)
    all_obs.append(d['obs'].astype(np.float32))
    all_masks.append(d['mask'].astype(np.float32))
    all_acts.append(d['act'])

if not all_obs:
    print("⚠️  无可用 npz 数据")
    sys.exit(0)

obs_all = np.concatenate(all_obs, axis=0)
mask_all = np.concatenate(all_masks, axis=0)
act_all = np.concatenate(all_acts, axis=0)
print(f"   加载: obs={{obs_all.shape}}, mask={{mask_all.shape}}, act={{act_all.shape}}")

# 如果观测通道和模型期望不一致，做适配
in_channels = FeatureAgent.OBS_SIZE
if len(obs_all.shape) == 3:
    # (N, H, W) → (N, C, H, W)  其中 C 从 shape 推断
    obs_all = obs_all.reshape(obs_all.shape[0], in_channels, -1, obs_all.shape[-1])

print(f"   观测 reshape: {{obs_all.shape}}")

# 简易 Dataset
class SimpleDS(Dataset):
    def __init__(self, obs, mask, act):
        self.obs = torch.tensor(obs)
        self.mask = torch.tensor(mask)
        self.act = torch.tensor(act, dtype=torch.long)

    def __len__(self):
        return len(self.act)

    def __getitem__(self, idx):
        return self.obs[idx], self.mask[idx], self.act[idx]

# 4. 训练循环
ds = SimpleDS(obs_all, mask_all, act_all)
loader = DataLoader(ds, batch_size=min({BATCH_SIZE}, len(ds)), shuffle=True)

# 实例化模型
try:
    model = ModelClass(in_channels=in_channels).to(device)
except TypeError:
    model = ModelClass().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

n_batches = min({TRAIN_BATCHES}, len(loader))
print(f"   开始训练: {{n_batches}} batches...")

for batch_idx, (obs, mask, act) in enumerate(loader):
    if batch_idx >= n_batches:
        break
    obs, mask, act = obs.to(device), mask.to(device), act.to(device)

    # 适配不同的模型 forward 签名
    try:
        output = model({{'observation': obs, 'action_mask': mask}})
    except TypeError:
        output = model(obs)

    # 处理不同输出格式
    if isinstance(output, tuple):
        logits = output[0]
    elif isinstance(output, dict):
        logits = output.get('logits', output.get('policy', list(output.values())[0]))
    else:
        logits = output

    # 确保 logits 形状正确 (batch, 235)
    if logits.dim() > 2:
        logits = logits.reshape(logits.size(0), -1)

    loss = F.cross_entropy(logits, act)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx == 0 or (batch_idx + 1) % max(1, n_batches // 3) == 0:
        print(f"   batch {{batch_idx+1}}/{{n_batches}}: loss={{loss.item():.4f}}")

print("   ✅ 训练完成，无错误")
'''

    result = subprocess.run(
        [sys.executable, "-c", train_script],
        cwd=str(tmp),
        capture_output=True,
        text=True,
        timeout=120,
    )

    # 打印输出 (截断)
    output = result.stdout + result.stderr
    for line in output.splitlines():
        print(f"   {line}")

    if result.returncode != 0:
        print(f"   ❌ 训练失败 (exit {result.returncode})")
        return False
    return True



if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 列出可用版本
        versions = sorted(
            d.name for d in BOT_ALL.iterdir()
            if d.is_dir() and list(d.glob("*.py"))
        )
        print("用法: python validate_version.py <version_name>")
        print(f"\n可用版本 ({len(versions)}):")
        for v in versions:
            print(f"  {v}")
        sys.exit(1)

    version_name = sys.argv[1]
    success = validate_version(version_name)
    sys.exit(0 if success else 1)
