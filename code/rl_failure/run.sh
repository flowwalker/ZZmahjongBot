#!/bin/bash
# baseline 一键训练 (16通道扩展特征 + MPS GPU + 断点续传)
set -e
cd "$(dirname "$0")"
source ../venv/bin/activate

echo "===== 开始: $(date) ====="

# 预处理 (16通道, 需重新跑)
echo "[1/2] 数据预处理..."
if [ ! -f data/count.json ]; then
    ln -sf ../../starter_code/SL/data/data.txt data/data.txt
    python preprocess.py
else
    echo "  -> 已有预处理数据, 跳过"
fi

# SL 预训练 (支持 MPS + 断点续传)
echo "[2/2] SL 预训练..."
python sl_pretrain.py

echo "===== 完成: $(date) ====="
echo "模型: checkpoint/sl_pretrained.pt"
echo ""
echo "上传到 Botzone: 将 checkpoint/sl_pretrained.pt 上传到存储空间, 命名为 model.pt"
echo "Bot 文件: bot.zip (先用随机权重测试接口)"
