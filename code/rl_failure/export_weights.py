#!/usr/bin/env python3
"""
从 SL 训练 checkpoint 中提取纯模型权重，用于 Botzone 上传。

用法:
    python export_weights.py                          # 从默认路径提取
    python export_weights.py checkpoint/sl_pretrained.pt  # 指定 checkpoint 路径

输出:
    checkpoint/model_weights.pt  — 纯权重文件
    改名为 model.pt 后上传到 Botzone 存储空间即可。
"""
import sys, os
import torch

BASE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'

def export(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f'[ERROR] checkpoint not found: {ckpt_path}')
        print(f'  Run SL training first (python sl_pretrain.py)')
        sys.exit(1)

    print(f'Loading: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # 判断格式: 完整 checkpoint 还是纯权重
    if isinstance(ckpt, dict) and 'model' in ckpt:
        epoch = ckpt.get('epoch', '?')
        best_acc = ckpt.get('best_acc', 0)
        print(f'  Format: full checkpoint (epoch={epoch}, best_acc={best_acc:.4f})')
        weights = ckpt['model']
    else:
        print(f'  Format: pure state_dict ({len(ckpt)} layers)')
        weights = ckpt

    out_path = os.path.join(os.path.dirname(ckpt_path) or '.', 'model_weights.pt')
    torch.save(weights, out_path)
    size_kb = os.path.getsize(out_path) / 1024
    print(f'Saved: {out_path} ({size_kb:.1f} KB, {len(weights)} layers)')
    print()
    print('下一步:')
    print(f'  1. 将 {out_path} 改名为 model.pt')
    print(f'  2. 上传到 Botzone 存储空间')
    print(f'  3. 上传 bot.zip 作为 Bot 程序')

if __name__ == '__main__':
    default = os.path.join(BASE, 'checkpoint', 'sl_pretrained.pt')
    path = sys.argv[1] if len(sys.argv) > 1 else default
    export(path)
