#!/usr/bin/env python3
"""从 checkpoint 提取模型权重，用于 Botzone 上传."""

import sys, os
import torch

BASE = os.path.dirname(os.path.abspath(__file__))


def export(ckpt_path: str) -> str:
    if not os.path.exists(ckpt_path):
        print(f'[ERROR] checkpoint not found: {ckpt_path}')
        sys.exit(1)

    print(f'Loading: {ckpt_path}')

    # 关键: map_location='cpu' 将 NPU tensor 安全转到 CPU
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict) and 'model' in ckpt:
        epoch = ckpt.get('epoch', '?')
        best_acc = ckpt.get('best_acc', 0.0)
        print(f'  Full checkpoint: epoch={epoch}, best_acc={best_acc:.4f}')
        weights = ckpt['model']
    else:
        print(f'  Pure state_dict: {len(ckpt)} tensors')
        weights = ckpt

    # 逐层确保 CPU + contiguous
    clean = {}
    for k, v in weights.items():
        t = v.cpu().contiguous().float()
        if t.device.type != 'cpu':
            print(f'[WARN] {k} still on {t.device}, forcing CPU...')
            t = t.to('cpu')
        clean[k] = t

    out_path = os.path.join(os.path.dirname(ckpt_path) or '.', 'model_final_pure_cnnSE.pt')
    torch.save(clean, out_path)

    size_kb = os.path.getsize(out_path) / 1024
    n_params = sum(v.numel() for v in clean.values())
    print(f'Exported: {out_path}')
    print(f'  {size_kb:.1f} KB | {n_params:,} params | {len(clean)} tensors | all CPU')
    print()
    print('上传 Botzone:')
    print(f'  将 {out_path} 与 bot.zip 一起上传到 Botzone')

    return out_path


if __name__ == '__main__':
    default = os.path.join(BASE, 'checkpoint', 'sl_pretrained.pt')
    path = sys.argv[1] if len(sys.argv) > 1 else default
    export(path)
