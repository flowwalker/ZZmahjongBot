#!/usr/bin/env python3
"""从 checkpoint 提取模型权重，用于 Botzone 上传."""
import sys, os, torch

BASE = os.path.dirname(os.path.abspath(__file__))

def export(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f'[ERROR] Not found: {ckpt_path}'); sys.exit(1)
    print(f'Loading: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        epoch = ckpt.get('epoch', '?'); acc = ckpt.get('best_acc', 0)
        print(f'  Format: checkpoint (epoch={epoch}, best_acc={acc:.4f})')
        weights = ckpt['model']
    else:
        print(f'  Format: pure state_dict ({len(ckpt)} layers)')
        weights = ckpt
    out = os.path.join(os.path.dirname(ckpt_path) or '.', 'model_weights.pt')
    torch.save(weights, out)
    size = os.path.getsize(out) / (1024 * 1024)
    print(f'Saved: {out} ({size:.1f} MB)')
    print(f'Next: rename to model_final.pt and upload to Botzone')

if __name__ == '__main__':
    default = os.path.join(BASE, 'checkpoint', 'sl_pretrained.pt')
    path = sys.argv[1] if len(sys.argv) > 1 else default
    export(path)
