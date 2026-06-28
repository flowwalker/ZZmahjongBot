#!/usr/bin/env python3
import sys, os, torch
BASE = os.path.dirname(os.path.abspath(__file__))

def export(ckpt_path):
    if not os.path.exists(ckpt_path): print(f'[ERROR] Not found: {ckpt_path}'); sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    weights = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    out = os.path.join(os.path.dirname(ckpt_path) or '.', 'model_monster.pt')
    torch.save(weights, out)
    print(f'Saved: {out} ({os.path.getsize(out)/(1024*1024):.1f} MB)')

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'checkpoint', 'sl_pretrained.pt')
    export(path)
