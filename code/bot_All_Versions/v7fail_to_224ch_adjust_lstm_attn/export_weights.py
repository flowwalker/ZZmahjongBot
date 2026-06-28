#!/usr/bin/env python3
"""从 checkpoint 提取模型权重，用于 Botzone 上传."""
import sys, os, io, types, pickle, torch

BASE = os.path.dirname(os.path.abspath(__file__))


class _NPUUnpickler(pickle.Unpickler):
    """拦截所有 torch_npu.* 引用，处理 NPU persistent id"""
    def find_class(self, module, name):
        if module.startswith('torch_npu'):
            return type('NPUDummy', (), {'__init__': lambda s, *a, **k: None})
        return super().find_class(module, name)

    def persistent_load(self, pid):
        """NPU checkpoint 使用 persistent_id 存储 tensor storage。
        pid 格式通常是 ('storage', storage_type, key, location, size)"""
        if isinstance(pid, tuple) and pid[0] == 'storage':
            # 返回一个空的 CPU storage，torch 会在后续 fill 它
            return torch.storage.TypedStorage(
                wrap_storage=torch.UntypedStorage(pid[4] if len(pid) > 4 else 0),
                dtype=pid[1] if len(pid) > 1 else torch.float32)
        raise pickle.UnpicklingError(f'Unsupported persistent id: {pid}')


def export(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f'[ERROR] Not found: {ckpt_path}'); sys.exit(1)

    # 用自定义 Unpickler 加载，绕过 torch_npu 依赖
    with open(ckpt_path, 'rb') as f:
        ckpt = _NPUUnpickler(f).load()

    weights = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    out = os.path.join(os.path.dirname(ckpt_path) or '.', 'model_weights.pt')
    torch.save(weights, out)
    print(f'Saved: {out} ({os.path.getsize(out)/(1024*1024):.1f} MB)')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'checkpoint', 'sl_pretrained.pt')
    export(path)
