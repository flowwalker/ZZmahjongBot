"""SL 预训练 — 内存预加载版."""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import signal
import sys
import platform
from torch.utils.data import DataLoader

from model import CNNModel
from dataset_mem import MemPreloadDataset, LazyAugSampler

#  增强配置
#  训练: 每样本从 288 随机取 1 个变换 (per_sample_random)
#  验证: 仅恒等变换，跨 epoch 可比
N_AUG_VAL = 1


def get_device():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device('npu')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def save_ckpt(path, model, optimizer, epoch, best_acc):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
    }, path)


def load_ckpt(path, model, optimizer):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['best_acc']


if __name__ == '__main__':
    device = get_device()
    print(f'Device: {device}')

    split = 0.9
    print(f'Training: per-sample random 1/288 transform')
    print(f'Validation: identity transform only')

    train_ds = MemPreloadDataset('data', 0.0, split, n_aug=1,
                                 per_sample_random=True)
    val_ds = MemPreloadDataset('data', split, 1.0, n_aug=N_AUG_VAL)

    # 轻量 sampler: 只 shuffle 原始索引 (5.87M)
    train_sampler = LazyAugSampler(train_ds.n_raw, n_aug=1)
    val_sampler = LazyAugSampler(val_ds.n_raw, n_aug=N_AUG_VAL)

    # Windows spawn 模式下每个 worker 会复制 30GB → 必须 num_workers=0
    # Linux/Mac fork 模式下 CoW 共享内存 → 可以用多 workers
    if platform.system() == 'Windows':
        nw = 0
        print('[Windows] num_workers=0 (spawn would duplicate 30GB per worker)')
    else:
        nw = 10

    train_loader = DataLoader(train_ds, batch_size=4096, sampler=train_sampler,
                              num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, sampler=val_sampler,
                            num_workers=nw, pin_memory=True)

    model = CNNModel(in_channels=160).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    ckpt_path = 'checkpoint/sl_pretrained.pt'
    start_epoch = 0
    best_acc = 0.0
    if os.path.exists(ckpt_path):
        start_epoch, best_acc = load_ckpt(ckpt_path, model, optimizer)
        start_epoch += 1
        print(f'Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}')

    max_epochs = 50
    interrupted = False

    def on_signal(sig, frame):
        global interrupted
        interrupted = True
        print('\nInterrupted! Saving checkpoint...')

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, 'SIGTERM'): signal.signal(signal.SIGTERM, on_signal)

    for epoch in range(start_epoch, max_epochs):
        if interrupted:
            break

        model.train()
        total_loss = 0.0
        for obs, mask, act in train_loader:
            if interrupted:
                break
            obs, mask, act = obs.to(device), mask.to(device), act.to(device)
            logits, _ = model({'observation': obs, 'action_mask': mask})
            loss = F.cross_entropy(logits, act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if interrupted:
            break

        model.eval()
        correct = 0
        with torch.no_grad():
            for obs, mask, act in val_loader:
                obs, mask, act = obs.to(device), mask.to(device), act.to(device)
                logits, _ = model({'observation': obs, 'action_mask': mask})
                correct += (logits.argmax(dim=1) == act).sum().item()
        acc = correct / len(val_ds)

        print(f'Epoch {epoch+1}/{max_epochs}: loss={total_loss/len(train_loader):.4f}, '
              f'val_acc={acc:.4f}')

        epoch_ckpt_path = f'checkpoint/sl_pretrained_epoch{epoch+1}.pt'
        save_ckpt(epoch_ckpt_path, model, optimizer, epoch, acc)
        print(f'  -> epoch checkpoint saved: {epoch_ckpt_path}')

        if acc > best_acc:
            best_acc = acc
            save_ckpt(ckpt_path, model, optimizer, epoch, best_acc)
            torch.save(model.state_dict(), 'checkpoint/model_weights.pt')
            print(f'  -> best model updated (pure weights also exported)')

    if interrupted:
        save_ckpt(ckpt_path, model, optimizer,
                  epoch if 'epoch' in dir() else start_epoch, best_acc)
    print(f'Done. best_val_acc={best_acc:.4f}, ckpt={ckpt_path}')
