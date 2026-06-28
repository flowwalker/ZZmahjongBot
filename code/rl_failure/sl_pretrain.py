"""
SL 预训练 — 支持 MPS GPU、断点续传、Ctrl+C 安全中断

用法: python sl_pretrain.py
中断后再次运行自动从 checkpoint 恢复。
"""

import torch, torch.nn.functional as F, numpy as np, os, json, signal, sys
from torch.utils.data import Dataset, DataLoader
from bisect import bisect_right
from model import CNNModel
from feature import FeatureAgent


class SLDataset(Dataset):
    def __init__(self, data_dir='data', begin=0.0, end=1.0):
        with open(os.path.join(data_dir, 'count.json')) as f:
            self.match_samples = json.load(f)
        total = len(self.match_samples)
        self.begin = int(begin * total)
        self.end = int(end * total)
        self.match_samples = self.match_samples[self.begin:self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        offsets = [0]
        for m in self.match_samples:
            offsets.append(offsets[-1] + m)
        self.offsets = offsets
        print(f'Dataset: {self.matches} matches, {self.samples} samples')

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        mid = bisect_right(self.offsets, idx) - 1
        sid = idx - self.offsets[mid]
        d = np.load(f'data/{mid + self.begin}.npz')
        return (torch.tensor(d['obs'][sid]).float(),
                torch.tensor(d['mask'][sid]).float(),
                torch.tensor(d['act'][sid]).long())


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

    # 数据集
    split = 0.9
    train_ds = SLDataset('data', 0.0, split)
    val_ds = SLDataset('data', split, 1.0)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    # 模型 (通道数从 FeatureAgent 获取)
    model = CNNModel(in_channels=FeatureAgent.OBS_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # 断点续传
    ckpt_path = 'checkpoint/sl_pretrained.pt'
    start_epoch = 0
    best_acc = 0.0
    if os.path.exists(ckpt_path):
        start_epoch, best_acc = load_ckpt(ckpt_path, model, optimizer)
        start_epoch += 1  # 已完成 epoch 的下一个
        print(f'Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}')

    patience = 5
    no_improve = 0 if start_epoch == 0 else sum(1 for _ in range(start_epoch))
    max_epochs = 50
    interrupted = False

    def on_signal(sig, frame):
        global interrupted
        interrupted = True
        print('\nInterrupted! Saving checkpoint...')

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    for epoch in range(start_epoch, max_epochs):
        if interrupted:
            break

        # Train
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

        # Validate
        model.eval()
        correct = 0
        with torch.no_grad():
            for obs, mask, act in val_loader:
                obs, mask, act = obs.to(device), mask.to(device), act.to(device)
                logits, _ = model({'observation': obs, 'action_mask': mask})
                correct += (logits.argmax(dim=1) == act).sum().item()
        acc = correct / len(val_ds)

        print(f'Epoch {epoch+1}/{max_epochs}: loss={total_loss/len(train_loader):.4f}, val_acc={acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            save_ckpt(ckpt_path, model, optimizer, epoch, best_acc)
            # 同时导出纯权重 (Botzone 上传直接用)
            torch.save(model.state_dict(), 'checkpoint/model_weights.pt')
            print(f'  -> best model saved (pure weights also exported)')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # 无论如何保存当前状态
    if interrupted:
        save_ckpt(ckpt_path, model, optimizer,
                   epoch if 'epoch' in dir() else start_epoch, best_acc)
    print(f'Done. best_val_acc={best_acc:.4f}, ckpt={ckpt_path}')
