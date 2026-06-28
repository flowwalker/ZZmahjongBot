"""SL 预训练脚本."""
import torch, torch.nn.functional as F, numpy as np, os, json, signal
from torch.utils.data import Dataset, DataLoader
from bisect import bisect_right
from model import CNNModel
from feature import FeatureAgent


class SLDataset(Dataset):
    def __init__(self, data_dir='data', begin=0.0, end=1.0):
        with open(os.path.join(data_dir, 'count.json')) as f:
            ms = json.load(f)
        total = len(ms)
        b, e = int(begin * total), int(end * total)
        self.ms, self.data_dir = ms[b:e], data_dir
        self.begin = b  # 全局 matchid 偏移，用于正确加载 npz 文件名
        offsets = [0]
        for m in self.ms: offsets.append(offsets[-1] + m)
        self.offsets = offsets
        self.samples = sum(self.ms)
        print(f'Dataset: {len(self.ms)} matches, {self.samples} samples')

    def __len__(self): return self.samples

    def __getitem__(self, idx):
        mid = bisect_right(self.offsets, idx) - 1
        sid = idx - self.offsets[mid]
        d = np.load(f'{self.data_dir}/{self.begin + mid}.npz')

        obs = d['obs'][sid].astype(np.float32)
        mask = d['mask'][sid].astype(np.float32)
        act = d['act'][sid]

        # V 目标：优先用真实终局得分，旧 npz 回退 mask proxy
        # 时序衰减：早期状态价值不确定，越靠近终局标签越准确
        n_samples = len(d['act'])
        position = sid / max(n_samples - 1, 1)  # 0.0(开局) → 1.0(终局)
        confidence = 0.5 + 0.5 * position        # 0.5 → 1.0

        if 'score' in d and np.ptp(d['score']) > 0:  # ptp > 0 → 分数不全相同
            raw_score = d['score'][sid].astype(np.float32)
            v_target = np.clip(raw_score / 50.0, -1.0, 1.0) * confidence
        else:
            n_valid = mask.sum()
            v_target = np.float32((1.0 - (n_valid / 235.0)) * confidence)

        return (torch.tensor(obs), torch.tensor(mask), torch.tensor(act).long(),
                torch.tensor(v_target, dtype=torch.float32))


def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            t = torch.zeros(1, device='mps')
            del t
            return torch.device('mps')
        except RuntimeError:
            pass
    return torch.device('cpu')


if __name__ == '__main__':
    device = get_device()
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f'Device: {device}, AMP: {use_amp}')

    split = 0.9
    train_ds = SLDataset('data', 0.0, split)
    val_ds = SLDataset('data', split, 1.0)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0,
                            pin_memory=False)

    model = CNNModel(in_channels=FeatureAgent.OBS_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: Cross-attention, Params: {n_params/1e6:.2f}M')

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, epochs=50,
        steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos')

    ckpt_path = 'checkpoint/sl_pretrained.pt'
    start_epoch, best_acc = 0, 0.0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler'): scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1; best_acc = ckpt['best_acc']
        print(f'Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}')

    interrupted = False
    def on_sig(s, f):
        global interrupted; interrupted = True
    signal.signal(signal.SIGINT, on_sig); signal.signal(signal.SIGTERM, on_sig)

    patience, no_improve = 7, 0
    value_weight = 0.3

    for epoch in range(start_epoch, 50):
        if interrupted: break
        model.train()
        total_loss, total_p_loss, total_v_loss = 0.0, 0.0, 0.0

        for obs, mask, act, v_target in train_loader:
            if interrupted: break
            obs, mask, act, v_target = obs.to(device), mask.to(device), \
                                       act.to(device), v_target.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits, values = model({'observation': obs, 'action_mask': mask})
                    p_loss = F.cross_entropy(logits, act, label_smoothing=0.1)
                    v_loss = F.mse_loss(values.squeeze(-1), v_target)
                    loss = p_loss + value_weight * v_loss
            else:
                logits, values = model({'observation': obs, 'action_mask': mask})
                p_loss = F.cross_entropy(logits, act, label_smoothing=0.1)
                v_loss = F.mse_loss(values.squeeze(-1), v_target)
                loss = p_loss + value_weight * v_loss

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()

        if interrupted: break

        value_weight = max(0.05, value_weight * 0.95)

        model.eval(); correct = 0
        with torch.no_grad():
            for obs, mask, act, _ in val_loader:
                obs, mask, act = obs.to(device), mask.to(device), act.to(device)
                logits, _ = model({'observation': obs, 'action_mask': mask})
                correct += (logits.argmax(dim=1) == act).sum().item()

        acc = correct / len(val_ds)
        n = len(train_loader)
        print(f'Ep {epoch+1}: loss={total_loss/n:.4f} p={total_p_loss/n:.4f} '
              f'v={total_v_loss/n:.4f} acc={acc:.4f} vw={value_weight:.3f}')

        if acc > best_acc:
            best_acc = acc; no_improve = 0
            os.makedirs('checkpoint', exist_ok=True)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_acc': best_acc},
                       ckpt_path)
            torch.save(model.state_dict(), 'checkpoint/model_weights.pt')
            print('  -> Best saved!')
        else:
            no_improve += 1
            if no_improve >= patience:
                print('Early stop')
                break

        # 每 epoch 存一份中间检查点
        epoch_path = f'checkpoint/sl_epoch_{epoch+1}.pt'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch,
                    'best_acc': best_acc, 'val_acc': acc}, epoch_path)

    if interrupted:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch if 'epoch' in dir() else start_epoch,
                    'best_acc': best_acc}, ckpt_path)
    print(f'Done. best_acc={best_acc:.4f}')
