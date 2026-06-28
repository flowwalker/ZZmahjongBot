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
        offsets = [0]
        for m in self.ms: offsets.append(offsets[-1] + m)
        self.offsets = offsets
        self.samples = sum(self.ms)
        print(f'Dataset: {len(self.ms)} matches, {self.samples} samples')

    def __len__(self): return self.samples
    def __getitem__(self, idx):
        mid = bisect_right(self.offsets, idx) - 1
        sid = idx - self.offsets[mid]
        d = np.load(f'{self.data_dir}/{mid}.npz')
        return (torch.tensor(d['obs'][sid].astype(np.float32)),
                torch.tensor(d['mask'][sid].astype(np.float32)),
                torch.tensor(d['act'][sid]).long())

def get_device():
    try:
        import torch_npu
        if torch.npu.is_available(): return torch.device('npu')
    except ImportError: pass
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

if __name__ == '__main__':
    device = get_device()
    use_amp = device.type in ('cuda', 'npu')
    scaler = torch.amp.GradScaler('npu') if device.type == 'npu' else (torch.cuda.amp.GradScaler() if use_amp else None)
    print(f'Device: {device}, AMP: {use_amp}')
    split = 0.9
    train_ds = SLDataset('data', 0.0, split)
    val_ds = SLDataset('data', split, 1.0)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = CNNModel(in_channels=FeatureAgent.OBS_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {os.environ.get("MODEL_VARIANT", "vultra")}, Params: {n_params/1e6:.1f}M')

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
    for epoch in range(start_epoch, 50):
        if interrupted: break
        model.train()
        total_loss = 0.0
        for obs, mask, act in train_loader:
            if interrupted: break
            obs, mask, act = obs.to(device), mask.to(device), act.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits, _ = model({'observation': obs, 'action_mask': mask})
                    loss = F.cross_entropy(logits, act, label_smoothing=0.1)
            else:
                logits, _ = model({'observation': obs, 'action_mask': mask})
                loss = F.cross_entropy(logits, act, label_smoothing=0.1)
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
        if interrupted: break

        model.eval(); correct = 0
        with torch.no_grad():
            for obs, mask, act in val_loader:
                obs, mask, act = obs.to(device), mask.to(device), act.to(device)
                logits, _ = model({'observation': obs, 'action_mask': mask})
                correct += (logits.argmax(dim=1) == act).sum().item()
        acc = correct / len(val_ds)
        print(f'Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f} acc={acc:.4f}')

        if acc > best_acc:
            best_acc = acc; no_improve = 0
            os.makedirs('checkpoint', exist_ok=True)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_acc': best_acc}, ckpt_path)
            torch.save(model.state_dict(), 'checkpoint/model_weights.pt')
            print('  -> Best saved!')
        else:
            no_improve += 1
            if no_improve >= patience: print('Early stop'); break

    if interrupted:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch if 'epoch' in dir() else start_epoch,
                    'best_acc': best_acc}, ckpt_path)
    print(f'Done. best_acc={best_acc:.4f}')
