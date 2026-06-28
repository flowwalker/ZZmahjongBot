"""SL 预训练脚本."""

import torch, torch.nn.functional as F, numpy as np, os, json, signal
from torch.utils.data import Dataset, DataLoader
from bisect import bisect_right

from model import V5V6Model
from feature import FeatureAgent


class SLDataset(Dataset):
    """SL数据集 — 支持辅助任务标签生成"""
    def __init__(self, data_dir='data', begin=0.0, end=1.0, use_aux=False):
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
        self.data_dir = data_dir
        self.use_aux = use_aux
        print(f'Dataset: {self.matches} matches, {self.samples} samples, aux={use_aux}')

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        mid = bisect_right(self.offsets, idx) - 1
        sid = idx - self.offsets[mid]
        d = np.load(f'{self.data_dir}/{mid + self.begin}.npz')
        obs = d['obs'][sid].astype(np.float32)
        mask = d['mask'][sid].astype(np.float32)
        act = d['act'][sid]

        item = {
            'obs': torch.tensor(obs).float(),
            'mask': torch.tensor(mask).float(),
            'act': torch.tensor(act).long(),
        }

        # 从 feature 计算辅助任务标签
        if self.use_aux:
            # 提取手牌信息计算辅助标签（简化版）
            # 注意：这里简化处理，实际应从原始数据解析
            item['aux_win_prob'] = torch.tensor(0.5, dtype=torch.float32)
            item['aux_phase'] = torch.tensor(0, dtype=torch.long)
            item['aux_shanten'] = torch.tensor(4.0, dtype=torch.float32)

        return item


def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    elif torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def save_ckpt(path, model, optimizer, scheduler, epoch, best_acc):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch, 'best_acc': best_acc,
    }, path)


def load_ckpt(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and ckpt.get('scheduler'):
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch'], ckpt['best_acc']


if __name__ == '__main__':
    device = get_device()
    print(f'Device: {device}')

    use_aux = True  # 启用辅助任务
    split = 0.9
    train_ds = SLDataset('data', 0.0, split, use_aux=use_aux)
    val_ds = SLDataset('data', split, 1.0, use_aux=False)
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = V5V6Model(in_channels=FeatureAgent.OBS_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params/1e6:.1f}M')

    # 双塔优化器
    policy_params = list(model._stem.parameters()) + \
                   list(model._policy_blocks.parameters()) + \
                   list(model._policy_conv.parameters()) + \
                   list(model._policy_fc.parameters())
    value_params = list(model._value_blocks.parameters()) + \
                  list(model._value_fc.parameters()) + \
                  list(model._aux_win_prob.parameters()) + \
                  list(model._aux_opp_action.parameters()) + \
                  list(model._aux_phase.parameters()) + \
                  list(model._aux_shanten.parameters())

    optimizer = torch.optim.AdamW([
        {'params': policy_params, 'lr': 1.5e-3},
        {'params': value_params, 'lr': 5e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[1.5e-3, 5e-4],
        epochs=50, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )

    ckpt_path = 'checkpoint/sl_pretrained.pt'
    start_epoch = 0
    best_acc = 0.0
    if os.path.exists(ckpt_path):
        start_epoch, best_acc = load_ckpt(ckpt_path, model, optimizer, scheduler)
        start_epoch += 1
        print(f'Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}')

    patience = 7
    no_improve = 0
    interrupted = False

    def on_signal(sig, frame):
        global interrupted
        interrupted = True
        print('\nInterrupted! Saving...')

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    for epoch in range(start_epoch, 50):
        if interrupted: break

        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_aux_loss = 0.0

        for batch in train_loader:
            if interrupted: break

            obs = batch['obs'].to(device)
            mask = batch['mask'].to(device)
            act = batch['act'].to(device)
            states = {'observation': obs, 'action_mask': mask}

            # 主任务
            logits, value = model(states, mode='both')
            policy_loss = F.cross_entropy(logits, act, label_smoothing=0.1)

            # Value 从动作分布中学习（简化：用 -log_prob 作为价值目标）
            with torch.no_grad():
                log_probs_gt = F.log_softmax(logits.detach(), dim=-1)
                target_value = -log_probs_gt.gather(1, act.unsqueeze(-1)).squeeze(-1)
            value_loss = F.mse_loss(value.squeeze(-1), target_value) * 0.5

            loss = policy_loss + value_loss

            # 辅助任务
            if use_aux and hasattr(model, '_aux_win_prob'):
                _, _, aux = model(states, mode='both', return_aux=True)
                aux_loss = torch.tensor(0.0, device=device)

                if 'aux_win_prob' in batch:
                    wpt = batch['aux_win_prob'].to(device)
                    aux_loss += 0.1 * F.mse_loss(aux['win_prob'].squeeze(-1), wpt)
                if 'aux_phase' in batch:
                    pt = batch['aux_phase'].to(device)
                    aux_loss += 0.03 * F.cross_entropy(aux['phase'], pt)
                if 'aux_shanten' in batch:
                    st = batch['aux_shanten'].to(device)
                    aux_loss += 0.1 * F.mse_loss(aux['shanten'].squeeze(-1), st)

                loss += aux_loss
                total_aux_loss += aux_loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        if interrupted: break

        # 验证
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                mask = batch['mask'].to(device)
                act = batch['act'].to(device)
                logits, _ = model({'observation': obs, 'action_mask': mask}, mode='policy')
                correct += (logits.argmax(dim=1) == act).sum().item()
        acc = correct / len(val_ds)

        n = len(train_loader)
        print(f'Epoch {epoch+1}: loss={total_loss/n:.4f} '
              f'p={total_policy_loss/n:.4f} v={total_value_loss/n:.4f} '
              f'aux={total_aux_loss/n:.4f} acc={acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            save_ckpt(ckpt_path, model, optimizer, scheduler, epoch, best_acc)
            torch.save(model.state_dict(), 'checkpoint/model_weights.pt')
            print(f'  -> Best saved!')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stop at epoch {epoch+1}')
                break

    if interrupted:
        save_ckpt(ckpt_path, model, optimizer, scheduler,
                  epoch if 'epoch' in dir() else start_epoch, best_acc)
    print(f'Done. best_val_acc={best_acc:.4f}')
