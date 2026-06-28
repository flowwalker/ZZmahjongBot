"""PPO Learner (Freeze P, Train V first) — 防 RL 初期随机 V 头污染 Policy

与 learner.py 的区别：
  初始阶段冻结 policy head，仅优化 value head。
  当 value loss 的 EMA 降至 v_loss_threshold 以下时自动解冻，正常 PPO。

用法：train.py 中 from learner_freezeV import Learner
      python3 train.py --v_loss_threshold 0.5 --v_only_lr 5e-4
"""
from multiprocessing import Process, Event
import time, os, numpy as np, torch
from torch.nn import functional as F
import signal

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel
from feature import FeatureAgent


def _get_device(config):
    req = str(config.get('device', 'cpu')).strip().lower()
    if req == 'auto':
        if torch.cuda.is_available(): return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(req)


class Learner(Process):
    def __init__(self, config, replay_buffer):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.stop_event = Event()

    def stop(self): self.stop_event.set()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda s, f: self.stop_event.set())

        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        try:
            device = _get_device(self.config)
            print(f'[Learner-FreezeV] Device: {device}', flush=True)

            if device.type == 'mps':
                try:
                    test_t = torch.zeros(1, device=device)
                    del test_t
                except RuntimeError:
                    print('[Learner-FreezeV] MPS init failed, falling back to CPU', flush=True)
                    device = torch.device('cpu')

            model = CNNModel(in_channels=FeatureAgent.OBS_SIZE)
            pp = self.config.get('pretrained_path', '')
            if pp and os.path.exists(pp):
                state = torch.load(pp, map_location='cpu')
                if isinstance(state, dict) and 'model' in state: state = state['model']
                model.load_state_dict(state, strict=False)
                print(f'Loaded pretrained from {pp}', flush=True)

            model_pool.push({k: v.cpu() for k, v in model.state_dict().items()})
            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
            use_aux = self.config.get('use_aux_tasks', False)
            aux_w = self.config.get('aux_weights', {'win_prob': 0.1, 'phase': 0.03, 'shanten': 0.1})

            v_loss_threshold = self.config.get('v_loss_threshold', 0.5)   # EMA 低于此值即解冻
            min_freeze_iters = self.config.get('min_freeze_iters', 10)     # 至少跑这么多轮（防止单批运气）
            max_freeze_iters = self.config.get('max_freeze_iters', 500)    # 安全上限
            v_only_lr = self.config.get('v_only_lr', self.config['lr'])
            ema_decay = 0.9   # EMA 平滑系数

            while self.replay_buffer.size() < self.config['min_sample'] and not self.stop_event.is_set():
                time.sleep(0.1)

            cur_time, iterations = time.time(), 0
            kl_thresh = self.config.get('kl_threshold', 0.02)
            phase = 'freeze'   # freeze → normal
            ema_v_loss = None  # EMA of value loss

            while not self.stop_event.is_set():
                batch = self.replay_buffer.sample(self.config['batch_size'])
                obs = torch.tensor(batch['state']['observation'], dtype=torch.float32).to(device)
                mask = torch.tensor(batch['state']['action_mask'], dtype=torch.float32).to(device)
                states = {'observation': obs, 'action_mask': mask}
                actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
                advs = torch.tensor(batch['adv'], dtype=torch.float32).to(device)
                targets = torch.tensor(batch['target'], dtype=torch.float32).to(device)

                if phase == 'freeze':
                    for name, param in model.named_parameters():
                        if name.startswith('_value'):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

                    if iterations == 0:
                        v_params = [p for n, p in model.named_parameters() if n.startswith('_value')]
                        optimizer = torch.optim.Adam(v_params, lr=v_only_lr)
                        n_v = sum(p.numel() for p in v_params)
                        print(f'[Learner-FreezeV] Phase=FREEZE | threshold={v_loss_threshold} '
                              f'min={min_freeze_iters} max={max_freeze_iters} | '
                              f'V params={n_v} lr={v_only_lr}', flush=True)

                    logits, values = model(states)
                    vs = values.squeeze(-1)
                    v_loss_val = F.mse_loss(vs, targets).item()
                    loss = self.config['value_coeff'] * F.mse_loss(vs, targets)

                    # EMA 平滑
                    if ema_v_loss is None:
                        ema_v_loss = v_loss_val
                    else:
                        ema_v_loss = ema_decay * ema_v_loss + (1 - ema_decay) * v_loss_val

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get('max_grad_norm', 0.5))
                    optimizer.step()

                    if iterations >= min_freeze_iters and ema_v_loss < v_loss_threshold:
                        for p in model.parameters():
                            p.requires_grad = True
                        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
                        print(f'[Learner-FreezeV] Iter {iterations}: V converged '
                              f'(EMA v_loss={ema_v_loss:.4f} < {v_loss_threshold}), unfreeze → NORMAL',
                              flush=True)
                        phase = 'normal'
                    elif iterations >= max_freeze_iters:
                        for p in model.parameters():
                            p.requires_grad = True
                        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
                        print(f'[Learner-FreezeV] Iter {iterations}: max_freeze reached '
                              f'(EMA v_loss={ema_v_loss:.4f}), unfreeze → NORMAL', flush=True)
                        phase = 'normal'

                    if iterations % 10 == 0:
                        print('Iter %d [V-ONLY] | buf %d/%d | v %.4f | ema_v %.4f | thresh %.3f' % (
                            iterations, self.replay_buffer.stats['sample_in'],
                            self.replay_buffer.stats['sample_out'],
                            v_loss_val, ema_v_loss, v_loss_threshold), flush=True)

                else:
                    with torch.no_grad():
                        old_logits, old_values = model(states)
                        old_logp_all = F.log_softmax(old_logits, dim=-1)
                        old_log_probs = old_logp_all.gather(1, actions).detach()
                        old_values = old_values.squeeze(-1).detach()

                    if advs.std() > 1e-8:
                        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                    total_p, total_v, total_e, total_kl = 0, 0, 0, 0
                    early_stop = False

                    for epoch in range(self.config['epochs']):
                        if use_aux:
                            logits, values, aux = model(states, return_aux=True)
                        else:
                            logits, values = model(states)
                            aux = None

                        logp_all = F.log_softmax(logits, dim=-1)
                        log_probs = logp_all.gather(1, actions).squeeze(-1)
                        probs_all = torch.exp(logp_all)

                        ratio = torch.exp(log_probs - old_log_probs.squeeze(-1))
                        with torch.no_grad():
                            kl = (old_log_probs.squeeze(-1) - log_probs).mean().item()
                            total_kl = kl
                        if kl > kl_thresh * 1.5 and epoch > 0:
                            early_stop = True
                            break

                        cr = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip'])
                        policy_loss = -torch.mean(torch.min(ratio * advs, cr * advs))

                        vs = values.squeeze(-1)
                        vc = old_values + torch.clamp(vs - old_values,
                                                       -self.config.get('value_clip', 0.2),
                                                       self.config.get('value_clip', 0.2))
                        value_loss = torch.mean(torch.max(
                            F.mse_loss(vs, targets, reduction='none'),
                            F.mse_loss(vc, targets, reduction='none')))

                        ent_dist = torch.distributions.Categorical(probs=probs_all)
                        entropy = ent_dist.entropy().mean()

                        loss = policy_loss + self.config['value_coeff'] * value_loss \
                               + self.config['entropy_coeff'] * (-entropy)

                        if use_aux and aux and 'aux_labels' in batch:
                            al = batch['aux_labels']
                            if 'win_prob' in al:
                                wpt = torch.tensor(al['win_prob'], dtype=torch.float32).to(device)
                                loss += aux_w.get('win_prob', 0.1) * F.mse_loss(
                                    aux['win_prob'].squeeze(-1), wpt)
                            if 'phase' in al:
                                pt = torch.tensor(al['phase'], dtype=torch.long).to(device)
                                loss += aux_w.get('phase', 0.03) * F.cross_entropy(aux['phase'], pt)
                            if 'shanten' in al:
                                st = torch.tensor(al['shanten'], dtype=torch.float32).to(device)
                                loss += aux_w.get('shanten', 0.1) * F.mse_loss(
                                    aux['shanten'].squeeze(-1), st)

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get('max_grad_norm', 0.5))
                        optimizer.step()

                        total_p, total_v, total_e = policy_loss.item(), value_loss.item(), entropy.item()

                    if iterations % 10 == 0:
                        print('Iter %d | buf %d/%d | p %.4f | v %.4f | ent %.4f | kl %.4f | es %s' % (
                            iterations, self.replay_buffer.stats['sample_in'],
                            self.replay_buffer.stats['sample_out'],
                            total_p, total_v, total_e, total_kl, 'Y' if early_stop else 'N'), flush=True)

                model_pool.push({k: v.detach().cpu() for k, v in model.state_dict().items()})

                t = time.time()
                if t - cur_time > self.config['ckpt_save_interval']:
                    path = self.config['ckpt_save_path'] + 'model_%d.pt' % iterations
                    os.makedirs(self.config['ckpt_save_path'], exist_ok=True)
                    torch.save({'model': model.state_dict(), 'iteration': iterations, 'phase': phase}, path)
                    cur_time = t
                iterations += 1
        finally:
            model_pool.close()
