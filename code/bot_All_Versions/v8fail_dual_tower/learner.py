"""V5V6 Learner — PPO with separate policy/value optimizers and optional auxiliary losses."""

from multiprocessing import Process, Event
import time
import os
import numpy as np
import torch
from torch.nn import functional as F
import signal

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from feature import FeatureAgent
from model import V5V6Model


class Learner(Process):
    def __init__(self, config, replay_buffer):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.stop_event = Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda s, f: self.stop_event.set())

        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])

        try:
            # 设备选择
            req = str(self.config.get('device', 'cpu')).strip().lower()
            if req == 'auto':
                if torch.cuda.is_available(): device = torch.device('cuda')
                elif torch.backends.mps.is_available(): device = torch.device('mps')
                else: device = torch.device('cpu')
            else:
                device = torch.device(req)
            print(f'[Learner] Device: {device}')

            model = V5V6Model(in_channels=FeatureAgent.OBS_SIZE)

            # 加载 SL 预训练权重
            pp = self.config.get('pretrained_path', '')
            if pp and os.path.exists(pp):
                state = torch.load(pp, map_location='cpu')
                if isinstance(state, dict) and 'model' in state:
                    state = state['model']
                model.load_state_dict(state, strict=False)
                print(f'[Learner] Loaded SL pretrained from {pp}')

            model_pool.push({k: v.cpu() for k, v in model.state_dict().items()})
            model = model.to(device)

            # 双塔独立学习率
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

            lr_p = self.config.get('lr_policy', self.config['lr'] * 3)
            lr_v = self.config.get('lr_value', self.config['lr'])

            optimizer = torch.optim.Adam([
                {'params': policy_params, 'lr': lr_p},
                {'params': value_params, 'lr': lr_v},
            ])

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100000, eta_min=min(lr_p, lr_v) * 0.1
            )

            # 等待初始样本
            while self.replay_buffer.size() < self.config['min_sample'] and not self.stop_event.is_set():
                time.sleep(0.1)

            cur_time = time.time()
            iterations = 0
            kl_thresh = self.config.get('kl_threshold', 0.02)
            use_aux = self.config.get('use_aux_tasks', False)

            # 辅助任务损失权重
            aux_weights = {
                'win_prob': self.config.get('aux_win_prob_weight', 0.1),
                'opp_action': self.config.get('aux_opp_action_weight', 0.05),
                'phase': self.config.get('aux_phase_weight', 0.03),
                'shanten': self.config.get('aux_shanten_weight', 0.1),
            }

            while not self.stop_event.is_set():
                batch = self.replay_buffer.sample(self.config['batch_size'])

                obs = torch.tensor(batch['state']['observation'], dtype=torch.float32).to(device)
                mask = torch.tensor(batch['state']['action_mask'], dtype=torch.float32).to(device)
                states = {'observation': obs, 'action_mask': mask}
                actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
                advs = torch.tensor(batch['adv'], dtype=torch.float32).to(device)
                targets = torch.tensor(batch['target'], dtype=torch.float32).to(device)

                with torch.no_grad():
                    old_logits, old_values = model(states)
                    old_logp_all = F.log_softmax(old_logits, dim=-1)
                    old_log_probs = old_logp_all.gather(1, actions).detach()
                    old_values = old_values.squeeze(-1).detach()

                # Advantage 归一化
                if advs.std() > 1e-8:
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                total_p_loss = 0
                total_v_loss = 0
                total_ent = 0
                total_kl = 0
                total_aux = 0
                early_stop = False

                for epoch in range(self.config['epochs']):
                    # 前向
                    if use_aux and hasattr(model, '_aux_win_prob'):
                        logits, values, aux = model(states, mode='both', return_aux=True)
                    else:
                        logits, values = model(states, mode='both')
                        aux = None

                    logp_all = F.log_softmax(logits, dim=-1)
                    log_probs = logp_all.gather(1, actions).squeeze(-1)
                    probs = torch.exp(logp_all)

                    # Ratio
                    ratio = torch.exp(log_probs - old_log_probs.squeeze(-1))

                    # KL early stopping
                    with torch.no_grad():
                        kl = (old_log_probs.squeeze(-1) - log_probs).mean().item()
                        total_kl = kl
                    if kl > kl_thresh * 1.5 and epoch > 0:
                        early_stop = True
                        break

                    # PPO clipped loss
                    cr = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip'])
                    surr1 = ratio * advs
                    surr2 = cr * advs
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss with clipping
                    vs = values.squeeze(-1)
                    vc = old_values + torch.clamp(vs - old_values,
                                                   -self.config.get('value_clip', 0.2),
                                                   self.config.get('value_clip', 0.2))
                    v_loss1 = F.mse_loss(vs, targets)
                    v_loss2 = F.mse_loss(vc, targets)
                    value_loss = torch.max(v_loss1, v_loss2)

                    # Entropy
                    ent_dist = torch.distributions.Categorical(probs=probs)
                    entropy = ent_dist.entropy().mean()

                    # Total loss
                    loss = (policy_loss
                            + self.config['value_coeff'] * value_loss
                            + self.config['entropy_coeff'] * (-entropy))

                    # 辅助任务损失（仅在启用时）
                    if aux and use_aux:
                        # 从 batch 中提取辅助标签（如果有）
                        # 如果没有标签，则跳过
                        aux_loss = torch.tensor(0.0, device=device)
                        if 'win_prob_target' in batch:
                            wpt = torch.tensor(batch['win_prob_target'], dtype=torch.float32).to(device)
                            aux_loss += aux_weights['win_prob'] * F.mse_loss(
                                aux['win_prob'].squeeze(-1), wpt)
                        if 'phase_target' in batch:
                            pt = torch.tensor(batch['phase_target'], dtype=torch.long).to(device)
                            aux_loss += aux_weights['phase'] * F.cross_entropy(aux['phase'], pt)
                        if 'shanten_target' in batch:
                            st = torch.tensor(batch['shanten_target'], dtype=torch.float32).to(device)
                            aux_loss += aux_weights['shanten'] * F.mse_loss(
                                aux['shanten'].squeeze(-1), st)
                        loss += aux_loss
                        total_aux = aux_loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    self.config.get('max_grad_norm', 0.5))
                    optimizer.step()

                    total_p_loss = policy_loss.item()
                    total_v_loss = value_loss.item()
                    total_ent = entropy.item()

                scheduler.step()
                model_pool.push({k: v.detach().cpu() for k, v in model.state_dict().items()})

                if iterations % 10 == 0:
                    lrs = [g['lr'] for g in optimizer.param_groups]
                    print('Iter %d | buf %d/%d | p_loss %.4f | v_loss %.4f | '
                          'ent %.4f | kl %.4f | aux %.4f | lr %.2e/%.2e | es %s' % (
                        iterations, self.replay_buffer.stats['sample_in'],
                        self.replay_buffer.stats['sample_out'],
                        total_p_loss, total_v_loss, total_ent, total_kl, total_aux,
                        lrs[0], lrs[1], 'Y' if early_stop else 'N'))

                t = time.time()
                if t - cur_time > self.config['ckpt_save_interval']:
                    path = self.config['ckpt_save_path'] + 'model_rl_%d.pt' % iterations
                    os.makedirs(self.config['ckpt_save_path'], exist_ok=True)
                    torch.save({'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'iteration': iterations}, path)
                    cur_time = t

                iterations += 1

        finally:
            model_pool.close()
