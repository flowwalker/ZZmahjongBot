"""vfinal_v2 Learner — PPO with KL early stopping, value clipping, and advantage normalization."""

from multiprocessing import Process, Event
import time, os, numpy as np, torch
from torch.nn import functional as F
import signal

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel
from feature import FeatureAgent

def _npu_available():
    try: import torch_npu
    except ImportError: return False
    return hasattr(torch, 'npu') and torch.npu.is_available() and torch.npu.is_available()


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
            req = str(self.config.get('device', 'cpu')).strip().lower()
            if req == 'auto':
                if _npu_available(): req = 'npu'
                elif torch.cuda.is_available(): req = 'cuda'
                elif torch.backends.mps.is_available(): req = 'mps'
                else: req = 'cpu'
            if req.startswith('npu') and not _npu_available():
                print('Warning: NPU unavailable, using CPU.', flush=True); req = 'cpu'
            if req.startswith('cuda') and not torch.cuda.is_available():
                print('Warning: CUDA unavailable, using CPU.', flush=True); req = 'cpu'
            if req.startswith('mps') and not torch.backends.mps.is_available():
                print('Warning: MPS unavailable, using CPU.', flush=True); req = 'cpu'
            device = torch.device(req)
            print(f'[Learner] Device: {device}', flush=True)

            model = CNNModel(in_channels=FeatureAgent.OBS_SIZE)

            pp = self.config.get('pretrained_path', '')
            if pp and os.path.exists(pp):
                state = torch.load(pp, map_location='cpu')
                if isinstance(state, dict) and 'model' in state:
                    state = state['model']
                model.load_state_dict(state, strict=False)
                print(f'Loaded pretrained from {pp}', flush=True)

            model_pool.push({k: v.cpu() for k, v in model.state_dict().items()})
            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])

            while self.replay_buffer.size() < self.config['min_sample'] and not self.stop_event.is_set():
                time.sleep(0.1)

            cur_time = time.time()
            iterations = 0
            kl_thresh = self.config.get('kl_threshold', 0.02)

            while not self.stop_event.is_set():
                batch = self.replay_buffer.sample(self.config['batch_size'])
                obs = torch.tensor(batch['state']['observation'], dtype=torch.float32).to(device)
                mask = torch.tensor(batch['state']['action_mask'], dtype=torch.float32).to(device)
                states = {'observation': obs, 'action_mask': mask}
                actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
                advs = torch.tensor(batch['adv'], dtype=torch.float32).to(device)
                targets = torch.tensor(batch['target'], dtype=torch.float32).to(device)
                # Use stored log probabilities from actor
                old_log_probs = torch.tensor(batch['logp'], dtype=torch.float32).unsqueeze(-1).to(device)
                with torch.no_grad():
                    _, old_values = model(states)
                    old_values = old_values.squeeze(-1).detach()

                if advs.std() > 1e-8:
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                total_p, total_v, total_e, total_kl = 0, 0, 0, 0
                early_stop = False

                for epoch in range(self.config['epochs']):
                    logits, values = model(states)
                    logp_all = F.log_softmax(logits, dim=-1)
                    log_probs = logp_all.gather(1, actions).squeeze(-1)
                    probs_all = torch.exp(logp_all)

                    ratio = torch.exp(log_probs - old_log_probs.squeeze(-1))

                    with torch.no_grad():
                        kl = (old_log_probs.squeeze(-1) - log_probs).mean().item()
                        total_kl = kl
                    if kl > kl_thresh * 1.5 and epoch > 0:
                        early_stop = True; break

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
                    entropy_loss = -torch.mean(ent_dist.entropy())

                    loss = policy_loss + self.config['value_coeff'] * value_loss \
                           + self.config['entropy_coeff'] * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get('max_grad_norm', 0.5))
                    optimizer.step()

                    total_p, total_v, total_e = policy_loss.item(), value_loss.item(), entropy_loss.item()

                model_pool.push({k: v.detach().cpu() for k, v in model.state_dict().items()})

                if iterations % 10 == 0:
                    print('Iter %d | buf %d/%d | p %.4f | v %.4f | ent %.4f | kl %.4f | es %s' % (
                        iterations, self.replay_buffer.stats['sample_in'],
                        self.replay_buffer.stats['sample_out'],
                        total_p, total_v, -total_e, total_kl, 'Y' if early_stop else 'N'), flush=True)

                t = time.time()
                if t - cur_time > self.config['ckpt_save_interval']:
                    path = self.config['ckpt_save_path'] + 'model_%d.pt' % iterations
                    os.makedirs(self.config['ckpt_save_path'], exist_ok=True)
                    torch.save({'model': model.state_dict(), 'iteration': iterations}, path)
                    cur_time = t
                iterations += 1
        finally:
            model_pool.close()
