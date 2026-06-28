"""Cross-attention Actor — GAE + Value Collection + Robust Error Handling"""
from multiprocessing import Process, Event
import numpy as np, torch, torch.nn.functional as F, signal

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel


class Actor(Process):
    def __init__(self, config, replay_buffer):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        self.stop_event = Event()

    def stop(self): self.stop_event.set()

    def run(self):
        import sys
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda s, f: self.stop_event.set())
        torch.set_num_threads(1)

        print(f'[{self.name}] Starting...', flush=True)
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        print(f'[{self.name}] ModelPool connected.', flush=True)
        try:
            model = CNNModel(in_channels=FeatureAgent.OBS_SIZE)
            model = model.to('cpu')
            model.eval()
            print(f'[{self.name}] Model loaded.', flush=True)

            version = model_pool.get_latest_model()
            sd = model_pool.load_model(version)
            if sd is not None:
                model.load_state_dict(sd, strict=False)

            env = MahjongGBEnv(config={'agent_clz': FeatureAgent, 'duplicate': True})
            temp = self.config.get('temperature', 1.0)

            for episode in range(self.config['episodes_per_actor']):
                if self.stop_event.is_set(): break

                latest = model_pool.get_latest_model()
                if latest['id'] > version['id']:
                    sd = model_pool.load_model(latest)
                    if sd is not None:
                        try: model.load_state_dict(sd, strict=False)
                        except RuntimeError as e: print(f'[{self.name}] Load err: {e}')
                    version = latest

                obs = env.reset()
                ep_data = {a: {'obs': [], 'mask': [], 'act': [], 'rew': [], 'val': []}
                            for a in env.agent_names}
                done, steps = False, 0

                while not done and not self.stop_event.is_set() and steps < 200:
                    actions = {}
                    for agent_name in obs:
                        s = obs[agent_name]
                        ep_data[agent_name]['obs'].append(s['observation'])
                        ep_data[agent_name]['mask'].append(s['action_mask'])

                        ot = torch.tensor(s['observation'], dtype=torch.float32).unsqueeze(0).cpu()
                        mt = torch.tensor(s['action_mask'], dtype=torch.float32).unsqueeze(0).cpu()

                        with torch.no_grad():
                            logits, values = model({'observation': ot, 'action_mask': mt})
                            if temp != 1.0: logits = logits / temp
                            lp = F.log_softmax(logits, dim=-1)
                            pr = torch.exp(lp)
                            action = torch.multinomial(pr, 1).item()
                        actions[agent_name] = action
                        ep_data[agent_name]['act'].append(action)
                        ep_data[agent_name]['val'].append(values.squeeze(-1).item())

                    next_obs, rewards, done = env.step(actions)
                    steps += 1
                    for a in rewards: ep_data[a]['rew'].append(rewards[a])
                    obs = next_obs

                if self.stop_event.is_set(): break

                gamma = self.config['gamma']
                lam = self.config.get('lambda', 0.95)
                for agent_name, data in ep_data.items():
                    ml = min(len(data['obs']), len(data['act']), len(data['rew']), len(data['val']))
                    if ml == 0: continue
                    ob = np.stack(data['obs'][:ml])
                    mk = np.stack(data['mask'][:ml])
                    ac = np.array(data['act'][:ml], dtype=np.int64)
                    rw = np.array(data['rew'][:ml], dtype=np.float32)
                    vals = np.array(data['val'][:ml], dtype=np.float32)

                    # GAE computation
                    advantages = np.zeros_like(rw)
                    gae = 0
                    for t in range(ml - 1, -1, -1):
                        next_val = vals[t + 1] if t < ml - 1 else 0.0
                        delta = rw[t] + gamma * next_val - vals[t]
                        gae = delta + gamma * lam * gae
                        advantages[t] = gae

                    # CRITICAL: Clip advantages to prevent explosion from untrained value head
                    advantages = np.clip(advantages, -10, 10)

                    # returns = advantages + values (target for value function)
                    targets = advantages + vals

                    # Normalize advantages (only for policy gradient, not value target)
                    if advantages.std() > 1e-8:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    payload = {
                        'state': {'observation': ob, 'action_mask': mk},
                        'action': ac, 'adv': advantages.astype(np.float32),
                        'target': targets.astype(np.float32)
                    }
                    while not self.stop_event.is_set():
                        if self.replay_buffer.push(payload, timeout=0.5): break

                if episode % 10 == 0:
                    print(f'{self.name} Ep {episode} M{version["id"]} S{steps}')

        except Exception as e:
            import traceback
            print(f'[{self.name}] CRASH: {e}', flush=True)
            traceback.print_exc()
        finally:
            model_pool.close()
