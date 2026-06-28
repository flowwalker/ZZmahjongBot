"""
修复版 Actor — 支持 V5V6 合并模型的 mode 参数

关键修复：
1. 模型显式移到 CPU
2. 数值稳定的 log_softmax + multinomial 采样
3. 使用蒙特卡洛 returns 替代 TD target
4. episode 长度一致性检查和截断

"""

from multiprocessing import Process, Event
import numpy as np
import torch
import torch.nn.functional as F
import signal

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import V5V6Model


class Actor(Process):
    def __init__(self, config, replay_buffer):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        self.stop_event = Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda s, f: self.stop_event.set())
        torch.set_num_threads(1)

        model_pool = ModelPoolClient(self.config['model_pool_name'])
        try:
            model = V5V6Model(in_channels=FeatureAgent.OBS_SIZE)
            model = model.to('cpu')
            model.eval()

            version = model_pool.get_latest_model()
            state_dict = model_pool.load_model(version)
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=False)

            env = MahjongGBEnv(config={'agent_clz': FeatureAgent, 'duplicate': True})
            temperature = self.config.get('temperature', 1.0)

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
                ep_data = {agent_name: {
                    'state': {'observation': [], 'action_mask': []},
                    'action': [], 'reward': []
                } for agent_name in env.agent_names}

                done = False
                step_count = 0
                while not done and not self.stop_event.is_set() and step_count < 200:
                    actions = {}
                    for agent_name in obs:
                        state = obs[agent_name]
                        ep_data[agent_name]['state']['observation'].append(state['observation'])
                        ep_data[agent_name]['state']['action_mask'].append(state['action_mask'])

                        obs_t = torch.tensor(state['observation'], dtype=torch.float32).unsqueeze(0).cpu()
                        mask_t = torch.tensor(state['action_mask'], dtype=torch.float32).unsqueeze(0).cpu()

                        with torch.no_grad():
                            logits, _ = model({'observation': obs_t, 'action_mask': mask_t})
                            if temperature != 1.0:
                                logits = logits / temperature
                            logp_all = F.log_softmax(logits, dim=-1)
                            probs = torch.exp(logp_all)
                            action = torch.multinomial(probs, 1).item()
                        actions[agent_name] = action
                        ep_data[agent_name]['action'].append(action)

                    next_obs, rewards, done = env.step(actions)
                    step_count += 1
                    for agent_name in rewards:
                        ep_data[agent_name]['reward'].append(rewards[agent_name])
                    obs = next_obs

                if self.stop_event.is_set(): break

                # Post-process: compute Monte Carlo returns
                gamma = self.config['gamma']
                for agent_name, data in ep_data.items():
                    n_s = len(data['state']['observation'])
                    n_a = len(data['action'])
                    n_r = len(data['reward'])
                    ml = min(n_s, n_a, n_r)
                    if ml == 0: continue

                    ob = np.stack(data['state']['observation'][:ml])
                    mk = np.stack(data['state']['action_mask'][:ml])
                    ac = np.array(data['action'][:ml], dtype=np.int64)
                    rw = np.array(data['reward'][:ml], dtype=np.float32)

                    # Monte Carlo returns
                    returns = np.zeros_like(rw)
                    running = 0
                    for t in range(ml - 1, -1, -1):
                        running = rw[t] + gamma * running
                        returns[t] = running

                    adv = returns - returns.mean()
                    if adv.std() > 1e-8:
                        adv = adv / (adv.std() + 1e-8)

                    payload = {
                        'state': {'observation': ob, 'action_mask': mk},
                        'action': ac, 'adv': adv.astype(np.float32),
                        'target': returns.astype(np.float32)
                    }
                    while not self.stop_event.is_set():
                        if self.replay_buffer.push(payload, timeout=0.5): break

                if episode % 10 == 0:
                    print(f'{self.name} Ep {episode} Model {version["id"]} Steps {step_count}')

        finally:
            model_pool.close()
