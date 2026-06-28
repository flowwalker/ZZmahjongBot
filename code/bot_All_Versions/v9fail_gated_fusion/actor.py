"""Actor — 收集辅助任务标签"""
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
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda s, f: self.stop_event.set())
        torch.set_num_threads(1)

        model_pool = ModelPoolClient(self.config['model_pool_name'])
        try:
            model = CNNModel(in_channels=FeatureAgent.OBS_SIZE)
            model = model.to('cpu'); model.eval()

            version = model_pool.get_latest_model()
            sd = model_pool.load_model(version)
            if sd is not None: model.load_state_dict(sd, strict=False)

            env = MahjongGBEnv(config={'agent_clz': FeatureAgent, 'duplicate': True})
            temp = self.config.get('temperature', 1.0)
            use_aux = self.config.get('use_aux_tasks', False)

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
                ep_data = {a: {'obs': [], 'mask': [], 'act': [], 'rew': []} for a in env.agent_names}
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
                            logits, _ = model({'observation': ot, 'action_mask': mt})
                            if temp != 1.0: logits = logits / temp
                            lp = F.log_softmax(logits, dim=-1)
                            pr = torch.exp(lp)
                            action = torch.multinomial(pr, 1).item()
                        actions[agent_name] = action
                        ep_data[agent_name]['act'].append(action)
                    next_obs, rewards, done = env.step(actions)
                    steps += 1
                    for a in rewards: ep_data[a]['rew'].append(rewards[a])
                    obs = next_obs

                if self.stop_event.is_set(): break

                gamma = self.config['gamma']
                for agent_name, data in ep_data.items():
                    ml = min(len(data['obs']), len(data['act']), len(data['rew']))
                    if ml == 0: continue
                    ob = np.stack(data['obs'][:ml]); mk = np.stack(data['mask'][:ml])
                    ac = np.array(data['act'][:ml], dtype=np.int64)
                    rw = np.array(data['rew'][:ml], dtype=np.float32)

                    returns = np.zeros_like(rw); run = 0
                    for t in range(ml - 1, -1, -1):
                        run = rw[t] + gamma * run; returns[t] = run
                    adv = returns - returns.mean()
                    if adv.std() > 1e-8: adv = adv / (adv.std() + 1e-8)

                    payload = {
                        'state': {'observation': ob, 'action_mask': mk},
                        'action': ac, 'adv': adv.astype(np.float32),
                        'target': returns.astype(np.float32)
                    }
                    while not self.stop_event.is_set():
                        if self.replay_buffer.push(payload, timeout=0.5): break

                if episode % 10 == 0:
                    print(f'{self.name} Ep {episode} M{version["id"]} S{steps}')
        finally:
            model_pool.close()
