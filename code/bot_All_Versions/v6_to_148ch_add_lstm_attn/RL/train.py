"""vfinal_v2 RL 训练入口"""
import os, time, argparse, signal
import torch, torch.multiprocessing as mp

from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner

def parse_args():
    p = argparse.ArgumentParser(description='vfinal_v2 RL Training')
    p.add_argument('--num_actors', type=int, default=16)
    p.add_argument('--device', default='auto')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--clip', type=float, default=0.15)
    p.add_argument('--entropy_coeff', type=float, default=0.005)
    p.add_argument('--value_coeff', type=float, default=0.5)
    p.add_argument('--kl_threshold', type=float, default=0.02)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--pretrained_path', default='checkpoint/sl_pretrained.pt')
    return p.parse_args()

def main():
    args = parse_args()
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    import uuid
    pool_name = 'vfinal-pool-' + uuid.uuid4().hex[:6]
    config = {
        'replay_buffer_size': 100000, 'replay_buffer_episode': 500,
        'model_pool_size': 20, 'model_pool_name': pool_name,
        'num_actors': args.num_actors, 'episodes_per_actor': 2000,
        'gamma': 0.99, 'lambda': 0.95, 'min_sample': 10000,
        'batch_size': args.batch_size, 'epochs': args.epochs,
        'clip': args.clip, 'lr': args.lr, 'value_coeff': args.value_coeff,
        'entropy_coeff': args.entropy_coeff, 'kl_threshold': args.kl_threshold,
        'value_clip': 0.2, 'max_grad_norm': 0.5, 'temperature': args.temperature,
        'reward_scale': 48.0,   # 缩小 reward 量级使 PPO 超参自然生效
        'device': args.device, 'ckpt_save_interval': 3600,
        'ckpt_save_path': os.path.join(os.path.dirname(__file__), 'checkpoint') + os.sep,
        'pretrained_path': args.pretrained_path if os.path.exists(args.pretrained_path) else '',
    }
    print(f'[Main] vfinal_v2 | Actors:{args.num_actors} | Dev:{args.device}')

    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    stop_event = mp.Event()
    def _stop(s, f): stop_event.set()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    actors, learner = [], None
    try:
        learner = Learner(config, replay_buffer)
        learner.start()
        time.sleep(2)
        for i in range(args.num_actors):
            cfg = dict(config); cfg['name'] = f'Actor-{i}'
            a = Actor(cfg, replay_buffer); a.start(); actors.append(a)

        step = 0
        while not stop_event.is_set():
            dead = [a for a in actors if not a.is_alive()]
            if dead:
                for i, a in enumerate(actors):
                    if not a.is_alive():
                        cfg = dict(config); cfg['name'] = f'Actor-{i}'
                        na = Actor(cfg, replay_buffer); na.start(); actors[i] = na
            if learner and not learner.is_alive(): stop_event.set(); break
            time.sleep(2); step += 1
            if step % 15 == 0: print(f'[Main] step={step} buf={replay_buffer.size()}')
    finally:
        stop_event.set()
        if learner: learner.stop(); learner.join(timeout=10)
        for a in actors: a.stop()
        for a in actors: a.join(timeout=5)
        for a in actors:
            if a.is_alive(): a.terminate(); a.join(timeout=1)
        replay_buffer.close()
        print('[Main] Done.')

if __name__ == '__main__':
    main()
