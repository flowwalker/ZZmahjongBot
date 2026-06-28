"""
PPO RL 训练主入口 — V5V6 合并版

用法:
    python train.py                    # 默认 16 actors
    python train.py --num_actors 8     # 8 actors
    python train.py --device mps       # MPS GPU
"""

import os, time, argparse, signal
import torch
import torch.multiprocessing as mp

from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner


def parse_args():
    p = argparse.ArgumentParser(description='V5V6 Merged RL Training')
    p.add_argument('--num_actors', type=int, default=16)
    p.add_argument('--device', default='auto', choices=['cpu', 'mps', 'cuda', 'auto'])
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_policy', type=float, default=3e-4)
    p.add_argument('--lr_value', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--min_sample', type=int, default=10000)
    p.add_argument('--max_steps', type=int, default=1_000_000)
    p.add_argument('--ckpt_interval', type=int, default=300)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--clip', type=float, default=0.15)
    p.add_argument('--entropy_coeff', type=float, default=0.005)
    p.add_argument('--value_coeff', type=float, default=0.5)
    p.add_argument('--kl_threshold', type=float, default=0.02)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--use_aux', action='store_true', help='RL阶段启用辅助任务')
    p.add_argument('--pretrained_path', default='checkpoint/sl_pretrained.pt')
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    config = {
        'replay_buffer_size': 100000,
        'replay_buffer_episode': 500,
        'model_pool_size': 20,
        'model_pool_name': 'v5v6-model-pool',
        'num_actors': args.num_actors,
        'episodes_per_actor': 2000,
        'gamma': args.gamma,
        'lambda': 0.95,
        'min_sample': args.min_sample,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'clip': args.clip,
        'lr': args.lr,
        'lr_policy': args.lr_policy,
        'lr_value': args.lr_value,
        'value_coeff': args.value_coeff,
        'entropy_coeff': args.entropy_coeff,
        'kl_threshold': args.kl_threshold,
        'value_clip': 0.2,
        'max_grad_norm': 0.5,
        'temperature': args.temperature,
        'use_aux_tasks': args.use_aux,
        'device': args.device,
        'ckpt_save_interval': args.ckpt_interval,
        'ckpt_save_path': os.path.join(base_dir, 'checkpoint') + os.sep,
        'pretrained_path': args.pretrained_path if os.path.exists(args.pretrained_path) else '',
    }

    print(f'[Main] V5V6 Merged RL Training')
    print(f'  Actors: {args.num_actors} | Device: {args.device}')
    print(f'  Policy LR: {args.lr_policy} | Value LR: {args.lr_value}')
    print(f'  Aux tasks: {args.use_aux}')

    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    stop_event = mp.Event()

    def _request_stop(s, f):
        if not stop_event.is_set():
            print(f'\n[Main] Signal {s}, shutting down...')
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    actors = []
    learner = None

    try:
        learner = Learner(config, replay_buffer)
        learner.start()
        print('[Main] Learner started')
        time.sleep(2)

        print(f'[Main] Starting {args.num_actors} actors...')
        for i in range(args.num_actors):
            cfg = dict(config)
            cfg['name'] = f'Actor-{i}'
            actor = Actor(cfg, replay_buffer)
            actor.start()
            actors.append(actor)

        step = 0
        while not stop_event.is_set() and step < args.max_steps:
            dead = [a for a in actors if not a.is_alive()]
            if dead:
                print(f'[CRITICAL] {len(dead)} actors died! Restarting...')
                for i, a in enumerate(actors):
                    if not a.is_alive():
                        cfg = dict(config)
                        cfg['name'] = f'Actor-{i}'
                        new_a = Actor(cfg, replay_buffer)
                        new_a.start()
                        actors[i] = new_a

            if learner and not learner.is_alive():
                print('[CRITICAL] Learner died!')
                stop_event.set()
                break

            time.sleep(2.0)
            step += 1
            if step % 15 == 0:
                buf_size = replay_buffer.size()
                alive = sum(1 for a in actors if a.is_alive())
                print(f'[Main] step={step} buffer={buf_size} alive={alive}/{len(actors)}')

    except KeyboardInterrupt:
        print('[Main] Interrupted.')
    finally:
        print('[Main] Cleaning up...')
        stop_event.set()
        if learner:
            learner.stop()
            learner.join(timeout=10)
            if learner.is_alive():
                learner.terminate()
                learner.join(timeout=2)
        for a in actors:
            a.stop()
        for a in actors:
            a.join(timeout=5)
        for a in actors:
            if a.is_alive():
                a.terminate()
                a.join(timeout=1)
        replay_buffer.close()
        print('[Main] Done.')


if __name__ == '__main__':
    main()
