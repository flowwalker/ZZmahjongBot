"""国标麻将 Botzone Bot — 全局平均投票 (160ch, 10.39M CNNModel ×3)"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
import torch
torch.set_num_threads(1)

from feature import FeatureAgent
from model import CNNModel
from augment import get_transforms, apply_transform

N_TTA = 100                         # 每次推理从288变换中随机采样数
MODEL_FILES = ['model_vote_1.pt', 'model_vote_2.pt', 'model_vote_3.pt']


def _load_one_model(filename):
    """严格按 baseline-vfinal 格式加载单个模型，返回 model 或 None"""
    model = CNNModel(in_channels=FeatureAgent.OBS_SIZE)
    for p in [os.path.join(os.path.dirname(__file__), filename),
              filename,
              'data/' + filename,
              '/data/mahjong/' + filename]:
        if os.path.exists(p):
            ckpt = torch.load(p, map_location='cpu')
            if isinstance(ckpt, dict) and 'model' in ckpt:
                ckpt = ckpt['model']
            try:
                model.load_state_dict(ckpt, strict=False)
            except RuntimeError:
                pass
            model.train(False)
            return model
    return None


def ensemble_infer(models, all_transforms, n_tta, obs_np, mask_np):
    """全局平均: 随机采 n_tta 个变换 → 所有模型 × 变换的 logits 取均值 → masked argmax"""
    orig_mask_t = torch.from_numpy(mask_np).float().unsqueeze(0)

    # 随机采样 n_tta 个变换
    indices = np.random.choice(len(all_transforms), size=n_tta, replace=False)
    sampled = [all_transforms[i] for i in indices]

    all_logits = []
    for model in models:
        for tf in sampled:
            aug_obs, aug_mask, _ = apply_transform(obs_np, mask_np, 0, tf)
            state = {
                'observation': torch.from_numpy(np.expand_dims(aug_obs, 0)),
                'action_mask': torch.from_numpy(np.expand_dims(aug_mask, 0)),
            }
            with torch.no_grad():
                logits_aug, _ = model(state)
            # 逆映射回原始动作空间
            action_fwd = torch.from_numpy(tf['action_fwd'])
            logits_orig = logits_aug[:, action_fwd]
            all_logits.append(logits_orig)

    # 全局平均
    avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)  # (1, 235)
    # 应用原始 mask
    avg_logits = torch.where(
        orig_mask_t > 0.5, avg_logits,
        torch.tensor(-1e8, dtype=avg_logits.dtype))
    return int(avg_logits.argmax(dim=-1).item())


if __name__ == '__main__':
    models = []
    for fname in MODEL_FILES:
        m = _load_one_model(fname)
        if m is not None:
            models.append(m)

    if not models:
        sys.exit(1)

    all_transforms = get_transforms()

    angang = None
    zimo = False

    try:
        input()
    except EOFError:
        sys.exit(0)

    while True:
        try:
            request = input()
            while not request.strip():
                request = input()
        except EOFError:
            break
        t = request.split()

        if t[0] == '0':
            seatWind = int(t[1])
            agent = FeatureAgent(seatWind)
            agent.request2obs('Wind %s' % t[2])
            print('PASS')

        elif t[0] == '1':
            agent.request2obs(' '.join(['Deal', *t[5:]]))
            print('PASS')

        elif t[0] == '2':
            obs = agent.request2obs('Draw %s' % t[1])
            action = ensemble_infer(models, all_transforms, N_TTA,
                                    obs['observation'], obs['action_mask'])
            response = agent.action2response(action)
            r = response.split()
            if r[0] == 'Hu':
                print('HU')
            elif r[0] == 'Play':
                print('PLAY %s' % r[1])
            elif r[0] == 'Gang':
                print('GANG %s' % r[1])
                angang = r[1]
            elif r[0] == 'BuGang':
                print('BUGANG %s' % r[1])

        elif t[0] == '3':
            p = int(t[1])

            if t[2] == 'DRAW':
                agent.request2obs('Player %d Draw' % p)
                zimo = True
                print('PASS')

            elif t[2] == 'GANG':
                if p == seatWind and angang:
                    agent.request2obs('Player %d AnGang %s' % (p, angang))
                elif zimo:
                    agent.request2obs('Player %d AnGang' % p)
                else:
                    agent.request2obs('Player %d Gang' % p)
                print('PASS')

            elif t[2] == 'BUGANG':
                obs = agent.request2obs('Player %d BuGang %s' % (p, t[3]))
                if p == seatWind:
                    print('PASS')
                else:
                    action = ensemble_infer(models, all_transforms, N_TTA,
                                            obs['observation'], obs['action_mask'])
                    if agent.action2response(action) == 'Hu':
                        print('HU')
                    else:
                        print('PASS')

            else:
                zimo = False
                if t[2] == 'CHI':
                    agent.request2obs('Player %d Chi %s' % (p, t[3]))
                elif t[2] == 'PENG':
                    agent.request2obs('Player %d Peng' % p)
                obs = agent.request2obs('Player %d Play %s' % (p, t[-1]))

                if p == seatWind:
                    print('PASS')
                else:
                    action = ensemble_infer(models, all_transforms, N_TTA,
                                            obs['observation'], obs['action_mask'])
                    response = agent.action2response(action)
                    r = response.split()
                    if r[0] == 'Hu':
                        print('HU')
                    elif r[0] == 'Pass':
                        print('PASS')
                    elif r[0] == 'Gang':
                        print('GANG')
                        angang = None
                    elif r[0] in ('Peng', 'Chi'):
                        obs2 = agent.request2obs('Player %d %s' % (seatWind, response))
                        action2 = ensemble_infer(models, all_transforms, N_TTA,
                                                 obs2['observation'], obs2['action_mask'])
                        response2 = agent.action2response(action2)
                        print(' '.join([r[0].upper(), *r[1:], response2.split()[-1]]))
                        agent.request2obs('Player %d Un%s' % (seatWind, response))

        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()
