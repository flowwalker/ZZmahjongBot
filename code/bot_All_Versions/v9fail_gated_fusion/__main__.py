"""国标麻将 Botzone Bot."""
import os, sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
for k, v in [('OMP_NUM_THREADS','1'),('MKL_NUM_THREADS','1'),('OPENBLAS_NUM_THREADS','1'),
             ('VECLIB_MAXIMUM_THREADS','1'),('NUMEXPR_NUM_THREADS','1')]:
    os.environ[k] = v
import numpy as np, torch
torch.set_num_threads(1)
from feature import FeatureAgent
from model import CNNModel

if __name__ == '__main__':
    model = CNNModel(in_channels=FeatureAgent.OBS_SIZE)
    for p in [os.path.join(os.path.dirname(__file__), 'model_vultra.pt'),
              'model_vultra.pt', 'data/model_vultra.pt', '/data/mahjong/model_vultra.pt']:
        if os.path.exists(p):
            try:
                ckpt = torch.load(p, map_location='cpu')
                if isinstance(ckpt, dict) and 'model' in ckpt: ckpt = ckpt['model']
                model.load_state_dict(ckpt, strict=False)
                break
            except RuntimeError: pass
    model.train(False)
    angang, zimo = None, False
    try: input()
    except EOFError: sys.exit(0)
    while True:
        try:
            request = input()
            while not request.strip(): request = input()
        except EOFError: break
        t = request.split()
        if t[0] == '0':
            seatWind = int(t[1]); agent = FeatureAgent(seatWind)
            agent.request2obs('Wind %s' % t[2])
            print('PASS')
        elif t[0] == '1':
            agent.request2obs(' '.join(['Deal', *t[5:]]))
            print('PASS')
        elif t[0] == '2':
            obs = agent.request2obs('Draw %s' % t[1])
            state = {
                'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
                'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0)),
            }
            with torch.no_grad(): logits, _ = model(state)
            action = logits.numpy().flatten().argmax()
            response = agent.action2response(action)
            r = response.split()
            if r[0] == 'Hu': print('HU')
            elif r[0] == 'Play': print('PLAY %s' % r[1])
            elif r[0] == 'Gang': print('GANG %s' % r[1]); angang = r[1]
            elif r[0] == 'BuGang': print('BUGANG %s' % r[1])
        elif t[0] == '3':
            p = int(t[1])
            if t[2] == 'DRAW': agent.request2obs('Player %d Draw' % p); zimo = True; print('PASS')
            elif t[2] == 'GANG':
                if p == seatWind and angang: agent.request2obs('Player %d AnGang %s' % (p, angang))
                elif zimo: agent.request2obs('Player %d AnGang' % p)
                else: agent.request2obs('Player %d Gang' % p)
                print('PASS')
            elif t[2] == 'BUGANG':
                obs = agent.request2obs('Player %d BuGang %s' % (p, t[3]))
                if p == seatWind: print('PASS')
                else:
                    state = {
                        'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
                        'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0)),
                    }
                    with torch.no_grad(): logits, _ = model(state)
                    print('HU' if agent.action2response(logits.numpy().flatten().argmax()) == 'Hu' else 'PASS')
            else:
                zimo = False
                if t[2] == 'CHI': agent.request2obs('Player %d Chi %s' % (p, t[3]))
                elif t[2] == 'PENG': agent.request2obs('Player %d Peng' % p)
                obs = agent.request2obs('Player %d Play %s' % (p, t[-1]))
                if p == seatWind: print('PASS')
                else:
                    state = {
                        'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
                        'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0)),
                    }
                    with torch.no_grad(): logits, _ = model(state)
                    response = agent.action2response(logits.numpy().flatten().argmax())
                    r = response.split()
                    if r[0] == 'Hu': print('HU')
                    elif r[0] == 'Pass': print('PASS')
                    elif r[0] == 'Gang': print('GANG'); angang = None
                    elif r[0] in ('Peng', 'Chi'):
                        obs2 = agent.request2obs('Player %d %s' % (seatWind, response))
                        state2 = {
                            'observation': torch.from_numpy(np.expand_dims(obs2['observation'], 0)),
                            'action_mask': torch.from_numpy(np.expand_dims(obs2['action_mask'], 0)),
                        }
                        with torch.no_grad(): logits2, _ = model(state2)
                        response2 = agent.action2response(logits2.numpy().flatten().argmax())
                        print(' '.join([r[0].upper(), *r[1:], response2.split()[-1]]))
                        agent.request2obs('Player %d Un%s' % (seatWind, response))
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()
