"""预处理 data.txt → npz 样本."""
from feature import FeatureAgent
import numpy as np, json, os, sys

if len(sys.argv) >= 4:
    START = int(sys.argv[1])
    END   = int(sys.argv[2])
    PID   = int(sys.argv[3])
else:
    # 单进程模式：处理全部数据
    START = 0
    END   = 10**9  # 足够大，读到文件末尾
    PID   = 0

obs = [[] for _ in range(4)]
actions = [[] for _ in range(4)]
matchid = -1
l = []
init_done = False

def filterData():
    global obs, actions
    newobs, newactions = [[] for _ in range(4)], [[] for _ in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            if o['action_mask'].sum() > 1:
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
    obs, actions = newobs, newactions

def saveData():
    global matchid
    nd = sum(len(x) for x in obs)
    l.append(nd)
    if nd > 0:
        np.savez(f'data/{matchid}.npz',
            obs=np.stack([x['observation'] for i in range(4) for x in obs[i]]).astype(np.int8),
            mask=np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8),
            act=np.array([x for i in range(4) for x in actions[i]]))
    for x in obs: x.clear()
    for x in actions: x.clear()

with open('data/data.txt', encoding='UTF-8') as f:
    for line in f:
        t = line.split()
        if not t: continue

        if t[0] == 'Match':
            if init_done:
                filterData(); saveData()
            matchid += 1
            if matchid >= END: break
            if matchid >= START:
                agents = [FeatureAgent(i) for i in range(4)]
                init_done = True
                if matchid % 128 == 0:
                    print(f'[P{PID}] {matchid}')
        elif matchid < START or not init_done:
            continue
        elif t[0] == 'Wind':
            for a in agents: a.request2obs(line)
        elif t[0] == 'Player':
            p = int(t[1])
            if t[2] == 'Deal':
                agents[p].request2obs(' '.join(t[2:]))
            elif t[2] == 'Draw':
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs(' '.join(t[2:])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs(' '.join(t[:3]))
            elif t[2] == 'Play':
                actions[p].pop()
                actions[p].append(agents[p].response2action(' '.join(t[2:])))
                for i in range(4):
                    if i == p: agents[p].request2obs(line)
                    else:
                        obs[i].append(agents[i].request2obs(line))
                        actions[i].append(0)
                curTile = t[3]
            elif t[2] == 'Chi':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Chi %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
            elif t[2] == 'Peng':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Peng %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
            elif t[2] == 'Gang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                for i in range(4):
                    agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
            elif t[2] == 'AnGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('AnGang %s' % t[3]))
                for i in range(4):
                    if i == p: agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                    else: agents[i].request2obs('Player %d AnGang' % p)
            elif t[2] == 'BuGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('BuGang %s' % t[3]))
                for i in range(4):
                    if i == p: agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                    else:
                        obs[i].append(agents[i].request2obs('Player %d BuGang %s' % (p, t[3])))
                        actions[i].append(0)
            elif t[2] == 'Hu':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Hu'))
            if t[2] in ('Peng', 'Gang', 'Hu'):
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == 'Chi':
                            actions[p].pop(); actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[k + 3])))
                        elif t[k + 2] == 'Peng':
                            actions[p].pop(); actions[p].append(agents[p].response2action('Peng %s' % t[k + 3]))
                        elif t[k + 2] == 'Gang':
                            actions[p].pop(); actions[p].append(agents[p].response2action('Gang %s' % t[k + 3]))
                        elif t[k + 2] == 'Hu':
                            actions[p].pop(); actions[p].append(agents[p].response2action('Hu'))
                    else: break
        elif t[0] == 'Score' and init_done:
            filterData(); saveData()

# 最后一个 match
if init_done and matchid < END:
    filterData(); saveData()

count_file = f'data/count_{PID}.json' if len(sys.argv) >= 4 else 'data/count.json'
with open(count_file, 'w') as f:
    json.dump(l, f)
print(f'[P{PID}] Done: {len(l)} matches, {sum(l)} samples')
