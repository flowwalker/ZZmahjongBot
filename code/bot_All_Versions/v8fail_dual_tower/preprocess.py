"""预处理 data.txt → npz 样本."""
from feature import FeatureAgent
import numpy as np
import json
import os

os.makedirs('data', exist_ok=True)

obs = [[] for _ in range(4)]
actions = [[] for _ in range(4)]
matchid = -1
l = []
action_freq = {}


def filterData():
    global obs, actions
    newobs, newactions = [[] for _ in range(4)], [[] for _ in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            ms = o['action_mask'].sum()
            if ms >= 3:  # 至少3个合法动作才保留
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
            elif ms >= 2 and np.random.random() < 0.5:
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
    obs, actions = newobs, newactions


def saveData():
    global action_freq
    assert [len(x) for x in obs] == [len(x) for x in actions]
    for i in range(4):
        for a in actions[i]:
            action_freq[int(a)] = action_freq.get(int(a), 0) + 1
    ns = sum(len(x) for x in obs)
    l.append(ns)
    if ns > 0:
        np.savez('data/%d.npz' % matchid,
            obs=np.stack([x['observation'] for i in range(4) for x in obs[i]]).astype(np.int8),
            mask=np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8),
            act=np.array([x for i in range(4) for x in actions[i]]))
    for x in obs: x.clear()
    for x in actions: x.clear()


with open('data/data.txt', encoding='UTF-8') as f:
    line = f.readline()
    curTile = ''
    while line:
        t = line.split()
        if len(t) == 0:
            line = f.readline()
            continue
        if t[0] == 'Match':
            agents = [FeatureAgent(i) for i in range(4)]
            matchid += 1
            if matchid % 128 == 0:
                print('Processing match %d %s...' % (matchid, t[1]))
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
                    else: agents[i].request2obs(' '.join(t[:3]))
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
                    else: agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
            elif t[2] == 'Peng':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Peng %s' % (p, t[3])))
                        actions[p].append(0)
                    else: agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
            elif t[2] in ('Gang', 'AnGang', 'BuGang'):
                actions[p].pop()
                actions[p].append(agents[p].response2action(' '.join(t[2:4])))
                for i in range(4): agents[i].request2obs(' '.join(['Player', str(p), t[2], t[3]]))
            elif t[2] == 'Hu':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Hu'))
        elif t[0] == 'Score':
            filterData()
            saveData()
        line = f.readline()

with open('data/count.json', 'w') as f:
    json.dump(l, f)
with open('data/act_freq.json', 'w') as f:
    json.dump(action_freq, f)
print(f'Done: {matchid+1} matches, {sum(l)} samples, {len(action_freq)} action types')
