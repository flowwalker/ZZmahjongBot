"""预处理 data.txt → npz 样本."""

from feature import FeatureAgent
import numpy as np
import json
import os
import multiprocessing
import argparse


def get_virtual_offsets(data_file, out_dir):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f'{data_file} not found')

    os.makedirs(out_dir, exist_ok=True)
    offsets_path = f'{out_dir}/match_offsets.json'

    if os.path.exists(offsets_path):
        with open(offsets_path, 'r', encoding='UTF-8') as f:
            return json.load(f)

    print(f"Scanning {data_file} for Match boundaries (Fast Binary Mode)...")
    offsets = []
    
    with open(data_file, 'rb') as f:
        offset = f.tell()
        line = f.readline()
        while line:
            if line.startswith(b'Match'):
                offsets.append(offset)
            offset = f.tell()
            line = f.readline()

    with open(offsets_path, 'w', encoding='UTF-8') as f:
        json.dump(offsets, f)
        
    return offsets


def process_part_virtual(args):
    """鲁棒的虚拟分块处理逻辑"""
    part_idx, start_match_id, end_match_id, start_offset, data_file, out_dir = args
    pid = part_idx

    # 内存 O(1) 检索，避免 os.path.exists 造成的系统调用风暴
    existing_files = set(os.listdir(out_dir))

    obs = [[] for _ in range(4)]
    actions = [[] for _ in range(4)]
    
    current_match_id = start_match_id - 1
    counts = []
    agents = None
    curTile = ''

    def filter_and_save(player_scores=None):
        nonlocal obs, actions
        newobs, newactions = [[] for _ in range(4)], [[] for _ in range(4)]
        for i in range(4):
            for j, o in enumerate(obs[i]):
                if o['action_mask'].sum() > 1:
                    newobs[i].append(o)
                    newactions[i].append(actions[i][j])

        nd = sum(len(x) for x in newobs)
        counts.append(nd)
        if nd > 0:
            fpath = f'{out_dir}/{current_match_id}.npz'
            # 给每个样本标上所属玩家的终局得分
            if player_scores is not None:
                scores = sum(([player_scores[i]] * len(newobs[i]) for i in range(4)), [])
            else:
                scores = [0.0] * nd
            np.savez(fpath,
                obs=np.stack([x['observation'] for i in range(4) for x in newobs[i]]).astype(np.int8),
                mask=np.stack([x['action_mask'] for i in range(4) for x in newobs[i]]).astype(np.int8),
                act=np.array([x for i in range(4) for x in newactions[i]]),
                score=np.array(scores, dtype=np.float32))

        for x in obs: x.clear()
        for x in actions: x.clear()

    with open(data_file, 'rb') as f:
        f.seek(start_offset)
        
        for binary_line in f:
            line = binary_line.decode('UTF-8', errors='replace')
            t = line.split()
            if not t:
                continue

            if t[0] == 'Match':
                if current_match_id == end_match_id - 1:
                    if agents is not None:
                        filter_and_save()
                    agents = None  
                    break

                if agents is not None:
                    filter_and_save()

                current_match_id += 1
                target_filename = f"{current_match_id}.npz"

                if target_filename in existing_files:
                    try:
                        with np.load(f'{out_dir}/{target_filename}') as data:
                            counts.append(len(data['act']))
                        agents = None
                        continue
                    except Exception:
                        # 发现坏包，移出记录，由本进程重新解析并覆盖写入
                        existing_files.remove(target_filename)

                # 重置状态，防止越界污染
                for x in obs: x.clear()
                for x in actions: x.clear()
                agents = [FeatureAgent(i) for i in range(4)]
                continue

            if agents is None:
                continue

            if t[0] == 'Wind':
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
                    for i in range(4): agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
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
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[k + 3])))
                            elif t[k + 2] == 'Peng':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Peng %s' % t[k + 3]))
                            elif t[k + 2] == 'Gang':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Gang %s' % t[k + 3]))
                            elif t[k + 2] == 'Hu':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Hu'))
                        else:
                            break

            elif t[0] == 'Score':
                # Score <s0> <s1> <s2> <s3> 按位置对应玩家 0-3
                player_scores = [0.0] * 4
                for i in range(min(4, len(t) - 1)):
                    player_scores[i] = float(t[i + 1])
                filter_and_save(player_scores)
                agents = None

    if agents is not None:
        filter_and_save()

    with open(f'{out_dir}/count_{pid}.json', 'w', encoding='UTF-8') as f:
        json.dump(counts, f)

    return pid, len(counts), sum(counts)


def merge_counts(out_dir, num_parts):
    all_counts = []
    for i in range(num_parts):
        cf = f'{out_dir}/count_{i}.json'
        if os.path.exists(cf):
            with open(cf, 'r', encoding='UTF-8') as f:
                all_counts.extend(json.load(f))
    if all_counts:
        with open(f'{out_dir}/count.json', 'w', encoding='UTF-8') as f:
            json.dump(all_counts, f)
    return all_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='data/data.txt')
    parser.add_argument('--out_dir', default='data')
    parser.add_argument('--num_parts', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=max(1, multiprocessing.cpu_count() - 2))
    args = parser.parse_args()

    offsets = get_virtual_offsets(args.data_file, args.out_dir)
    total_matches = len(offsets)
    if total_matches == 0:
        return

    per_part = (total_matches + args.num_parts - 1) // args.num_parts
    actual_parts = min(args.num_parts, total_matches)

    tasks = []
    for i in range(actual_parts):
        start_match_id = i * per_part
        end_match_id = min((i + 1) * per_part, total_matches)
        start_offset = offsets[start_match_id]
        tasks.append((i, start_match_id, end_match_id, start_offset, args.data_file, args.out_dir))

    print(f'[Preprocess] {len(tasks)} virtual parts, {args.num_workers} workers.')

    with multiprocessing.Pool(args.num_workers) as pool:
        results = pool.map(process_part_virtual, tasks)

    total_m = sum(r[1] for r in results)
    total_s = sum(r[2] for r in results)
    print(f'[Preprocess] Done: {total_m} matches, {total_s} samples processed/aligned.')

    merge_counts(args.out_dir, args.num_parts)

if __name__ == '__main__':
    main()