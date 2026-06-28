"""预处理 data.txt → npz 样本."""

import sys
import os
import json
import time
import numpy as np
from multiprocessing import Pool

from feature import FeatureAgent
from augment import augment_batch

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data.txt')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


#  第一个阶段：扫描 data.txt，记录每场 Match 的字节偏移
#  使用二进制模式，保证 tell() 偏移量精确可靠

def scan_match_offsets(data_file):
    """返回 [(match_idx, byte_offset), ...] 列表，最后一个元素是 EOF 偏移"""
    offsets = []
    with open(data_file, 'rb') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            # 快速判断: b'Match ' 开头
            if line.startswith(b'Match '):
                offsets.append((len(offsets), pos))
        offsets.append((len(offsets), f.tell()))
    return offsets


#  第二个阶段：每个 worker 处理自己负责的 match 段

def process_matches(args):
    """
    单个 worker 入口。
    二进制模式读文件——seek 到精确字节偏移，每行手动 decode。

    Args:
        args: (worker_id, match_range, data_file, output_dir)
        match_range: (start_match_idx, end_match_idx, start_offset, end_offset)
    """
    worker_id, match_range, data_file, output_dir = args
    start_match_idx, end_match_idx, start_offset, end_offset = match_range

    obs = [[] for _ in range(4)]
    actions = [[] for _ in range(4)]
    matchid = start_match_idx
    actual_match_count = 0

    def filterData():
        nonlocal obs, actions
        newobs = [[] for _ in range(4)]
        newactions = [[] for _ in range(4)]
        for i in range(4):
            for j, o in enumerate(obs[i]):
                if o['action_mask'].sum() > 1:
                    newobs[i].append(o)
                    newactions[i].append(actions[i][j])
        obs = newobs
        actions = newactions

    def saveData(mid):
        nonlocal obs, actions
        assert [len(x) for x in obs] == [len(x) for x in actions], \
            f'[W{worker_id}] obs/actions mismatch at match {mid}'

        obs_stack = np.stack([x['observation'] for i in range(4) for x in obs[i]])
        mask_stack = np.stack([x['action_mask'] for i in range(4) for x in obs[i]])
        act_stack = np.array([x for i in range(4) for x in actions[i]], dtype=np.int64)

        # 288-fold 物理增强
        aug_obs, aug_mask, aug_act = augment_batch(obs_stack, mask_stack, act_stack)

        # int8 缩放
        aug_obs[:, 37:41, :, :] *= 21
        aug_obs[:, 154, :, :] *= 3
        aug_obs[:, 157:160, :, :] *= 14
        aug_obs_i8 = np.round(aug_obs).astype(np.int8)
        aug_mask_i8 = aug_mask.astype(np.int8)

        outpath = os.path.join(output_dir, f'{mid}.npz')
        np.savez_compressed(outpath, obs=aug_obs_i8, mask=aug_mask_i8, act=aug_act)

        for x in obs: x.clear()
        for x in actions: x.clear()

    with open(data_file, 'rb') as f:
        f.seek(start_offset)

        while True:
            # 边界检查：不能越界到下一个 worker 的地盘
            if end_offset is not None and f.tell() >= end_offset:
                break

            raw = f.readline()
            if not raw:
                break
            line = raw.decode('utf-8')

            t = line.split()
            if len(t) == 0:
                continue

            if t[0] == 'Match':
                agents = [FeatureAgent(i) for i in range(4)]
                if actual_match_count > 0 and actual_match_count % 128 == 0:
                    print(f'[W{worker_id}] Processing match {matchid}...', flush=True)

            elif t[0] == 'Wind':
                for agent in agents:
                    agent.request2obs(line)

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
                        if i == p:
                            agents[p].request2obs(line)
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
                        if i == p:
                            agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                        else:
                            agents[i].request2obs('Player %d AnGang' % p)
                elif t[2] == 'BuGang':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('BuGang %s' % t[3]))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                        else:
                            obs[i].append(agents[i].request2obs('Player %d BuGang %s' % (p, t[3])))
                            actions[i].append(0)
                elif t[2] == 'Hu':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('Hu'))

                # Ignore clause
                if t[2] in ['Peng', 'Gang', 'Hu']:
                    for k in range(5, 15, 5):
                        if len(t) > k:
                            p = int(t[k + 1])
                            if t[k + 2] == 'Chi':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action(
                                    'Chi %s %s' % (curTile, t[k + 3])))
                            elif t[k + 2] == 'Peng':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action(
                                    'Peng %s' % t[k + 3]))
                            elif t[k + 2] == 'Gang':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action(
                                    'Gang %s' % t[k + 3]))
                            elif t[k + 2] == 'Hu':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Hu'))
                        else:
                            break

            elif t[0] == 'Score':
                filterData()
                saveData(matchid)
                matchid += 1
                actual_match_count += 1

                if matchid > end_match_idx:
                    break

    print(f'[W{worker_id}] Done. Processed matches {start_match_idx}-{end_match_idx}'
          f' ({actual_match_count} total)', flush=True)

    return worker_id, actual_match_count


#  主流程

def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'[Main] Scanning match offsets from {DATA_FILE} ...')
    t0 = time.time()
    offsets = scan_match_offsets(DATA_FILE)
    total_matches = len(offsets) - 1
    print(f'[Main] Found {total_matches} matches in {time.time()-t0:.1f}s')

    chunk_size = (total_matches + n_workers - 1) // n_workers
    tasks = []
    for w in range(n_workers):
        start_idx = w * chunk_size
        end_idx = min((w + 1) * chunk_size, total_matches) - 1

        if start_idx >= total_matches:
            break

        start_offset = offsets[start_idx][1]
        end_offset = offsets[end_idx + 1][1] if end_idx + 1 < len(offsets) else None

        n_matches = end_idx - start_idx + 1
        tasks.append((w, (start_idx, end_idx, start_offset, end_offset),
                      DATA_FILE, OUTPUT_DIR))
        print(f'[Main] Worker {w}: matches {start_idx}-{end_idx} ({n_matches} matches)')

    print(f'\n[Main] Starting {len(tasks)} workers ...')
    t1 = time.time()

    with Pool(processes=len(tasks)) as pool:
        results = pool.map(process_matches, tasks)

    elapsed = time.time() - t1
    total_processed = sum(r[1] for r in results)
    print(f'\n[Main] All workers done in {elapsed/60:.1f} min')
    print(f'[Main] Total matches processed: {total_processed}')

    sample_counts = []
    for mid in range(total_matches):
        fpath = os.path.join(OUTPUT_DIR, f'{mid}.npz')
        if os.path.exists(fpath):
            d = np.load(fpath)
            sample_counts.append(d['act'].shape[0])
            d.close()

    total_samples = sum(sample_counts)
    total_size_bytes = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f'{mid}.npz'))
        for mid in range(total_matches)
        if os.path.exists(os.path.join(OUTPUT_DIR, f'{mid}.npz'))
    )

    print(f'[Main] Total samples: {total_samples}')
    print(f'[Main] Total disk:     {total_size_bytes/1024/1024/1024:.2f} GB')

    with open(os.path.join(OUTPUT_DIR, 'count.json'), 'w') as f:
        json.dump(sample_counts, f)


if __name__ == '__main__':
    main()
