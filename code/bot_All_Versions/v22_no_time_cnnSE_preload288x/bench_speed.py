"""
本地速度基准测试: MEM vs LAZY DataLoader 吞吐量
"""
import sys, time, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from augment import get_transforms, apply_transform

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

N_BATCHES = 30


class MemDS(Dataset):
    def __init__(self, all_obs, all_mask, all_act, tfs):
        self.all_obs = all_obs
        self.all_mask = all_mask
        self.all_act = all_act
        self.tfs = tfs
        self.n_raw = len(all_act)
    def __len__(self):
        return self.n_raw * 288
    def __getitem__(self, idx):
        raw_idx = idx // 288
        tf_idx = idx % 288
        obs = self.all_obs[raw_idx].astype(np.float32)
        obs[37:41] /= 21.0
        obs[154] /= 3.0
        obs[157:160] /= 14.0
        mask = self.all_mask[raw_idx].astype(np.float32)
        act = int(self.all_act[raw_idx])
        tf = self.tfs[tf_idx]
        aug_obs, aug_mask, aug_act = apply_transform(obs, mask, act, tf)
        return aug_obs, aug_mask, aug_act


class LazyDS(Dataset):
    def __init__(self, counts, tfs):
        offsets = [0]
        for m in counts:
            offsets.append(offsets[-1] + m)
        self.offsets = offsets
        self.n_raw = offsets[-1]
        self.tfs = tfs
        self._cache_id = -1
        self._cache = None
    def __len__(self):
        return self.n_raw * 288
    def __getitem__(self, idx):
        raw_idx = idx // 288
        tf_idx = idx % 288
        lo, hi = 0, len(self.offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.offsets[mid + 1] <= raw_idx:
                lo = mid + 1
            else:
                hi = mid
        mid = lo
        sid = raw_idx - self.offsets[mid]
        if self._cache_id != mid:
            self._cache = np.load(os.path.join(DATA_DIR, f'{mid}.npz'))
            self._cache_id = mid
        d = self._cache
        obs = d['obs'][sid].astype(np.float32)
        obs[37:41] /= 21.0
        obs[154] /= 3.0
        obs[157:160] /= 14.0
        mask = d['mask'][sid].astype(np.float32)
        act = int(d['act'][sid])
        tf = self.tfs[tf_idx]
        aug_obs, aug_mask, aug_act = apply_transform(obs, mask, act, tf)
        return aug_obs, aug_mask, aug_act


def bench(desc, ds, n_workers, shuffle, sampler=None):
    kwargs = dict(dataset=ds, batch_size=4096, num_workers=n_workers,
                  pin_memory=False, multiprocessing_context='fork' if n_workers > 0 else None)
    if sampler:
        kwargs['sampler'] = sampler
    else:
        kwargs['shuffle'] = shuffle

    loader = DataLoader(**kwargs)

    # Warmup
    for _, batch in zip(range(3), loader):
        pass

    t0 = time.time()
    total = 0
    for i, (obs, mask, act) in enumerate(loader):
        total += obs.shape[0]
        if i + 1 >= N_BATCHES:
            break
    elapsed = time.time() - t0
    print(f'{desc:18s}: {total:>8d} samp in {elapsed:>6.2f}s = '
          f'{total/elapsed:>10.0f} samp/s  ({elapsed/(i+1)*1000:>6.1f} ms/batch)')


def main():
    with open(os.path.join(DATA_DIR, 'count.json')) as f:
        all_counts = json.load(f)

    n_matches = 10000
    subset_counts = all_counts[:n_matches]
    n_raw = sum(subset_counts)

    print(f'=== 子集: {n_matches} matches, {n_raw:,} raw samples ===')

    t0 = time.time()
    all_obs = np.empty((n_raw, 160, 4, 9), dtype=np.int8)
    all_mask = np.empty((n_raw, 235), dtype=np.int8)
    all_act = np.empty((n_raw,), dtype=np.int64)

    offset = 0
    for i, n_samp in enumerate(subset_counts):
        d = np.load(os.path.join(DATA_DIR, f'{i}.npz'))
        n = d['act'].shape[0]
        all_obs[offset:offset+n] = d['obs']
        all_mask[offset:offset+n] = d['mask']
        all_act[offset:offset+n] = d['act']
        d.close()
        offset += n

    init_time = time.time() - t0
    mem_mb = offset * (160*4*9 + 235 + 8) / (1024*1024)
    print(f'加载耗时: {init_time:.2f}s  ({mem_mb:.0f} MB, {n_raw/init_time:.0f} samp/s)')

    total_raw = sum(all_counts)
    print(f'推算全量 ({total_raw:,} samples): ~{init_time * total_raw / n_raw:.0f}s '
          f'({init_time * total_raw / n_raw / 60:.1f} min)')

    tfs = get_transforms()
    ds_mem = MemDS(all_obs, all_mask, all_act, tfs)
    ds_lazy = LazyDS(subset_counts, tfs)

    print(f'\n{"="*70}')
    print(f'DataLoader 吞吐对比: batch_size=4096, {N_BATCHES} batches')
    print(f'{"="*70}')

    bench('MEM  seq 0w',     ds_mem,  0, False)
    bench('MEM  seq 4w',     ds_mem,  4, False)
    bench('LAZY seq 0w',     ds_lazy, 0, False)
    bench('LAZY seq 4w',     ds_lazy, 4, False)

    # Sampler 模式
    from dataset_mem import LazyAugSampler
    sampler = LazyAugSampler(ds_mem.n_raw)
    bench('MEM  sampler 0w', ds_mem,  0, False, sampler=sampler)
    bench('MEM  sampler 4w', ds_mem,  4, False, sampler=sampler)

    print(f'\n--- 随机访问 (最坏情况, 每个 sample 跳不同 match) ---')
    rng = np.random.default_rng(42)
    rand_idx = rng.integers(0, ds_mem.n_raw * 288, size=N_BATCHES * 4096)

    class RandSampler:
        def __init__(self, indices):
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __iter__(self):
            return iter(self.indices)

    rand_samp = RandSampler(rand_idx.tolist())
    bench('MEM  random 0w',  ds_mem,  0, False, sampler=rand_samp)
    bench('LAZY random 0w',  ds_lazy, 0, False, sampler=rand_samp)

    print(f'\nDone.')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork', force=True)
    main()
