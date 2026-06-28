"""内存预加载数据集 — 延迟增强."""

import numpy as np
import json
import os
import time
import torch
from torch.utils.data import Dataset, Sampler

from augment import get_transforms_by_level, apply_transform


class MemPreloadDataset(Dataset):
    """
    虚拟长度 = n_aug × 原始样本数。
    __init__ 一次性加载所有 npz → 3 个大数组 (int8)。
    __getitem__ 纯内存索引 + int8 解码 + 变换。
    """

    def __init__(self, data_dir='data', begin=0.0, end=1.0, n_aug=288):
        t0 = time.time()

        with open(os.path.join(data_dir, 'count.json')) as f:
            self.match_samples = json.load(f)

        total_matches = len(self.match_samples)
        self.begin = int(begin * total_matches)
        self.end = int(end * total_matches)
        self.match_samples = self.match_samples[self.begin:self.end]

        n_matches = len(self.match_samples)
        self.n_raw = sum(self.match_samples)
        self.n_aug = n_aug

        self.data_dir = data_dir

        # obs:  (N, 160, 4, 9)  int8
        # mask: (N, 235)         int8
        # act:  (N,)             int64
        print(f'[MemPreload] Allocating {self.n_raw:,} × (160×4×9 + 235 + 8) bytes...')
        self.all_obs = np.empty((self.n_raw, 160, 4, 9), dtype=np.int8)
        self.all_mask = np.empty((self.n_raw, 235), dtype=np.int8)
        self.all_act = np.empty((self.n_raw,), dtype=np.int64)

        offset = 0
        for i, n_samples in enumerate(self.match_samples):
            match_id = self.begin + i
            d = np.load(os.path.join(data_dir, f'{match_id}.npz'))
            n = d['act'].shape[0]
            assert n == n_samples, f'match {match_id}: expected {n_samples}, got {n}'

            self.all_obs[offset:offset + n] = d['obs']
            self.all_mask[offset:offset + n] = d['mask']
            self.all_act[offset:offset + n] = d['act']
            d.close()
            offset += n

            # 进度报告
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                mb_loaded = offset * (160 * 4 * 9 + 235) / (1024 * 1024)
                print(f'[MemPreload] Loaded {i+1}/{n_matches} matches, '
                      f'{offset:,} samples, {mb_loaded:.0f} MB, {elapsed:.1f}s')

        elapsed = time.time() - t0
        total_mb = offset * (160 * 4 * 9 + 235 + 8) / (1024 * 1024)
        print(f'[MemPreload] Done: {n_matches} matches, {self.n_raw:,} raw samples '
              f'× {self.n_aug} → {len(self):,} virtual samples')
        print(f'[MemPreload] Memory: {total_mb:.0f} MB loaded in {elapsed:.1f}s '
              f'({self.n_raw/elapsed:.0f} samples/s)')

        # 加载指定 level 的变换表
        self.transforms = get_transforms_by_level(self.n_aug)
        print(f'[MemPreload] Using {len(self.transforms)}/{288} transforms (level={self.n_aug}×)')

    def __len__(self):
        return self.n_raw * self.n_aug

    def __getitem__(self, idx):
        raw_idx = idx // self.n_aug
        tf_idx = idx % self.n_aug

        obs = self.all_obs[raw_idx].astype(np.float32)
        obs[37:41] /= 21.0          # ch37-40  WALL
        obs[154] /= 3.0             # ch154    TILE_FROM
        obs[157:160] /= 14.0        # ch157-159 HAND_SIZE
        mask = self.all_mask[raw_idx].astype(np.float32)
        act = int(self.all_act[raw_idx])

        tf = self.transforms[tf_idx]
        aug_obs, aug_mask, aug_act = apply_transform(obs, mask, act, tf)

        return aug_obs, aug_mask, aug_act


class LazyAugSampler(Sampler):
    """
    轻量 Sampler — 只 shuffle 原始索引 (N ≈ 5.87M)，避免 randperm(N*n_aug) 爆炸。

    策略: 对 n_aug 个变换逐一 shuffle 原始索引。
    - 每个 batch 包含 batch_size 个不同原始样本 (同一变换)
    - 每个 epoch 总 yield N*n_aug 个索引，所有(样本, 变换)对全覆盖
    - 峰值内存 ~47MB (randperm 5.87M)，vs 全量 randperm 的 6-54GB

    变换间分组对 GroupNorm 无影响 (GN 不跨 batch)。
    """

    def __init__(self, n_raw, n_aug=288):
        self.n_raw = n_raw
        self.n_aug = n_aug

    def __len__(self):
        return self.n_raw * self.n_aug

    def __iter__(self):
        n = self.n_raw
        for t in range(self.n_aug):
            perm = torch.randperm(n)
            base = t
            for r in perm.numpy():
                yield int(r) * self.n_aug + base
