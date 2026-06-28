import numpy as np
from torch.utils.data import Dataset
import os

class MahjongDataset(Dataset):
    def __init__(self, begin, matches, data_dir='data'):
        """初始化数据集，将所有 npz 数据拼接为连续 NumPy 数组."""
        super().__init__()
        self.begin = begin
        self.matches = matches
        self.data_dir = data_dir

        print(f"[Dataset] Loading {matches} files starting from {begin}...")

        temp_obs = []
        temp_mask = []
        temp_act = []

        for i in range(self.matches):
            file_path = os.path.join(self.data_dir, f'{i + self.begin}.npz')
            try:
                d = np.load(file_path)
                # int8 解码: 恢复缩放通道的原始 0~1 范围
                obs = d['obs'].astype(np.float32)
                obs[:, 37:41, :, :] /= 21.0   # ch37-40 WALL
                obs[:, 154, :, :] /= 3.0      # ch154 TILE_FROM
                obs[:, 157:160, :, :] /= 14.0 # ch157-159 HAND_SIZE
                temp_obs.append(obs)
                temp_mask.append(d['mask'])
                temp_act.append(d['act'])
            except Exception as e:
                print(f"[Dataset Warning] Failed to load {file_path}: {e}")
                continue

        if not temp_obs:
            raise ValueError("No data loaded. Please check your data directory and indices.")

        print("[Dataset] Concatenating data into contiguous memory block...")
        # 拼接为连续数组
        self.all_obs = np.concatenate(temp_obs, axis=0)
        self.all_mask = np.concatenate(temp_mask, axis=0)
        self.all_act = np.concatenate(temp_act, axis=0)

        self.total_size = len(self.all_act)
        print(f"[Dataset] Load complete. Total samples: {self.total_size}")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # 移除复杂索引计算，改为 O(1) 直接切片
        return self.all_obs[idx], self.all_mask[idx], self.all_act[idx]