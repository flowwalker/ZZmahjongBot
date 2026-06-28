"""国标麻将数据集."""
import numpy as np
from torch.utils.data import Dataset
import os


class MahjongDataset(Dataset):
    def __init__(self, begin, matches, data_dir='data', quality_filter=True):
        super().__init__()
        self.begin = begin
        self.matches = matches
        self.data_dir = data_dir
        print(f"[Dataset] Loading {matches} files from {begin}...")
        temp_obs, temp_mask, temp_act = [], [], []
        for i in range(matches):
            fp = os.path.join(data_dir, f'{i + begin}.npz')
            try:
                d = np.load(fp)
                obs, mask, act = d['obs'], d['mask'], d['act']
                if quality_filter:
                    valid = mask.sum(axis=-1) > 1
                    obs, mask, act = obs[valid], mask[valid], act[valid]
                if len(act) > 0:
                    temp_obs.append(obs); temp_mask.append(mask); temp_act.append(act)
            except Exception as e:
                print(f"[Warning] {fp}: {e}")
        if not temp_obs:
            raise ValueError("No data loaded!")
        self.all_obs = np.concatenate(temp_obs, axis=0)
        self.all_mask = np.concatenate(temp_mask, axis=0)
        self.all_act = np.concatenate(temp_act, axis=0)
        self.total_size = len(self.all_act)
        print(f"[Dataset] Total: {self.total_size} samples")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        return self.all_obs[idx], self.all_mask[idx], self.all_act[idx]
