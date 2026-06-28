"""国标麻将数据集."""
import numpy as np
from torch.utils.data import Dataset
import os

class MahjongDataset(Dataset):
    def __init__(self, begin, matches, data_dir='data'):
        super().__init__()
        self.begin, self.matches, self.data_dir = begin, matches, data_dir
        temp_obs, temp_mask, temp_act = [], [], []
        for i in range(matches):
            fp = os.path.join(data_dir, f'{i + begin}.npz')
            try:
                d = np.load(fp)
                temp_obs.append(d['obs']); temp_mask.append(d['mask']); temp_act.append(d['act'])
            except Exception as e:
                print(f"[Warning] {fp}: {e}")
        if not temp_obs: raise ValueError("No data loaded!")
        self.all_obs = np.concatenate(temp_obs, axis=0)
        self.all_mask = np.concatenate(temp_mask, axis=0)
        self.all_act = np.concatenate(temp_act, axis=0)
        self.total_size = len(self.all_act)
        print(f"[Dataset] Total: {self.total_size} samples")

    def __len__(self): return self.total_size
    def __getitem__(self, idx):
        return self.all_obs[idx], self.all_mask[idx], self.all_act[idx]
