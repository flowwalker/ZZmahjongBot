"""国标麻将数据集."""
import numpy as np
from torch.utils.data import Dataset
import os

class MahjongDataset(Dataset):
    def __init__(self, begin, matches, data_dir='data'):
        super().__init__()
        to, tm, ta = [], [], []
        for i in range(matches):
            fp = os.path.join(data_dir, f'{i + begin}.npz')
            try:
                d = np.load(fp)
                to.append(d['obs']); tm.append(d['mask']); ta.append(d['act'])
            except Exception as e: print(f"[Warning] {fp}: {e}")
        if not to: raise ValueError("No data loaded!")
        self.all_obs = np.concatenate(to, axis=0)
        self.all_mask = np.concatenate(tm, axis=0)
        self.all_act = np.concatenate(ta, axis=0)
        self.total_size = len(self.all_act)
        print(f"[Dataset] Total: {self.total_size} samples")
    def __len__(self): return self.total_size
    def __getitem__(self, idx):
        return self.all_obs[idx], self.all_mask[idx], self.all_act[idx]
