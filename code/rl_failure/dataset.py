import numpy as np
from torch.utils.data import Dataset
import os

class MahjongDataset(Dataset):
    def __init__(self, begin, matches, data_dir='data'):
        """
        初始化数据集。
        将所有 npz 文件的数据加载并拼接为连续的 NumPy 数组，避免多进程内存泄漏。
        """
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
                # 假设数据格式为字典结构的 npz，包含 'obs', 'mask', 'act'
                d = np.load(file_path)
                temp_obs.append(d['obs'])
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