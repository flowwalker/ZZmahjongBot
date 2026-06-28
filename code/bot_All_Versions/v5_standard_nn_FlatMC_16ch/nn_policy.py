"""NN 推理封装."""

import os
import numpy as np
import torch

from model import CNNModel
from state_encoder import OBS_SIZE


class NNEvaluator:
    """神经网络推理封装，支持单次和批量推理"""

    def __init__(self, model_path: str = None, device: str = 'cpu'):
        """
        加载模型权重，初始化推理环境

        Args:
            model_path: 模型权重文件路径，None 时自动搜索
            device: 推理设备 ('cpu' 或 'mps')
        """
        self.device = torch.device(device)
        self.model = CNNModel(in_channels=OBS_SIZE)
        self.model.to(self.device)
        self.model.eval()

        # 搜索模型权重文件
        if model_path is None:
            search_paths = [
                os.path.join(os.path.dirname(__file__), 'model_cnn_Res_Mish_SE_standard.pt'),
                'model_cnn_Res_Mish_SE_standard.pt',
                'data/model_cnn_Res_Mish_SE_standard.pt',
                '/data/mahjong/model_cnn_Res_Mish_SE_standard.pt',
            ]
            for p in search_paths:
                if os.path.exists(p):
                    model_path = p
                    break

        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                ckpt = ckpt['model']
            self.model.load_state_dict(ckpt, strict=False)
        else:
            pass

        # 预分配全1 mask用于value-only评估
        self._ones_mask = torch.ones(1, 235, device=self.device)

    def evaluate(self, obs: np.ndarray, mask: np.ndarray) -> tuple:
        """
        单次推理

        Args:
            obs: 观测张量 (16, 4, 9)
            mask: 动作掩码 (235,)

        Returns:
            (policy_probs: np.ndarray(235,), value: float)
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)
            logits, value = self.model({
                'observation': obs_t,
                'action_mask': mask_t,
            })
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            v = value.squeeze().item()
        return probs, v

    def evaluate_batch(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> tuple:
        """
        批量推理

        Args:
            obs_batch: 观测张量 (N, 16, 4, 9)
            mask_batch: 动作掩码 (N, 235)

        Returns:
            (policy_probs: np.ndarray(N, 235), values: np.ndarray(N,))
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_batch).float().to(self.device)
            mask_t = torch.from_numpy(mask_batch).float().to(self.device)
            logits, values = self.model({
                'observation': obs_t,
                'action_mask': mask_t,
            })
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            vs = values.squeeze(-1).cpu().numpy()
        return probs, vs

    def evaluate_value_only(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        仅评估价值（action_mask 全1），用于 FlatMC 叶节点批量评估

        Args:
            obs_batch: 观测张量 (N, 16, 4, 9)

        Returns:
            values: np.ndarray(N,)
        """
        n = obs_batch.shape[0]
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_batch).float().to(self.device)
            mask_t = self._ones_mask.expand(n, -1)
            _, values = self.model({
                'observation': obs_t,
                'action_mask': mask_t,
            })
            vs = values.squeeze(-1).cpu().numpy()
        return vs

    def get_discard_probs(self, obs: np.ndarray, mask: np.ndarray,
                          hand_tiles: list) -> list:
        """
        提取手牌对应的 Play 动作概率

        Args:
            obs: 观测张量 (16, 4, 9)
            mask: 动作掩码 (235,)
            hand_tiles: 手牌列表 (如 ['W1', 'T3', ...])

        Returns:
            [(tile, prob), ...] 按 prob 降序排列
        """
        from state_encoder import OFFSET_ACT, OFFSET_TILE
        probs, _ = self.evaluate(obs, mask)
        results = []
        for tile in set(hand_tiles):
            idx = OFFSET_ACT['Play'] + OFFSET_TILE[tile]
            results.append((tile, probs[idx]))
        results.sort(key=lambda x: -x[1])
        return results
