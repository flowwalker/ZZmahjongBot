"""国标麻将 Pure CNN 模型 — 155通道"""

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Reference 风格残差块: Conv-BN-ReLU-Conv-BN, +skip, ReLU"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class CNNModel(nn.Module):
    """
    Reference 架构纯 CNN，两阶段通道设计。

    架构:
      Stem:  155 → 256 → 256
      Stage1: 8× ResBlock(256)
      Trans:  Conv(256→128) + BN + ReLU
      Stage2: 8× ResBlock(128)
      Head:   Flatten(128×36=4608) → Linear(4608, 235)

    in_channels: 155 (从 FeatureAgent.OBS_SIZE 自动获取)
    """

    def __init__(self, in_channels=155):
        super().__init__()
        self.in_channels = in_channels

        self._stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1, bias=False),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self._stage1 = nn.ModuleList([ResBlock(256) for _ in range(8)])

        self._trans = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self._stage2 = nn.ModuleList([ResBlock(128) for _ in range(8)])

        self._policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 9, 235),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict: dict):
        obs = input_dict["observation"].float()

        x = self._stem(obs)
        for block in self._stage1:
            x = block(x)
        x = self._trans(x)
        for block in self._stage2:
            x = block(x)

        logits = self._policy(x)

        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)

        value = torch.zeros(obs.size(0), 1, device=obs.device)
        return logits + inf_mask, value
