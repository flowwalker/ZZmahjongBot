"""国标麻将 CNN 模型 — 6通道"""

import torch
from torch import nn


class CNNModel(nn.Module):
    """
    三层卷积 + 双头 (策略 + 价值) 模型

    输入: (batch, 6, 4, 9) — observation
    输出: (batch, 235) masked_logits, (batch, 1) value
    """

    def __init__(self, in_channels=6):
        super().__init__()
        self.in_channels = in_channels

        # 共享卷积塔
        self._tower = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Flatten()
        )

        # 策略头
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )

        # 价值头 (仅训练时使用)
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict: dict):
        """
        Args:
            input_dict: {
                'observation': FloatTensor (batch, 6, 4, 9),
                'action_mask': FloatTensor (batch, 235)
            }

        Returns:
            masked_logits: (batch, 235) — 非法动作被 mask 为 -inf
            value: (batch, 1) — 状态价值估计
        """
        obs = input_dict["observation"].float()
        hidden = self._tower(obs)
        logits = self._logits(hidden)

        # 对非法动作施加 -inf mask
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask

        # 价值估计
        value_hidden = self._value_branch[0](hidden)
        value_hidden = self._value_branch[1](value_hidden)
        try:
            value = self._value_branch[2](value_hidden)
        except RuntimeError as e:
            # 解决某些 aarch64 平台上的 CPU matmul 问题
            if value_hidden.device.type == 'cpu' and 'primitive descriptor' in str(e):
                w = self._value_branch[2].weight
                b = self._value_branch[2].bias
                value = torch.sum(value_hidden * w, dim=1, keepdim=True)
                if b is not None:
                    value = value + b.view(1, 1)
            else:
                raise

        return masked_logits, value
