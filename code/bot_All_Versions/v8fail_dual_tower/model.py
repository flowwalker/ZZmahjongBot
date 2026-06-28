"""国标麻将 DualTower 模型 — 224通道"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def _safe_linear(x, layer):
    try:
        return layer(x)
    except RuntimeError as e:
        if x.device.type == 'cpu' and 'primitive descriptor' in str(e):
            return F.linear(x, layer.weight, layer.bias)
        raise e


class SEResBlock(nn.Module):
    """SE残差块 — 策略和价值塔共用"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, channels)
        self.se_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.Mish(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        self.act = nn.Mish(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        b, c, _, _ = out.size()
        se = out.view(b, c, -1).mean(dim=2)
        se = self.se_fc(se).view(b, c, 1, 1)
        out = out * se
        return self.act(out + residual)


class V5V6Model(nn.Module):
    def __init__(self, in_channels=224):
        super().__init__()
        self.in_channels = in_channels

        # CoordConv
        grid_y, grid_x = torch.meshgrid(
            torch.arange(4), torch.arange(9), indexing='ij'
        )
        grid_y = (grid_y.float() / 3.0) * 2.0 - 1.0
        grid_x = (grid_x.float() / 8.0) * 2.0 - 1.0
        self.register_buffer(
            'coord_grid',
            torch.stack([grid_y, grid_x], dim=0).unsqueeze(0)
        )

        self._stem = nn.Sequential(
            nn.Conv2d(in_channels + 2, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.Mish(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.Mish(inplace=True),
        )

        self._policy_blocks = nn.ModuleList([
            SEResBlock(256) for _ in range(12)
        ])

        # Policy Head: 从局部特征直接映射到动作
        self._policy_conv = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.GroupNorm(8, 128),
            nn.Mish(inplace=True),
            nn.Conv2d(128, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.Mish(inplace=True),
        )
        # Flatten: 64 * 4 * 9 = 2304
        self._policy_fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2304, 512),
            nn.Mish(inplace=True),
            nn.Linear(512, 235)
        )

        self._value_blocks = nn.ModuleList([
            SEResBlock(256) for _ in range(8)
        ])
        self._value_pool = nn.AdaptiveAvgPool2d(1)

        # Value Head
        self._value_fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.Mish(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(512, 1)
        )

        # Aux1: 胡牌概率 (0~1)
        self._aux_win_prob = nn.Sequential(
            nn.Linear(256, 128),
            nn.Mish(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # Aux2: 对手动作预测 (5分类)
        self._aux_opp_action = nn.Sequential(
            nn.Linear(256, 128),
            nn.Mish(inplace=True),
            nn.Linear(128, 5)
        )
        # Aux3: 局面阶段分类 (4分类)
        self._aux_phase = nn.Sequential(
            nn.Linear(256, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 4)
        )
        # Aux4: 上听数回归 (0~8)
        self._aux_shanten = nn.Sequential(
            nn.Linear(256, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 1)
        )

        self._apply_init()

    def _apply_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 策略头极小化
        nn.init.orthogonal_(self._policy_fc[-1].weight, gain=0.01)
        nn.init.constant_(self._policy_fc[-1].bias, 0)
        # 价值头适中
        nn.init.orthogonal_(self._value_fc[-1].weight, gain=1.0)
        nn.init.constant_(self._value_fc[-1].bias, 0)

    def forward(self, input_dict, mode='both', return_aux=False):
        """
        Args:
            input_dict: {'observation': (B,C,4,9), 'action_mask': (B,235)}
            mode: 'policy' | 'value' | 'both'
            return_aux: 是否返回辅助任务输出（仅在training=True时有效）
        """
        obs = input_dict["observation"].float()
        batch_size = obs.size(0)

        # Shared Stem
        coords = self.coord_grid.expand(batch_size, -1, -1, -1)
        x = torch.cat([obs, coords], dim=1)
        shared = self._stem(x)

        outputs = {}

        # Policy Tower
        if mode in ('policy', 'both'):
            p = shared
            for block in self._policy_blocks:
                p = block(p)

            p_conv = self._policy_conv(p)
            p_flat = p_conv.view(batch_size, -1)
            logits = self._policy_fc(p_flat)

            # Masking
            mask = input_dict["action_mask"].float()
            masked_logits = torch.where(
                mask > 0.5,
                logits,
                torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
            )
            outputs['logits'] = masked_logits

        # Value Tower
        if mode in ('value', 'both'):
            v = shared
            for block in self._value_blocks:
                v = block(v)

            v_pool = self._value_pool(v).view(batch_size, -1)
            value = self._value_fc(v_pool)
            outputs['value'] = value

            # Auxiliary tasks
            if return_aux and self.training:
                outputs['aux'] = {
                    'win_prob': self._aux_win_prob(v_pool),
                    'opp_action': self._aux_opp_action(v_pool),
                    'phase': self._aux_phase(v_pool),
                    'shanten': self._aux_shanten(v_pool),
                }

        if mode == 'policy':
            return outputs['logits']
        elif mode == 'value':
            return outputs['value']
        elif return_aux and self.training and 'aux' in outputs:
            return outputs['logits'], outputs['value'], outputs['aux']
        else:
            return outputs['logits'], outputs['value']


# Botzone 兼容别名
CNNModel = V5V6Model
