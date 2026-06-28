"""国标麻将 CNN 模型 — 16通道"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def _safe_linear_forward(x, layer):
    # Work around MKLDNN primitive descriptor errors on aarch64.
    try:
        return layer(x)
    except RuntimeError as e:
        if x.device.type == 'cpu' and 'primitive descriptor' in str(e):
            return F.linear(x, layer.weight, layer.bias)
        raise e

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        # 使用 Sequential 保持 state_dict 键值严格对齐加载
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.Mish(inplace=True), 
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        # Guard SEBlock Linear layers against MKLDNN crashes
        y = _safe_linear_forward(y, self.fc[0])
        y = self.fc[1](y)
        y = _safe_linear_forward(y, self.fc[2])
        y = self.fc[3](y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    """ResBlock + GroupNorm + Mish + SE"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, channels)
        self.act = nn.Mish(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(4, channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.act(out)

class CNNModel(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.in_channels = in_channels
        hidden_channels = 64
        
        # CoordConv
        grid_y, grid_x = torch.meshgrid(torch.arange(4), torch.arange(9), indexing='ij')
        grid_y = (grid_y.float() / 3.0) * 2.0 - 1.0
        grid_x = (grid_x.float() / 8.0) * 2.0 - 1.0
        self.register_buffer('coord_grid', torch.stack([grid_y, grid_x], dim=0).unsqueeze(0))

        self._conv_init = nn.Conv2d(in_channels + 2, hidden_channels, kernel_size=3, padding=1, bias=False)
        self._gn_init = nn.GroupNorm(4, hidden_channels)
        self._act = nn.Mish(inplace=True)

        self._res_blocks = nn.Sequential(
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels)
        )
        
        self._conv_reduce = nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False)
        self._gn_reduce = nn.GroupNorm(4, 32)
        self._flatten = nn.Flatten()

        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 235)
        )

        self._value_branch = nn.ModuleList([
            nn.Linear(32 * 4 * 9, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 1)
        ])

        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        nn.init.orthogonal_(self._logits[-1].weight, gain=0.01)
        nn.init.constant_(self._logits[-1].bias, 0)

        nn.init.orthogonal_(self._value_branch[-1].weight, gain=1.0)
        nn.init.constant_(self._value_branch[-1].bias, 0)

    def forward(self, input_dict: dict):
        obs = input_dict["observation"].float()
        batch_size = obs.size(0)
        
        coords = self.coord_grid.expand(batch_size, -1, -1, -1)
        obs_with_coords = torch.cat([obs, coords], dim=1)

        hidden = self._conv_init(obs_with_coords)
        hidden = self._gn_init(hidden)
        hidden = self._act(hidden)
        hidden = self._res_blocks(hidden)
        
        hidden = self._conv_reduce(hidden)
        hidden = self._gn_reduce(hidden)
        hidden = self._act(hidden)
        hidden = self._flatten(hidden)

        x_logits = hidden
        for layer in self._logits:
            if isinstance(layer, nn.Linear):
                x_logits = _safe_linear_forward(x_logits, layer)
            else:
                x_logits = layer(x_logits)
        logits = x_logits

        mask = input_dict["action_mask"].float()
        masked_logits = torch.where(
            mask > 0.5, 
            logits, 
            torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
        )

        v_hidden = _safe_linear_forward(hidden, self._value_branch[0])
        v_hidden = self._value_branch[1](v_hidden)
        value = _safe_linear_forward(v_hidden, self._value_branch[2])

        return masked_logits, value